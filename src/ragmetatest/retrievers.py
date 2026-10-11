"""Persistent retriever adapters used by both baselines and RAGMetaTest."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from rank_bm25 import BM25Okapi

from ragmetatest.chunking import chunk_document
from ragmetatest.types import Chunk, Document, RetrievalHit


def safe_collection_name(name: str, *, limit: int = 60) -> str:
    """Return a Chroma/Qdrant-compatible deterministic collection name."""

    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", name)
    sanitized = sanitized[:limit].strip("-._")
    if len(sanitized) < 3:
        sanitized = f"rag-{sanitized}".strip("-._")
    return sanitized


class MutableRetriever(ABC):
    name: str

    def __init__(self, *, chunk_size: int = 256, chunk_overlap: int = 20):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    def build(self, documents: Iterable[Document]) -> None:
        self.reset()
        for document in documents:
            self.add_document(document)

    def add_document(self, document: Document, *, id_suffix: str = "") -> None:
        self.add_chunks(
            chunk_document(
                document,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                id_suffix=id_suffix,
            )
        )

    @abstractmethod
    def add_chunks(
        self,
        chunks: Iterable[Chunk],
        *,
        embedding_texts: Iterable[str] | None = None,
    ) -> None:
        """Add materialized chunks, optionally embedding alternate text."""

        raise NotImplementedError

    @abstractmethod
    def delete_document(self, doc_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete_chunks(self, chunk_ids: Iterable[str]) -> None:
        raise NotImplementedError

    def replace_document(
        self,
        document: Document,
        *,
        delete_existing: bool = True,
        id_suffix: str = "",
    ) -> None:
        if delete_existing:
            self.delete_document(document.doc_id)
        self.add_document(document, id_suffix=id_suffix)

    @abstractmethod
    def search(self, query: str, *, top_k: int = 1) -> list[RetrievalHit]:
        raise NotImplementedError

    @abstractmethod
    def all_chunks(self) -> list[Chunk]:
        raise NotImplementedError


class BM25Retriever(MutableRetriever):
    """Mutable lexical control with the same chunks as the dense retriever."""

    name = "bm25"

    def __init__(self, *, chunk_size: int = 256, chunk_overlap: int = 20):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._chunks: dict[str, Chunk] = {}
        self._ordered: list[Chunk] = []
        self._index: BM25Okapi | None = None

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return text.casefold().split()

    def _reindex(self) -> None:
        self._ordered = sorted(self._chunks.values(), key=lambda item: item.chunk_id)
        self._index = (
            BM25Okapi([self._tokens(chunk.text) for chunk in self._ordered])
            if self._ordered
            else None
        )

    def reset(self) -> None:
        self._chunks.clear()
        self._reindex()

    def add_chunks(
        self,
        chunks: Iterable[Chunk],
        *,
        embedding_texts: Iterable[str] | None = None,
    ) -> None:
        del embedding_texts
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
        self._reindex()

    def delete_document(self, doc_id: str) -> None:
        self._chunks = {
            chunk_id: chunk
            for chunk_id, chunk in self._chunks.items()
            if chunk.doc_id != doc_id
        }
        self._reindex()

    def delete_chunks(self, chunk_ids: Iterable[str]) -> None:
        for chunk_id in chunk_ids:
            self._chunks.pop(chunk_id, None)
        self._reindex()

    def search(self, query: str, *, top_k: int = 1) -> list[RetrievalHit]:
        if not self._index or not self._ordered or top_k <= 0:
            return []
        scores = np.asarray(self._index.get_scores(self._tokens(query)), dtype=float)
        order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))[:top_k]
        return [
            RetrievalHit(chunk=self._ordered[index], score=float(scores[index]), rank=rank)
            for rank, index in enumerate(order, start=1)
        ]

    def all_chunks(self) -> list[Chunk]:
        return sorted(self._chunks.values(), key=lambda item: item.chunk_id)


class ContrieverEncoder:
    """Thin adapter around the Contriever implementation already in ragtest."""

    def __init__(self, *, device: str = "cpu", batch_size: int = 32):
        import torch
        from sentinelrag.utils.embeddings import load_models

        model, _, tokenizer, get_emb = load_models("contriever")
        self._torch = torch
        self.model = model.eval().to(device)
        self.tokenizer = tokenizer
        self.get_emb = get_emb
        self.device = device
        self.batch_size = batch_size
        self._cache: dict[str, list[float]] = {}

    @property
    def dimension(self) -> int:
        return int(self.model.config.hidden_size)

    def encode(self, texts: list[str]) -> list[list[float]]:
        missing = list(dict.fromkeys(text for text in texts if text not in self._cache))
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self._torch.no_grad():
                vectors = self.get_emb(self.model, inputs)
            if vectors.ndim == 1:
                vectors = vectors.unsqueeze(0)
            for text, vector in zip(batch, vectors.cpu().tolist(), strict=True):
                self._cache[text] = vector
        return [self._cache[text] for text in texts]


class ChromaContrieverRetriever(MutableRetriever):
    """Persistent ChromaDB retriever using ragtest's Contriever model path."""

    name = "chroma-contriever"

    def __init__(
        self,
        *,
        persist_dir: str | Path,
        collection_name: str,
        encoder: ContrieverEncoder,
        chunk_size: int = 256,
        chunk_overlap: int = 20,
    ):
        import chromadb

        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.encoder = encoder
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self._get_or_create()

    def _get_or_create(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        collection_names = {item.name for item in self.client.list_collections()}
        if self.collection_name in collection_names:
            self.client.delete_collection(self.collection_name)
        self.collection = self._get_or_create()

    def add_chunks(
        self,
        chunks: Iterable[Chunk],
        *,
        embedding_texts: Iterable[str] | None = None,
    ) -> None:
        chunks = list(chunks)
        if not chunks:
            return
        texts = [chunk.text for chunk in chunks]
        vector_inputs = list(embedding_texts) if embedding_texts is not None else texts
        if len(vector_inputs) != len(chunks):
            raise ValueError("embedding_texts must align with chunks")
        self.collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=texts,
            embeddings=self.encoder.encode(vector_inputs),
            metadatas=[
                {
                    "doc_id": chunk.doc_id,
                    "version": chunk.version,
                    "chunk_index": chunk.chunk_index,
                    "content_hash": chunk.content_hash,
                    "embedding_input_hash": chunk.embedding_input_hash,
                    "embedding_version": chunk.embedding_version,
                    "chunker_version": chunk.chunker_version,
                }
                for chunk in chunks
            ],
        )

    def delete_document(self, doc_id: str) -> None:
        self.collection.delete(where={"doc_id": doc_id})

    def delete_chunks(self, chunk_ids: Iterable[str]) -> None:
        ids = list(chunk_ids)
        if ids:
            self.collection.delete(ids=ids)

    @staticmethod
    def _chunk_from_row(
        chunk_id: str,
        document: str,
        metadata: dict,
    ) -> Chunk:
        return Chunk(
            chunk_id=chunk_id,
            doc_id=str(metadata["doc_id"]),
            version=int(metadata["version"]),
            chunk_index=int(metadata["chunk_index"]),
            text=document,
            content_hash=str(metadata["content_hash"]),
            embedding_input_hash=str(
                metadata.get("embedding_input_hash", metadata["content_hash"])
            ),
            embedding_version=str(metadata.get("embedding_version", "contriever-v1")),
            chunker_version=str(metadata.get("chunker_version", "words-256-20")),
        )

    def search(self, query: str, *, top_k: int = 1) -> list[RetrievalHit]:
        if self.collection.count() == 0 or top_k <= 0:
            return []
        result = self.collection.query(
            query_embeddings=self.encoder.encode([query]),
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for rank, (chunk_id, document, metadata, distance) in enumerate(
            zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True,
            ),
            start=1,
        ):
            hits.append(
                RetrievalHit(
                    chunk=self._chunk_from_row(chunk_id, document, metadata),
                    score=1.0 - float(distance),
                    rank=rank,
                )
            )
        return hits

    def all_chunks(self) -> list[Chunk]:
        result = self.collection.get(include=["documents", "metadatas"])
        return sorted(
            (
                self._chunk_from_row(chunk_id, document, metadata)
                for chunk_id, document, metadata in zip(
                    result["ids"],
                    result["documents"],
                    result["metadatas"],
                    strict=True,
                )
            ),
            key=lambda item: item.chunk_id,
        )


class QdrantContrieverRetriever(MutableRetriever):
    """Qdrant retriever with the same chunk and Contriever contracts as Chroma."""

    name = "qdrant-contriever"

    def __init__(
        self,
        *,
        collection_name: str,
        encoder: ContrieverEncoder,
        url: str | None = "http://localhost:6333",
        path: str | Path | None = None,
        client=None,
        chunk_size: int = 256,
        chunk_overlap: int = 20,
    ):
        from qdrant_client import QdrantClient

        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if client is not None:
            self.client = client
        elif path is not None:
            self.client = QdrantClient(path=str(path))
        elif url is not None:
            self.client = QdrantClient(url=url)
        else:
            self.client = QdrantClient(location=":memory:")
        self.collection_name = collection_name
        self.encoder = encoder
        self._get_or_create()

    def _get_or_create(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.encoder.dimension,
                    distance=Distance.COSINE,
                ),
            )

    def reset(self) -> None:
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self._get_or_create()

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        return str(uuid5(NAMESPACE_URL, f"ragmetatest:{chunk_id}"))

    @staticmethod
    def _payload(chunk: Chunk) -> dict:
        return {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "version": chunk.version,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "content_hash": chunk.content_hash,
            "embedding_input_hash": chunk.embedding_input_hash,
            "embedding_version": chunk.embedding_version,
            "chunker_version": chunk.chunker_version,
        }

    @staticmethod
    def _chunk_from_payload(payload: dict) -> Chunk:
        return Chunk(
            chunk_id=str(payload["chunk_id"]),
            doc_id=str(payload["doc_id"]),
            version=int(payload["version"]),
            chunk_index=int(payload["chunk_index"]),
            text=str(payload["text"]),
            content_hash=str(payload["content_hash"]),
            embedding_input_hash=str(
                payload.get("embedding_input_hash", payload["content_hash"])
            ),
            embedding_version=str(payload.get("embedding_version", "contriever-v1")),
            chunker_version=str(payload.get("chunker_version", "words-256-20")),
        )

    def add_chunks(
        self,
        chunks: Iterable[Chunk],
        *,
        embedding_texts: Iterable[str] | None = None,
    ) -> None:
        from qdrant_client.models import PointStruct

        chunks = list(chunks)
        if not chunks:
            return
        texts = [chunk.text for chunk in chunks]
        vector_inputs = list(embedding_texts) if embedding_texts is not None else texts
        if len(vector_inputs) != len(chunks):
            raise ValueError("embedding_texts must align with chunks")
        vectors = self.encoder.encode(vector_inputs)
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=self._point_id(chunk.chunk_id),
                    vector=vector,
                    payload=self._payload(chunk),
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
            wait=True,
        )

    def delete_document(self, doc_id: str) -> None:
        from qdrant_client.models import (
            FieldCondition,
            Filter,
            FilterSelector,
            MatchValue,
        )

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
                    ]
                )
            ),
            wait=True,
        )

    def delete_chunks(self, chunk_ids: Iterable[str]) -> None:
        ids = [self._point_id(chunk_id) for chunk_id in chunk_ids]
        if ids:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=ids,
                wait=True,
            )

    def search(self, query: str, *, top_k: int = 1) -> list[RetrievalHit]:
        if top_k <= 0 or not self.all_chunks():
            return []
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=self.encoder.encode([query])[0],
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        return [
            RetrievalHit(
                chunk=self._chunk_from_payload(point.payload or {}),
                score=float(point.score),
                rank=rank,
            )
            for rank, point in enumerate(response.points, start=1)
        ]

    def all_chunks(self) -> list[Chunk]:
        chunks: list[Chunk] = []
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            chunks.extend(
                self._chunk_from_payload(record.payload or {}) for record in records
            )
            if offset is None:
                break
        return sorted(chunks, key=lambda item: item.chunk_id)

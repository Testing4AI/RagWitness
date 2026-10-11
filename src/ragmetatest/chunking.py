"""Paper-compatible word chunking with stable, inspectable identities."""

from __future__ import annotations

import hashlib

from ragmetatest.types import Chunk, Document


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_document(
    document: Document,
    *,
    chunk_size: int = 256,
    chunk_overlap: int = 20,
    id_suffix: str = "",
) -> list[Chunk]:
    """Split on whitespace like the WKC artifact's ``Corpus.rechunk`` method."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")

    words = document.text.split()
    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        text = " ".join(words[start:end])
        digest = content_hash(text)
        chunk_id = (
            f"{document.doc_id}@v{document.version}#c{index}"
            f"-{digest[:12]}{id_suffix}"
        )
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=document.doc_id,
                version=document.version,
                chunk_index=index,
                text=text,
                content_hash=digest,
                embedding_input_hash=digest,
                embedding_version=str(
                    document.metadata.get("embedding_version", "contriever-v1")
                ),
                chunker_version=str(
                    document.metadata.get(
                        "chunker_version", f"words-{chunk_size}-{chunk_overlap}"
                    )
                ),
            )
        )
        if end >= len(words):
            break
        start = end - chunk_overlap
        index += 1
    return chunks


def chunk_corpus(
    documents: list[Document],
    *,
    chunk_size: int = 256,
    chunk_overlap: int = 20,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in sorted(documents, key=lambda item: item.doc_id):
        chunks.extend(
            chunk_document(
                document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return chunks

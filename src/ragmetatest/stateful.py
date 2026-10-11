"""Stateful RAG execution, fault injection, auditing, and trace comparison."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

from ragmetatest.chunking import chunk_corpus, chunk_document, content_hash
from ragmetatest.retrievers import MutableRetriever
from ragmetatest.types import (
    AuditResult,
    Document,
    Event,
    EventResult,
    NormalizedState,
    QueryObservation,
    StateComparison,
)


@dataclass(frozen=True)
class FaultProfile:
    """One intentionally seeded maintenance bug."""

    name: str = "clean"

    @property
    def enabled(self) -> bool:
        return self.name != "clean"


class StatefulRAG:
    """A persistent retriever plus source state and an observable query cache."""

    def __init__(
        self,
        retriever: MutableRetriever,
        *,
        fault: FaultProfile | None = None,
        top_k: int = 1,
    ):
        self.retriever = retriever
        self.fault = fault or FaultProfile()
        self.top_k = top_k
        self.source_documents: dict[str, Document] = {}
        self.query_cache: dict[str, QueryObservation] = {}
        self._query_cache_top_k: dict[str, int] = {}
        self.event_results: list[EventResult] = []
        self.applied_event_ids: set[str] = set()
        self._event_counter = 0

    def build(self, documents: Iterable[Document]) -> None:
        docs = list(documents)
        self.source_documents = {document.doc_id: document for document in docs}
        self.retriever.build(docs)
        self.query_cache.clear()
        self._query_cache_top_k.clear()
        self.event_results.clear()
        self.applied_event_ids.clear()
        self._event_counter = 0

    def _invalidate_cache(self) -> None:
        if self.fault.name != "stale_query_cache":
            self.query_cache.clear()
            self._query_cache_top_k.clear()

    def _chunks(self, document: Document):
        return chunk_document(
            document,
            chunk_size=self.retriever.chunk_size,
            chunk_overlap=self.retriever.chunk_overlap,
        )

    def _replace_with_chunks(
        self,
        document: Document,
        chunks,
        *,
        embedding_texts=None,
        delete_existing: bool = True,
    ) -> None:
        if delete_existing:
            self.retriever.delete_document(document.doc_id)
        self.retriever.add_chunks(chunks, embedding_texts=embedding_texts)

    def _record(self, event: Event, detail: str, acknowledged: bool = True) -> EventResult:
        result = EventResult(
            event_id=event.event_id,
            action=event.action,
            acknowledged=acknowledged,
            detail=detail,
        )
        self.event_results.append(result)
        if acknowledged:
            self.applied_event_ids.add(event.event_id)
        return result

    def apply(self, event: Event) -> EventResult | QueryObservation:
        self._event_counter += 1

        if event.action in {"warm_cache", "query"}:
            if not event.question:
                raise ValueError(f"{event.action} requires question")
            observation = self.query(event.question, top_k=self.top_k, use_cache=True)
            self._record(event, f"query cache_hit={observation.cache_hit}")
            return observation

        if event.action == "replay":
            if not event.document:
                raise ValueError("replay requires document")
            if self.fault.name == "non_idempotent_replay":
                suffix = f"-replay-{self._event_counter}"
                self.retriever.add_document(event.document, id_suffix=suffix)
                self._invalidate_cache()
                return self._record(event, "replayed with duplicate physical IDs")
            if event.replay_of in self.applied_event_ids:
                return self._record(event, "duplicate event ignored idempotently")
            self.retriever.replace_document(event.document)
            self.source_documents[event.document.doc_id] = event.document
            self._invalidate_cache()
            return self._record(event, "replay applied because original event was unknown")

        if event.action in {"insert", "update", "rollback"}:
            if not event.document:
                raise ValueError(f"{event.action} requires document")
            document = event.document
            previous = self.source_documents.get(document.doc_id)

            # Correct behavior rejects an older event that arrives after a
            # newer committed source version. The seeded fault applies it to
            # the index while preserving the source-of-truth version.
            if (
                event.action == "update"
                and previous is not None
                and document.version < previous.version
            ):
                if self.fault.name == "out_of_order_update":
                    self.retriever.replace_document(document)
                    self._invalidate_cache()
                    return self._record(event, "stale version overwrote downstream index")
                return self._record(event, "stale version ignored")

            # The logical source changes even when the downstream indexing event
            # is dropped. This models a connector/indexer propagation failure.
            self.source_documents[document.doc_id] = document
            if self.fault.name == "dropped_update" and event.action in {"update", "rollback"}:
                self._invalidate_cache()
                return self._record(event, "source updated; downstream update silently dropped")

            chunks = self._chunks(document)
            if self.fault.name == "partial_rechunk" and event.action == "update":
                missing_index = len(chunks) // 2
                partial = [
                    chunk
                    for index, chunk in enumerate(chunks)
                    if index != missing_index
                ]
                self._replace_with_chunks(document, partial)
                self._invalidate_cache()
                return self._record(event, "only a prefix/suffix of new chunks was indexed")

            if (
                self.fault.name == "stale_embedding_vector"
                and event.action == "update"
                and previous is not None
            ):
                old_chunks = self._chunks(previous)
                embedding_texts = [
                    old_chunks[min(index, len(old_chunks) - 1)].text
                    for index in range(len(chunks))
                ]
                stale_chunks = [
                    replace(
                        chunk,
                        embedding_input_hash=content_hash(embedding_text),
                        embedding_version="contriever-stale-v1",
                    )
                    for chunk, embedding_text in zip(
                        chunks, embedding_texts, strict=True
                    )
                ]
                self._replace_with_chunks(
                    document,
                    stale_chunks,
                    embedding_texts=embedding_texts,
                )
                self._invalidate_cache()
                return self._record(event, "new chunks stored with stale embeddings")

            if (
                self.fault.name == "metadata_vector_skew"
                and event.action == "update"
                and previous is not None
            ):
                old_chunks = self._chunks(previous)
                skewed = []
                for index, chunk in enumerate(chunks):
                    old = old_chunks[min(index, len(old_chunks) - 1)]
                    skewed.append(
                        replace(
                            chunk,
                            text=old.text,
                            content_hash=old.content_hash,
                            embedding_input_hash=old.content_hash,
                        )
                    )
                self._replace_with_chunks(
                    document,
                    skewed,
                    embedding_texts=[chunk.text for chunk in skewed],
                )
                self._invalidate_cache()
                return self._record(event, "new version metadata paired with old content/vector")

            if (
                self.fault.name == "mixed_version_retrieval"
                and event.action == "update"
                and previous is not None
            ):
                self._replace_with_chunks(document, chunks)
                old_chunks = self._chunks(previous)
                if old_chunks:
                    self.retriever.add_chunks([old_chunks[len(old_chunks) // 2]])
                self._invalidate_cache()
                return self._record(event, "one old-version chunk survived beside new chunks")

            keep_existing = (
                self.fault.name == "orphan_old_chunks" and event.action == "update"
            ) or (
                self.fault.name == "rollback_leaves_new_chunks"
                and event.action == "rollback"
            )
            self._replace_with_chunks(
                document,
                chunks,
                delete_existing=not keep_existing,
            )
            self._invalidate_cache()
            return self._record(
                event,
                "document indexed"
                if not keep_existing
                else "document appended without removing prior chunks",
            )

        if event.action == "delete":
            if not event.doc_id:
                raise ValueError("delete requires doc_id")
            self.source_documents.pop(event.doc_id, None)
            if self.fault.name == "dropped_delete":
                self._invalidate_cache()
                return self._record(event, "source deleted; downstream delete silently dropped")
            if self.fault.name == "partial_delete":
                chunks = [
                    chunk
                    for chunk in self.retriever.all_chunks()
                    if chunk.doc_id == event.doc_id
                ]
                self.retriever.delete_chunks(
                    chunk.chunk_id for chunk in chunks[: max(1, len(chunks) // 2)]
                )
                self._invalidate_cache()
                return self._record(event, "only part of the document chunks were deleted")
            self.retriever.delete_document(event.doc_id)
            self._invalidate_cache()
            return self._record(event, "document removed")

        if event.action == "rename":
            if not event.doc_id or not event.new_doc_id:
                raise ValueError("rename requires doc_id and new_doc_id")
            old = self.source_documents.pop(event.doc_id)
            renamed = Document(
                doc_id=event.new_doc_id,
                version=old.version,
                text=old.text,
                metadata=old.metadata,
            )
            self.source_documents[event.new_doc_id] = renamed
            if self.fault.name != "rename_leaves_old_chunks":
                self.retriever.delete_document(event.doc_id)
            if self.fault.name != "rename_drops_new_document":
                self.retriever.add_document(renamed)
            self._invalidate_cache()
            return self._record(event, "document renamed")

        raise ValueError(f"Unsupported event action: {event.action}")

    def query(self, question: str, *, top_k: int | None = None, use_cache: bool = True) -> QueryObservation:
        top_k = self.top_k if top_k is None else top_k
        if use_cache and question in self.query_cache and self._query_cache_top_k.get(question) == top_k:
            cached = self.query_cache[question]
            return QueryObservation(
                question=question,
                hits=cached.hits,
                cache_hit=True,
                answer=cached.answer,
            )
        observation = QueryObservation(
            question=question,
            hits=tuple(self.retriever.search(question, top_k=top_k)),
            cache_hit=False,
        )
        if use_cache:
            self.query_cache[question] = observation
            self._query_cache_top_k[question] = top_k
        return observation

    def audit(self) -> AuditResult:
        expected = chunk_corpus(
            list(self.source_documents.values()),
            chunk_size=self.retriever.chunk_size,
            chunk_overlap=self.retriever.chunk_overlap,
        )
        actual = self.retriever.all_chunks()
        expected_ids = {chunk.chunk_id for chunk in expected}
        actual_ids = {chunk.chunk_id for chunk in actual}
        expected_versions = {
            (chunk.doc_id, chunk.version, chunk.content_hash) for chunk in expected
        }
        actual_versions = {
            (chunk.doc_id, chunk.version, chunk.content_hash) for chunk in actual
        }
        expected_content = {
            (
                chunk.chunk_id,
                chunk.doc_id,
                chunk.version,
                chunk.chunk_index,
                chunk.content_hash,
                chunk.embedding_input_hash,
                chunk.embedding_version,
                chunk.chunker_version,
            )
            for chunk in expected
        }
        actual_content = {
            (
                chunk.chunk_id,
                chunk.doc_id,
                chunk.version,
                chunk.chunk_index,
                chunk.content_hash,
                chunk.embedding_input_hash,
                chunk.embedding_version,
                chunk.chunker_version,
            )
            for chunk in actual
        }
        logical_keys = [
            (chunk.doc_id, chunk.version, chunk.chunk_index) for chunk in actual
        ]
        cache_coherent = all(
            evidence_signature(cached)
            == evidence_signature(
                QueryObservation(
                    question=question,
                    hits=tuple(self.retriever.search(question, top_k=self._query_cache_top_k.get(question, self.top_k))),
                )
            )
            for question, cached in self.query_cache.items()
        )
        return AuditResult(
            status_ok=all(result.acknowledged for result in self.event_results),
            count_ok=len(expected) == len(actual),
            exact_ids_ok=expected_ids == actual_ids,
            versions_ok=expected_versions == actual_versions,
            missing_ids=tuple(sorted(expected_ids - actual_ids)),
            extra_ids=tuple(sorted(actual_ids - expected_ids)),
            content_ok=expected_content == actual_content,
            embedding_lineage_ok=all(
                chunk.embedding_input_hash == chunk.content_hash for chunk in actual
            ),
            duplicate_logical_chunks_ok=len(logical_keys) == len(set(logical_keys)),
            cache_coherent=cache_coherent,
        )


def evidence_signature(observation: QueryObservation) -> tuple[tuple[str, int, str], ...]:
    return tuple(
        (hit.chunk.doc_id, hit.chunk.version, hit.chunk.content_hash)
        for hit in observation.hits
    )


def history_independent(
    clean: QueryObservation,
    incremental: QueryObservation,
) -> bool:
    """Strict top-k evidence equivalence for deterministic controlled runs."""

    return evidence_signature(clean) == evidence_signature(incremental)


def normalized_state(system: StatefulRAG) -> NormalizedState:
    source_documents = tuple(
        sorted(
            (
                document.doc_id,
                document.version,
                content_hash(document.text),
            )
            for document in system.source_documents.values()
        )
    )
    chunks = tuple(
        sorted(
            (
                chunk.chunk_id,
                chunk.doc_id,
                chunk.version,
                chunk.chunk_index,
                chunk.content_hash,
                chunk.embedding_input_hash,
                chunk.embedding_version,
                chunk.chunker_version,
            )
            for chunk in system.retriever.all_chunks()
        )
    )
    cached_evidence = tuple(
        sorted(
            (question, evidence_signature(observation))
            for question, observation in system.query_cache.items()
        )
    )
    return NormalizedState(
        source_documents=source_documents,
        chunks=chunks,
        cached_evidence=cached_evidence,
    )


def compare_states(clean: StatefulRAG, incremental: StatefulRAG) -> StateComparison:
    clean_state = normalized_state(clean)
    incremental_state = normalized_state(incremental)
    return StateComparison(
        logical_source_equal=(
            clean_state.source_documents == incremental_state.source_documents
        ),
        index_equal=clean_state.chunks == incremental_state.chunks,
        cache_coherent=incremental.audit().cache_coherent,
    )

"""Shared immutable records for snapshot and stateful RAG experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class Document:
    doc_id: str
    version: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    version: int
    chunk_index: int
    text: str
    content_hash: str
    embedding_input_hash: str = ""
    embedding_version: str = "contriever-v1"
    chunker_version: str = "words-256-20"


@dataclass(frozen=True)
class RetrievalHit:
    chunk: Chunk
    score: float
    rank: int


@dataclass(frozen=True)
class FactCase:
    case_id: str
    doc_v1: Document
    doc_v2: Document
    question: str
    answer_v1: str
    answer_v2: str
    background_documents: tuple[Document, ...] = ()


Action = Literal[
    "insert",
    "update",
    "delete",
    "rename",
    "replay",
    "rollback",
    "warm_cache",
    "query",
]


@dataclass(frozen=True)
class Event:
    action: Action
    event_id: str
    doc_id: str | None = None
    document: Document | None = None
    new_doc_id: str | None = None
    question: str | None = None
    replay_of: str | None = None


@dataclass(frozen=True)
class EventResult:
    event_id: str
    action: Action
    acknowledged: bool
    detail: str


@dataclass(frozen=True)
class QueryObservation:
    question: str
    hits: tuple[RetrievalHit, ...]
    cache_hit: bool = False
    answer: str | None = None

    @property
    def top_chunk(self) -> Chunk | None:
        return self.hits[0].chunk if self.hits else None


@dataclass(frozen=True)
class AuditResult:
    status_ok: bool
    count_ok: bool
    exact_ids_ok: bool
    versions_ok: bool
    missing_ids: tuple[str, ...]
    extra_ids: tuple[str, ...]
    content_ok: bool = True
    embedding_lineage_ok: bool = True
    duplicate_logical_chunks_ok: bool = True
    cache_coherent: bool = True


@dataclass(frozen=True)
class NormalizedState:
    """Backend-independent representation of a settled RAG state."""

    source_documents: tuple[tuple[str, int, str], ...]
    chunks: tuple[tuple[str, str, int, int, str, str, str, str], ...]
    cached_evidence: tuple[
        tuple[str, tuple[tuple[str, int, str], ...]], ...
    ] = ()


@dataclass(frozen=True)
class StateComparison:
    logical_source_equal: bool
    index_equal: bool
    cache_coherent: bool

    @property
    def equivalent(self) -> bool:
        return self.logical_source_equal and self.index_equal and self.cache_coherent

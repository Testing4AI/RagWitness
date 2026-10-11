"""Continuous operation histories and signature-preserving supplemental checks.

This module does not modify the frozen pilot generator, oracle, or shrinker.
Generators are fault-agnostic and share the same per-document operation multiset.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace

from ragmetatest.history import reduce_source
from ragmetatest.retrievers import BM25Retriever
from ragmetatest.stateful import FaultProfile, StatefulRAG, evidence_signature
from ragmetatest.types import Document, Event, FactCase, QueryObservation

FAULTS = (
    "dropped_update", "orphan_old_chunks", "stale_query_cache",
    "non_idempotent_replay", "dropped_delete", "out_of_order_update",
)
STRATEGIES = ("relation_guided", "operation_random")
SEEDS = (901, 902, 903)
MUTATING_ACTIONS = frozenset({"insert", "update", "delete", "replay", "rollback", "rename"})


@dataclass(frozen=True, order=True)
class ViolationSignature:
    layer: str
    document_id: str
    mismatch_kind: str


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _source_state(documents):
    return tuple(sorted((d.doc_id, d.version, d.text) for d in documents))


def _question(document: Document) -> str:
    entity = document.metadata.get("entity", document.doc_id)
    attribute = document.metadata.get("attribute", "operational setting")
    return f"What is the current {attribute} for {entity}?"


def initial_documents(case: FactCase) -> list[Document]:
    documents = [case.doc_v1, *case.background_documents]
    if len(documents) != 4 or len({d.doc_id for d in documents}) != 4:
        raise ValueError("Continuous study requires four distinct initial documents")
    return documents


def _version(document: Document, version: int) -> Document:
    marker = f"Recorded operational revision {version} for {document.doc_id}."
    return replace(document, version=version, text=f"{document.text} {marker}")


def token_events(case: FactCase, seed: int) -> dict[str, list[Event]]:
    """Identical payload/action multiset for both scheduling strategies.

    Eight events per document: warm, newer update, query, acknowledged replay,
    rollback, delete, insert newer, and delayed old update. No padding events.
    """
    result = {}
    for document in initial_documents(case):
        prefix = f"{case.case_id}-{seed}-{document.doc_id}"
        newer, inserted = _version(document, 2), _version(document, 3)
        old = document
        update_id = f"{prefix}-newer"
        result[document.doc_id] = [
            Event("warm_cache", f"{prefix}-warm", doc_id=document.doc_id, question=_question(document)),
            Event("update", update_id, doc_id=document.doc_id, document=newer),
            Event("query", f"{prefix}-query", doc_id=document.doc_id, question=_question(document)),
            Event("replay", f"{prefix}-replay", doc_id=document.doc_id, document=newer, replay_of=update_id),
            Event("rollback", f"{prefix}-rollback", doc_id=document.doc_id, document=old),
            Event("delete", f"{prefix}-delete", doc_id=document.doc_id),
            Event("insert", f"{prefix}-insert", doc_id=document.doc_id, document=inserted),
            Event("update", f"{prefix}-delayed", doc_id=document.doc_id, document=old),
        ]
    return result


def legal_history(initial: list[Document], events: list[Event]) -> bool:
    present = {document.doc_id for document in initial}
    seen = set()
    for event in events:
        if event.event_id in seen:
            return False
        if event.action == "insert":
            if event.doc_id in present:
                return False
            present.add(event.doc_id)
        elif event.action in {"update", "rollback", "delete"}:
            if event.doc_id not in present:
                return False
            if event.action == "delete":
                present.remove(event.doc_id)
        elif event.action == "replay":
            if event.replay_of not in seen:
                return False
        elif event.action not in {"query", "warm_cache"}:
            return False
        seen.add(event.event_id)
    return True


def generate_continuous_history(case: FactCase, strategy: str, seed: int) -> list[Event]:
    rng = random.Random(f"{case.case_id}:{seed}:{strategy}")
    tokens = token_events(case, seed)
    initial = initial_documents(case)
    if strategy == "relation_guided":
        events = []
        # Interleave four documents on one continuous SUT; do not reset per relation.
        for position in range(8):
            order = list(tokens)
            rng.shuffle(order)
            events.extend(tokens[document_id][position] for document_id in order)
    elif strategy == "operation_random":
        remaining = [event for sequence in tokens.values() for event in sequence]
        events = []
        while remaining:
            choices = [i for i, event in enumerate(remaining) if legal_history(initial, [*events, event])]
            if not choices:
                raise ValueError("No legal operation remains")
            events.append(remaining.pop(rng.choice(choices)))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    if len(events) != 32 or not legal_history(initial, events):
        raise AssertionError("Generated history violates the frozen budget or legality")
    return events


def signatures(reference: StatefulRAG, sut: StatefulRAG, question_owners: dict[str, str],
               query_output: QueryObservation | None = None) -> set[ViolationSignature]:
    """Observe declared source/index/cache/evidence projections, without fault labels.

    This supplementary observer does not invoke the SUT's audit implementation.
    It does not claim raw-vector or metadata-value coverage.
    """
    found = set()
    expected_docs, actual_docs = reference.source_documents, sut.source_documents
    expected_chunks, actual_chunks = reference.retriever.all_chunks(), sut.retriever.all_chunks()
    doc_ids = set(expected_docs) | set(actual_docs) | {chunk.doc_id for chunk in actual_chunks}
    for document_id in sorted(doc_ids):
        expected_doc, actual_doc = expected_docs.get(document_id), actual_docs.get(document_id)
        if expected_doc != actual_doc:
            found.add(ViolationSignature("source", document_id, "source_record_mismatch"))
        expected = [chunk for chunk in expected_chunks if chunk.doc_id == document_id]
        actual = [chunk for chunk in actual_chunks if chunk.doc_id == document_id]
        if not expected and actual:
            found.add(ViolationSignature("index", document_id, "deleted_document_present"))
            continue
        expected_by_id = {chunk.chunk_id: chunk for chunk in expected}
        actual_by_id = {chunk.chunk_id: chunk for chunk in actual}
        if set(expected_by_id) - set(actual_by_id):
            found.add(ViolationSignature("index", document_id, "missing_chunk"))
        logical_counts = Counter((chunk.version, chunk.chunk_index) for chunk in actual)
        if any(count > 1 for count in logical_counts.values()):
            found.add(ViolationSignature("index", document_id, "duplicate_logical_chunk"))
        if expected_doc and any(chunk.version != expected_doc.version for chunk in actual):
            found.add(ViolationSignature("index", document_id, "obsolete_version"))
        for chunk_id in set(expected_by_id) & set(actual_by_id):
            if expected_by_id[chunk_id] != actual_by_id[chunk_id]:
                found.add(ViolationSignature("index", document_id, "chunk_payload_mismatch"))
        if set(actual_by_id) - set(expected_by_id) and not any(
            signature.document_id == document_id and signature.mismatch_kind in {
                "obsolete_version", "duplicate_logical_chunk", "deleted_document_present",
            } for signature in found
        ):
            found.add(ViolationSignature("index", document_id, "unexpected_chunk"))
    for question, cached in sut.query_cache.items():
        fresh = QueryObservation(question=question, hits=tuple(sut.retriever.search(question, top_k=sut.top_k)))
        if evidence_signature(cached) != evidence_signature(fresh):
            found.add(ViolationSignature("cache", question_owners[question], "stale_evidence"))
    if query_output is not None:
        expected = reference.query(query_output.question, top_k=sut.top_k, use_cache=False)
        if evidence_signature(expected) != evidence_signature(query_output):
            found.add(ViolationSignature("evidence", question_owners[query_output.question], "evidence_sequence_mismatch"))
    return found


def execute_history(initial: list[Document], events: list[Event], fault: str, *, top_k: int = 5,
                    stop_signature: ViolationSignature | None = None) -> dict:
    if not legal_history(initial, events):
        raise ValueError("Illegal history")
    owners = {_question(document): document.doc_id for document in initial}
    sut = StatefulRAG(BM25Retriever(), fault=FaultProfile(fault), top_k=top_k)
    start = time.perf_counter()
    sut.build(initial)
    checkpoints, observed = [], set()
    source_changes = 0
    previous = _source_state(initial)
    sut_seconds = time.perf_counter() - start
    reference_seconds = 0.0
    for position, event in enumerate(events, 1):
        start = time.perf_counter()
        output = sut.apply(event)
        sut_seconds += time.perf_counter() - start
        start = time.perf_counter()
        final = reduce_source(initial, events[:position])
        changed = _source_state(final) != previous
        previous = _source_state(final)
        source_changes += changed
        reference = StatefulRAG(BM25Retriever(), top_k=top_k)
        reference.build(final)
        current = signatures(reference, sut, owners, output if isinstance(output, QueryObservation) else None)
        reference_seconds += time.perf_counter() - start
        observed |= current
        checkpoints.append({"position": position, "event_id": event.event_id, "action": event.action,
                            "source_changed": changed, "signatures": [asdict(value) for value in sorted(current)]})
        if stop_signature is not None and stop_signature in current:
            break
    return {
        "event_count": len(events), "executed_event_count": len(checkpoints),
        "action_counts": dict(Counter(event.action for event in events)),
        "mutating_operations": sum(event.action in MUTATING_ACTIONS for event in events),
        "effective_source_changes": source_changes,
        "signatures": [asdict(value) for value in sorted(observed)],
        "detected": bool(observed),
        "terminal_detected": bool(checkpoints and checkpoints[-1]["signatures"]),
        "first_detection_event": next((r["position"] for r in checkpoints if r["signatures"]), None),
        "sut_seconds": sut_seconds, "reference_and_observation_seconds": reference_seconds,
        "checkpoints": checkpoints,
    }


def shrink_signature(initial: list[Document], events: list[Event], fault: str,
                     signature: ViolationSignature, *, top_k: int = 5) -> dict:
    """Greedy one-minimal history for an explicit layer/document/mismatch signature.

    The same signature must occur at some checkpoint. The checkpoint and terminal
    source state may change; no unrelated target/layer/mismatch can replace it.
    """
    evaluations = 0
    rejected_illegal = 0

    def retains(candidate):
        nonlocal evaluations, rejected_illegal
        if not candidate or not legal_history(initial, candidate):
            rejected_illegal += 1
            return False
        evaluations += 1
        result = execute_history(initial, candidate, fault, top_k=top_k, stop_signature=signature)
        return asdict(signature) in result["signatures"]

    if not retains(events):
        raise ValueError("Input does not exhibit the requested violation signature")
    start = time.perf_counter()
    current = list(events)
    while True:
        for index in range(len(current)):
            candidate = current[:index] + current[index + 1:]
            if retains(candidate):
                current = candidate
                break
        else:
            break
    # Independent final single-deletion pass explicitly verifies one-minimality.
    one_minimal = not any(retains(current[:i] + current[i + 1:]) for i in range(len(current)))
    result = execute_history(initial, current, fault, top_k=top_k)
    removed_ids = {event.event_id for event in events} - {event.event_id for event in current}
    return {
        "signature": asdict(signature), "signature_preserved": asdict(signature) in result["signatures"],
        "one_minimal": one_minimal, "original_events": len(events), "minimized_events": len(current),
        "original_mutating_operations": sum(event.action in MUTATING_ACTIONS for event in events),
        "minimized_mutating_operations": sum(event.action in MUTATING_ACTIONS for event in current),
        "removed_mutating_operations": sum(event.event_id in removed_ids and event.action in MUTATING_ACTIONS for event in events),
        "minimized_effective_source_changes": result["effective_source_changes"],
        "evaluations": evaluations, "rejected_illegal": rejected_illegal,
        "seconds": time.perf_counter() - start, "minimized_trace": [asdict(event) for event in current],
    }

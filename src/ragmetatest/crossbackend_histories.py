"""Transfer frozen continuous histories to real mutable dense retrievers.

The original observer is reused over a recorded view, so observations used for
the decision and the independent artifact audit are the same physical reads.
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict

from ragmetatest.chunking import content_hash
from ragmetatest.history import reduce_source
from ragmetatest.stateful import FaultProfile, StatefulRAG, evidence_signature
from ragmetatest.supplemental_histories import (
    MUTATING_ACTIONS,
    _question,
    _source_state,
    legal_history,
    signatures,
)
from ragmetatest.types import QueryObservation


class RecordedRetriever:
    def __init__(self, chunks, searches):
        self.chunks = chunks
        self.searches = searches

    def all_chunks(self):
        return self.chunks

    def search(self, question, *, top_k):
        return self.searches[(question, top_k)].hits


class RecordedState:
    def __init__(self, documents, chunks, searches, cache, top_k):
        self.source_documents = documents
        self.retriever = RecordedRetriever(chunks, searches)
        self.query_cache = cache
        self.top_k = top_k

    def query(self, question, *, top_k, use_cache):
        return self.retriever.searches[(question, top_k)]


def encoded_record(record):
    values = asdict(record)
    values["text_sha256"] = content_hash(values.pop("text"))
    return values


def observe(reference, sut, owners, query_output=None):
    ref_chunks, sut_chunks = reference.retriever.all_chunks(), sut.retriever.all_chunks()
    cache = dict(sut.query_cache)
    sut_searches = {
        (question, sut.top_k): QueryObservation(
            question=question, hits=tuple(sut.retriever.search(question, top_k=sut.top_k)))
        for question in cache
    }
    ref_searches = {}
    if query_output is not None:
        ref_searches[(query_output.question, sut.top_k)] = reference.query(
            query_output.question, top_k=sut.top_k, use_cache=False)
    ref_view = RecordedState(dict(reference.source_documents), ref_chunks, ref_searches, {}, sut.top_k)
    sut_view = RecordedState(dict(sut.source_documents), sut_chunks, sut_searches, cache, sut.top_k)
    found = [asdict(item) for item in sorted(signatures(ref_view, sut_view, owners, query_output))]
    raw = {
        "reference": {
            "sources": [encoded_record(doc) for _, doc in sorted(reference.source_documents.items())],
            "chunks": [encoded_record(chunk) for chunk in ref_chunks],
        },
        "observed": {
            "sources": [encoded_record(doc) for _, doc in sorted(sut.source_documents.items())],
            "chunks": [encoded_record(chunk) for chunk in sut_chunks],
        },
        "cache": [{"question": question, "owner": owners[question],
                   "cached": evidence_signature(cached),
                   "fresh": evidence_signature(sut_searches[(question, sut.top_k)])}
                  for question, cached in sorted(cache.items())],
        "query": None,
    }
    if query_output is not None:
        raw["query"] = {
            "question": query_output.question, "owner": owners[query_output.question],
            "expected": evidence_signature(ref_searches[(query_output.question, sut.top_k)]),
            "observed": evidence_signature(query_output),
        }
    canonical = {(c.doc_id, c.version, c.content_hash) for c in ref_chunks}
    invalid_evidence = any(tuple(hit) not in canonical for entry in raw["cache"] for hit in entry["cached"])
    if raw["query"] is not None:
        invalid_evidence |= any(tuple(hit) not in canonical for hit in raw["query"]["observed"])
    return {"signatures": found, "raw": raw,
            "structural_detected": any(item["layer"] in {"source", "index"} for item in found),
            "strict_cache_or_evidence_detected": any(item["layer"] in {"cache", "evidence"} for item in found),
            "noncanonical_evidence_detected": invalid_evidence}


def execute_dense_history(initial, events, fault, retrievers, *, top_k=5):
    """A new SUT instance per condition, one live incremental index per history."""
    if not legal_history(initial, events):
        raise ValueError("Illegal frozen history")
    start = time.perf_counter()
    clocks = {"sut_seconds": 0.0, "reference_build_seconds": 0.0,
              "observation_seconds": 0.0, "reducer_seconds": 0.0}
    checkpoints = []
    row = {"status": "error", "event_count": len(events), "top_k": top_k,
           "action_counts": dict(Counter(event.action for event in events)),
           "mutating_operations": sum(event.action in MUTATING_ACTIONS for event in events),
           "checkpoints": checkpoints}
    owners = {_question(document): document.doc_id for document in initial}
    previous = _source_state(initial)
    sut = StatefulRAG(retrievers[0], fault=FaultProfile(fault), top_k=top_k)
    try:
        tick = time.perf_counter()
        sut.build(initial)
        clocks["sut_seconds"] += time.perf_counter() - tick
        for position, event in enumerate(events, 1):
            tick = time.perf_counter()
            output = sut.apply(event)
            clocks["sut_seconds"] += time.perf_counter() - tick
            tick = time.perf_counter()
            terminal = reduce_source(initial, events[:position])
            changed = _source_state(terminal) != previous
            previous = _source_state(terminal)
            clocks["reducer_seconds"] += time.perf_counter() - tick
            tick = time.perf_counter()
            reference = StatefulRAG(retrievers[1], top_k=top_k)
            reference.build(terminal)
            reference_seconds = time.perf_counter() - tick
            clocks["reference_build_seconds"] += reference_seconds
            tick = time.perf_counter()
            observation = observe(reference, sut, owners, output if isinstance(output, QueryObservation) else None)
            observation_seconds = time.perf_counter() - tick
            clocks["observation_seconds"] += observation_seconds
            checkpoints.append({"position": position, "event_id": event.event_id, "action": event.action,
                                "source_changed": changed, "reference_build_seconds": reference_seconds,
                                "observation_seconds": observation_seconds, **observation})
        row["status"] = "complete"
        row["detected"] = any(point["signatures"] for point in checkpoints)
        row["terminal_detected"] = bool(checkpoints[-1]["signatures"])
        row["structural_detected"] = any(point["structural_detected"] for point in checkpoints)
        row["strict_cache_or_evidence_detected"] = any(point["strict_cache_or_evidence_detected"] for point in checkpoints)
        row["noncanonical_evidence_detected"] = any(point["noncanonical_evidence_detected"] for point in checkpoints)
        row["first_detection_event"] = next((point["position"] for point in checkpoints if point["signatures"]), None)
    except Exception as error:  # A failed infrastructure call is unknown, never a passing run.
        row["error"] = {"type": type(error).__name__, "message": str(error)}
        row["detected"] = row["terminal_detected"] = None
    row["effective_source_changes"] = sum(point["source_changed"] for point in checkpoints)
    row["completed_checkpoints"] = len(checkpoints)
    row.update(clocks)
    row["total_seconds"] = time.perf_counter() - start
    return row

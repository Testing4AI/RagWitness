"""Controlled stateful experiment runner with machine-readable artifacts."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from ragmetatest.chunking import content_hash
from ragmetatest.fixtures import versioned_fact_cases
from ragmetatest.llm import TargetLLM
from ragmetatest.retrievers import BM25Retriever, MutableRetriever
from ragmetatest.stateful import (
    FaultProfile,
    StatefulRAG,
    compare_states,
    evidence_signature,
    history_independent,
    normalized_state,
)
from ragmetatest.traces import (
    delete_trace,
    out_of_order_trace,
    rename_trace,
    replay_trace,
    rollback_trace,
    update_trace,
)
from ragmetatest.types import Document, Event, FactCase, QueryObservation

RetrieverFactory = Callable[[str], MutableRetriever]


def _final_documents(case: FactCase, trace_name: str):
    background = list(case.background_documents)
    if trace_name == "delete":
        return background
    if trace_name == "rollback":
        return [case.doc_v1, *background]
    if trace_name == "rename":
        return [
            Document(
                doc_id=f"{case.doc_v1.doc_id}-renamed",
                version=case.doc_v1.version,
                text=case.doc_v1.text,
                metadata=case.doc_v1.metadata,
            )
        ] + background
    return [case.doc_v2, *background]


def _events(case: FactCase, trace_name: str) -> list[Event]:
    if trace_name == "update":
        return update_trace(case)
    if trace_name == "warm_update":
        return update_trace(case, warm_cache=True)
    if trace_name == "replay":
        return replay_trace(case)
    if trace_name == "rollback":
        return rollback_trace(case)
    if trace_name == "delete":
        return delete_trace(case)
    if trace_name == "rename":
        return rename_trace(case)
    if trace_name == "out_of_order":
        return out_of_order_trace(case)
    raise ValueError(f"Unknown trace: {trace_name}")


def _last_observation(results: list[object]) -> QueryObservation:
    for result in reversed(results):
        if isinstance(result, QueryObservation):
            return result
    raise ValueError("Trace contains no query observation")


def _hit_rows(observation: QueryObservation) -> list[dict]:
    return [
        {
            "rank": hit.rank,
            "score": hit.score,
            "chunk_id": hit.chunk.chunk_id,
            "doc_id": hit.chunk.doc_id,
            "version": hit.chunk.version,
            "content_hash": hit.chunk.content_hash,
        }
        for hit in observation.hits
    ]


def _event_row(event: Event) -> dict:
    row = {
        "action": event.action,
        "event_id": event.event_id,
        "doc_id": event.doc_id,
        "new_doc_id": event.new_doc_id,
        "replay_of": event.replay_of,
        "question": event.question,
    }
    if event.document is not None:
        row["document"] = {
            "doc_id": event.document.doc_id,
            "version": event.document.version,
            "content_hash": content_hash(event.document.text),
        }
    else:
        row["document"] = None
    return row


def run_stateful_matrix(
    *,
    output_path: str | Path,
    case_limit: int = 4,
    retriever_factory: RetrieverFactory | None = None,
    retriever_name: str = "bm25",
    target_llms: list[TargetLLM] | None = None,
    top_k: int = 1,
    cases: list[FactCase] | None = None,
    resume: bool = False,
) -> list[dict]:
    if retriever_factory is None:
        retriever_factory = lambda _name: BM25Retriever()

    cases = cases if cases is not None else versioned_fact_cases(case_limit)
    target_llms = target_llms or []
    generated_cache: dict[tuple[str, str, tuple], tuple[str, dict]] = {}

    def generate(model: TargetLLM, question: str, hits):
        signature = tuple(
            (hit.chunk.chunk_id, hit.chunk.content_hash, hit.rank) for hit in hits
        )
        key = (model.preset.name, question, signature)
        if key not in generated_cache:
            generated_cache[key] = model.answer(question, list(hits))
        return generated_cache[key]
    trace_faults = {
        "update": [
            "clean",
            "dropped_update",
            "orphan_old_chunks",
            "partial_rechunk",
            "stale_embedding_vector",
            "metadata_vector_skew",
            "mixed_version_retrieval",
        ],
        "warm_update": ["clean", "stale_query_cache"],
        "replay": ["clean", "non_idempotent_replay"],
        "rollback": ["clean", "rollback_leaves_new_chunks"],
        "delete": ["clean", "dropped_delete", "partial_delete"],
        "rename": ["clean", "rename_leaves_old_chunks", "rename_drops_new_document"],
        "out_of_order": ["clean", "out_of_order_update"],
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = json.loads(path.read_text()) if resume and path.exists() else []
    completed = {
        (row["case_id"], row["trace"], row["fault"], row["retriever"], row["top_k"])
        for row in rows
    }

    for case in cases:
        for trace_name, faults in trace_faults.items():
            events = _events(case, trace_name)
            final_documents = _final_documents(case, trace_name)
            for fault_name in faults:
                key = (case.case_id, trace_name, fault_name, retriever_name, top_k)
                if key in completed:
                    continue
                started = time.perf_counter()
                incremental = StatefulRAG(
                    retriever_factory(
                        f"incremental-{case.case_id}-{trace_name}-{fault_name}"
                    ),
                    fault=FaultProfile(fault_name),
                    top_k=top_k,
                )
                incremental.build([case.doc_v1, *case.background_documents])
                old_observation = incremental.query(
                    case.question, top_k=top_k, use_cache=False
                )
                event_outputs = [incremental.apply(event) for event in events]
                observed = _last_observation(event_outputs)

                clean = StatefulRAG(
                    retriever_factory(
                        f"clean-{case.case_id}-{trace_name}-{fault_name}"
                    ),
                    top_k=top_k,
                )
                clean.build(final_documents)
                expected = clean.query(case.question, top_k=top_k, use_cache=False)
                audit = incremental.audit()
                equivalent = history_independent(expected, observed)
                state_comparison = compare_states(clean, incremental)
                wkc_applicable = trace_name in {"update", "warm_update", "delete"}
                # Evidence-level analogue of WKC's non-preserving output MR:
                # a relevant update/delete should change the observation or
                # cause abstention. This is deliberately generous to WKC.
                wkc_incremental_detected = (
                    wkc_applicable
                    and bool(observed.hits)
                    and evidence_signature(old_observation) == evidence_signature(observed)
                )
                state_relation_violated = not state_comparison.equivalent
                trace_mr_detected = (not equivalent) or state_relation_violated

                target_results = []
                for model in target_llms:
                    old_answer, old_usage = generate(
                        model, case.question, old_observation.hits
                    )
                    expected_answer, expected_usage = generate(
                        model, case.question, expected.hits
                    )
                    observed_answer, observed_usage = generate(
                        model, case.question, observed.hits
                    )
                    expected_normalized = expected_answer.strip().casefold()
                    observed_normalized = observed_answer.strip().casefold()
                    old_normalized = old_answer.strip().casefold()
                    target_results.append(
                        {
                            "preset": model.preset.name,
                            "model": model.preset.model,
                            "old_answer": old_answer,
                            "clean_final_answer": expected_answer,
                            "incremental_final_answer": observed_answer,
                            "ragmetatest_answer_lexically_different": (
                                expected_normalized != observed_normalized
                            ),
                            "wkc_incremental_answer_detected": (
                                wkc_applicable
                                and bool(observed_answer)
                                and old_normalized == observed_normalized
                            ),
                            "usage": {
                                "old": old_usage,
                                "clean_final": expected_usage,
                                "incremental_final": observed_usage,
                            },
                        }
                    )

                rows.append(
                    {
                        "case_id": case.case_id,
                        "question": case.question,
                        "retriever": retriever_name,
                        "top_k": top_k,
                        "trace": trace_name,
                        "fault": fault_name,
                        "fault_seeded": fault_name != "clean",
                        "execution_seconds": time.perf_counter() - started,
                        "status_check_detected": not audit.status_ok,
                        "count_check_detected": not audit.count_ok,
                        "exact_index_audit_detected": not (
                            audit.exact_ids_ok
                            and audit.versions_ok
                            and audit.content_ok
                        ),
                        "lineage_audit_detected": not (
                            audit.embedding_lineage_ok
                            and audit.duplicate_logical_chunks_ok
                        ),
                        "cache_audit_detected": not audit.cache_coherent,
                        "wkc_incremental_applicable": wkc_applicable,
                        "wkc_incremental_evidence_detected": wkc_incremental_detected,
                        "ragmetatest_evidence_detected": not equivalent,
                        "ragmetatest_state_relation_detected": state_relation_violated,
                        "ragmetatest_detected": trace_mr_detected,
                        "severity": (
                            "L2-evidence"
                            if not equivalent
                            else "L1-structural"
                            if state_relation_violated
                            else "none"
                        ),
                        "cache_hit": observed.cache_hit,
                        "expected_hits": _hit_rows(expected),
                        "observed_hits": _hit_rows(observed),
                        "audit": asdict(audit),
                        "state_comparison": asdict(state_comparison),
                        "clean_state": asdict(normalized_state(clean)),
                        "incremental_state": asdict(normalized_state(incremental)),
                        "events": [_event_row(event) for event in events],
                        "target_results": target_results,
                    }
                )

        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
        temporary.replace(path)

    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows

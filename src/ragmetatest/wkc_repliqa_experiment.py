"""Small RepliQA reproduction of WKC steps 1–3 with ragtest retrievers."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path

from ragmetatest.llm import TargetLLM, parse_correctness, parse_wkc_score
from ragmetatest.repliqa import load_repliqa_cases
from ragmetatest.retrievers import MutableRetriever
from ragmetatest.stateful import (
    FaultProfile,
    StatefulRAG,
    compare_states,
    history_independent,
)
from ragmetatest.types import Event, QueryObservation
from ragmetatest.wkc_adapter import apply_upstream_mutation

RetrieverFactory = Callable[[str], MutableRetriever]


def _hit_rows(hits):
    return [
        {
            "rank": hit.rank,
            "score": hit.score,
            "chunk_id": hit.chunk.chunk_id,
            "chunk_index": hit.chunk.chunk_index,
            "content_hash": hit.chunk.content_hash,
        }
        for hit in hits
    ]


def run_wkc_repliqa(
    *,
    indices: list[int],
    output_path: str | Path,
    upstream_dir: str | Path,
    retriever_factory: RetrieverFactory,
    retriever_name: str,
    target_llms: list[TargetLLM],
    operators: tuple[str, ...] = (
        "semantic_noise",
        "datatype_noise",
        "illegal_sentence_noise",
        "supportive_noise",
        "orthographic_noise",
        "counterfactual_noise",
    ),
    resume: bool = True,
    joint_transitions: bool = False,
) -> list[dict]:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = (
        json.loads(path.read_text(encoding="utf-8"))
        if resume and path.exists()
        else []
    )
    by_key = {
        (int(row["dataset_index"]), str(row["target_preset"])): row for row in rows
    }

    def checkpoint() -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(path)

    for case in load_repliqa_cases(indices):
        retriever = retriever_factory(f"repliqa-{case.question_id}-baseline")
        retriever.build([case.document])
        baseline_hits = retriever.search(case.question, top_k=1)
        for model in target_llms:
            key = (case.dataset_index, model.preset.name)
            row = by_key.get(key)
            if row is not None and row.get("complete", True):
                continue
            if row is None:
                # Eligibility is a conjunction. Checking baseline correctness
                # first is equivalent to the paper's gate and avoids unnecessary
                # closed-book calls when retrieval generation already fails.
                baseline_answer, baseline_usage = model.answer(
                    case.question, baseline_hits
                )
                baseline_judge, baseline_judge_usage = model.judge_correctness(
                    generated_answer=baseline_answer,
                    ground_truth_short=case.answer,
                    ground_truth_long=case.long_answer,
                )
                baseline_correct = parse_correctness(baseline_judge)

                closed_answer = "NOT_RUN"
                closed_judge = "NOT_RUN"
                closed_usage = {}
                closed_judge_usage = {}
                closed_correct = None
                if baseline_correct is True:
                    closed_answer, closed_usage = model.closed_book_answer(case.question)
                    closed_judge, closed_judge_usage = model.judge_correctness(
                        generated_answer=closed_answer,
                        ground_truth_short=case.answer,
                        ground_truth_long=case.long_answer,
                    )
                    closed_correct = parse_correctness(closed_judge)

                eligible = closed_correct is False and baseline_correct is True
                row = {
                    "dataset": "repliqa",
                    "dataset_index": case.dataset_index,
                    "question_id": case.question_id,
                    "question": case.question,
                    "retriever": retriever_name,
                    "target_preset": model.preset.name,
                    "target_model": model.preset.model,
                    "closed_book_runs": 1 if baseline_correct is True else 0,
                    "paper_closed_book_runs": 10,
                    "closed_answer": closed_answer,
                    "closed_judge": closed_judge,
                    "closed_correct": closed_correct,
                    "baseline_answer": baseline_answer,
                    "baseline_judge": baseline_judge,
                    "baseline_correct": baseline_correct,
                    "baseline_hits": _hit_rows(baseline_hits),
                    "eligible": eligible,
                    "eligibility_reason": (
                        "eligible"
                        if eligible
                        else "baseline_incorrect_or_unparsed"
                        if baseline_correct is not True
                        else "closed_book_correct_or_unparsed"
                    ),
                    "mutations": [],
                    "complete": not eligible,
                    "usage": {
                        "closed": closed_usage,
                        "closed_judge": closed_judge_usage,
                        "baseline": baseline_usage,
                        "baseline_judge": baseline_judge_usage,
                    },
                }
                rows.append(row)
                by_key[key] = row
                checkpoint()

            if row["eligible"]:
                baseline_answer = row["baseline_answer"]
                completed_operators = {
                    mutation["operator"] for mutation in row["mutations"]
                }
                for operator_name in operators:
                    if operator_name in completed_operators:
                        continue
                    mutation = apply_upstream_mutation(
                        case.document,
                        operator_name=operator_name,
                        question=case.question,
                        upstream_dir=upstream_dir,
                    )
                    mutated_document = (
                        replace(mutation.document, version=case.document.version + 1)
                        if joint_transitions else mutation.document
                    )
                    mutated_retriever = retriever_factory(
                        f"repliqa-{case.question_id}-{operator_name}-{model.preset.name}"
                    )
                    mutated_retriever.build([mutated_document])
                    mutated_hits = mutated_retriever.search(case.question, top_k=1)
                    mutated_answer, mutated_usage = model.answer(
                        case.question, mutated_hits
                    )
                    judge, judge_usage = model.judge_wkc(
                        question=case.question,
                        original_answer=baseline_answer,
                        mutated_answer=mutated_answer,
                        mutation=operator_name,
                        preserving=mutation.preserving,
                    )
                    score = parse_wkc_score(judge)
                    transitions = []
                    if joint_transitions:
                        reference = StatefulRAG(mutated_retriever, top_k=1)
                        reference.source_documents = {mutated_document.doc_id: mutated_document}
                        expected = QueryObservation(case.question, tuple(mutated_hits))
                        for fault_name in ("clean", "dropped_update", "stale_query_cache"):
                            incremental = StatefulRAG(
                                retriever_factory(
                                    f"joint-{case.question_id}-{operator_name}-{model.preset.name}-{fault_name}"
                                ), fault=FaultProfile(fault_name), top_k=1,
                            )
                            incremental.build([case.document])
                            old = incremental.query(case.question, top_k=1)
                            incremental.apply(Event(
                                action="update", event_id="mutation-update",
                                doc_id=mutated_document.doc_id, document=mutated_document,
                            ))
                            observed = incremental.query(case.question, top_k=1)
                            comparison = compare_states(reference, incremental)
                            extra_usage = {}
                            if history_independent(expected, observed):
                                incremental_answer = mutated_answer
                            elif history_independent(old, observed):
                                incremental_answer = baseline_answer
                            else:
                                incremental_answer, extra_usage = model.answer(
                                    case.question, list(observed.hits)
                                )
                            transitions.append({
                                "fault": fault_name, "fault_seeded": fault_name != "clean",
                                "state_comparison": asdict(comparison),
                                "transition_violated": not comparison.equivalent,
                                "evidence_diverged": not history_independent(expected, observed),
                                "incremental_answer": incremental_answer,
                                "observed_hits": _hit_rows(observed.hits),
                                "usage": extra_usage,
                                "quadrant": (
                                    "unparsed" if score is None
                                    else "both" if score < 3 and not comparison.equivalent
                                    else "wkc_only" if score is not None and score < 3
                                    else "transition_only" if not comparison.equivalent
                                    else "neither"
                                ),
                            })
                    row["mutations"].append(
                        {
                            "operator": operator_name,
                            "preserving": mutation.preserving,
                            "joint_transitions": transitions,
                            "mutated_answer": mutated_answer,
                            "mutated_hits": _hit_rows(mutated_hits),
                            "judge": judge,
                            "score": score,
                            "verdict": (
                                "SATISFIED"
                                if score is not None and score >= 3
                                else "VIOLATED"
                                if score is not None
                                else "PARSE_ERROR"
                            ),
                            "usage": {
                                "answer": mutated_usage,
                                "judge": judge_usage,
                                "transitions": {str(i): t["usage"] for i, t in enumerate(transitions)},
                            },
                        }
                    )
                    checkpoint()
            row["complete"] = True
            checkpoint()

    return rows

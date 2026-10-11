"""Core WKC reproduction runner using pinned operators and ragtest retrievers."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from ragmetatest.fixtures import versioned_fact_cases
from ragmetatest.llm import TargetLLM, parse_wkc_score
from ragmetatest.retrievers import BM25Retriever, MutableRetriever
from ragmetatest.wkc_adapter import apply_upstream_mutation

RetrieverFactory = Callable[[str], MutableRetriever]


def _hits_as_rows(hits):
    return [
        {
            "rank": hit.rank,
            "score": hit.score,
            "chunk_id": hit.chunk.chunk_id,
            "doc_id": hit.chunk.doc_id,
            "version": hit.chunk.version,
            "content_hash": hit.chunk.content_hash,
            "contains_v1_answer": False,
        }
        for hit in hits
    ]


def run_wkc_synthetic(
    *,
    output_path: str | Path,
    upstream_dir: str | Path,
    case_limit: int = 4,
    operators: tuple[str, ...] = (
        "semantic_noise",
        "datatype_noise",
        "illegal_sentence_noise",
        "supportive_noise",
        "orthographic_noise",
        "counterfactual_noise",
    ),
    retriever_factory: RetrieverFactory | None = None,
    retriever_name: str = "bm25",
    target_llms: list[TargetLLM] | None = None,
) -> list[dict]:
    """Run a small faithful-method reproduction before scaling to real data."""

    if retriever_factory is None:
        retriever_factory = lambda _name: BM25Retriever()
    target_llms = target_llms or []
    rows: list[dict] = []

    for case in versioned_fact_cases(case_limit):
        baseline_retriever = retriever_factory(f"wkc-{case.case_id}-baseline")
        baseline_retriever.build([case.doc_v1])
        baseline_hits = baseline_retriever.search(case.question, top_k=1)
        baseline_supported = bool(baseline_hits) and case.answer_v1.casefold() in (
            baseline_hits[0].chunk.text.casefold()
        )
        baseline_answers: dict[str, tuple[str, dict]] = {}
        if baseline_supported:
            for model in target_llms:
                baseline_answers[model.preset.name] = model.answer(case.question, baseline_hits)

        for operator_name in operators:
            mutation = apply_upstream_mutation(
                case.doc_v1,
                operator_name=operator_name,
                question=case.question,
                upstream_dir=upstream_dir,
            )
            mutated_retriever = retriever_factory(
                f"wkc-{case.case_id}-{operator_name}"
            )
            mutated_retriever.build([mutation.document])
            mutated_hits = mutated_retriever.search(case.question, top_k=1)
            mutated_supports_original = bool(mutated_hits) and case.answer_v1.casefold() in (
                mutated_hits[0].chunk.text.casefold()
            )
            deterministic_satisfied = (
                mutated_supports_original if mutation.preserving else None
            )

            base_row = {
                "case_id": case.case_id,
                "retriever": retriever_name,
                "operator": operator_name,
                "preserving": mutation.preserving,
                "description": mutation.description,
                "baseline_retrieval_supported": baseline_supported,
                "mutated_retrieval_supports_original": mutated_supports_original,
                "deterministic_mr_satisfied": deterministic_satisfied,
                "baseline_hits": _hits_as_rows(baseline_hits),
                "mutated_hits": _hits_as_rows(mutated_hits),
                "mutation_log": list(mutation.mutation_log),
                "target_results": [],
            }

            if baseline_supported:
                for model in target_llms:
                    original_answer, original_usage = baseline_answers[model.preset.name]
                    mutated_answer, mutated_usage = model.answer(case.question, mutated_hits)
                    judge_text, judge_usage = model.judge_wkc(
                        question=case.question,
                        original_answer=original_answer,
                        mutated_answer=mutated_answer,
                        mutation=operator_name,
                        preserving=mutation.preserving,
                    )
                    score = parse_wkc_score(judge_text)
                    base_row["target_results"].append(
                        {
                            "preset": model.preset.name,
                            "model": model.preset.model,
                            "original_answer": original_answer,
                            "mutated_answer": mutated_answer,
                            "judge_text": judge_text,
                            "score": score,
                            "verdict": (
                                "SATISFIED"
                                if score is not None and score >= 3
                                else "VIOLATED"
                                if score is not None
                                else "PARSE_ERROR"
                            ),
                            "usage": {
                                "original": original_usage,
                                "mutated": mutated_usage,
                                "judge": judge_usage,
                            },
                        }
                    )
            rows.append(base_row)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows

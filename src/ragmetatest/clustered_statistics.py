"""Descriptive binary rates resampled at the frozen source-document grain."""

from __future__ import annotations

import random
from collections import defaultdict


def document_clustered_rate(observations, documents, *, inference_ready, repeats=2000, seed=42):
    """Keep all observations from each sampled document together.

    Observations are (document_id, bool | None) pairs. None is unknown, never
    false. The estimand is the ratio of summed event counts to summed known
    observations, not an unweighted average of document-specific rates.
    Zero-observation documents remain in the frozen sampling frame.
    """
    documents = sorted(set(documents))
    if repeats < 2:
        raise ValueError("At least two bootstrap draws are required")
    grouped = defaultdict(lambda: [0, 0])
    unknown = 0
    for document, value in observations:
        if document not in documents:
            raise ValueError("Observation has no frozen source document")
        if value is None:
            unknown += 1
        elif isinstance(value, bool):
            grouped[document][0] += int(value)
            grouped[document][1] += 1
        else:
            raise ValueError("Binary observations must be bool or None")
    numerator = sum(value[0] for value in grouped.values())
    denominator = sum(value[1] for value in grouped.values())
    result = {
        "count": numerator, "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
        "unknown": unknown, "population_rows": denominator + unknown,
        "candidate_source_documents": len(documents),
        "contributing_source_documents": len(grouped),
        "document_cluster_percentile_95": None,
        "bootstrap_requested_draws": repeats, "bootstrap_valid_draws": 0,
        "bootstrap_zero_denominator_draws": 0, "bootstrap_seed": seed,
    }
    if not inference_ready:
        result["interval_status"] = "withheld_until_all_candidates_complete"
        return result
    if len(grouped) < 2:
        result["interval_status"] = "insufficient_contributing_documents"
        return result
    if numerator in (0, denominator):
        # A degenerate empirical [0, 0] / [1, 1] interval is not a useful bound
        # on unseen failures. Retain the observed rate without asserting one.
        result["interval_status"] = "withheld_no_observed_binary_variation"
        return result
    values = [grouped.get(document, (0, 0)) for document in documents]
    rng = random.Random(seed)
    draws = []
    for _ in range(repeats):
        draw = [rng.choice(values) for _ in documents]
        total = sum(value[1] for value in draw)
        if total:
            draws.append(sum(value[0] for value in draw) / total)
        else:
            result["bootstrap_zero_denominator_draws"] += 1
    result["bootstrap_valid_draws"] = len(draws)
    if len(draws) < 2:
        result["interval_status"] = "insufficient_nonempty_bootstrap_draws"
        return result
    draws.sort()
    result["document_cluster_percentile_95"] = [
        draws[round((len(draws) - 1) * probability)] for probability in (.025, .975)
    ]
    result["interval_status"] = "descriptive_percentile_conditional_on_nonempty_draws"
    return result


def case_clustered_paired_difference(observations, cases, *, inference_ready, repeats=2000, seed=42):
    """Summed incremental-minus-clean correctness, resampling entire cases.

    Each observation is (case_id, incremental_bool_or_none, clean_bool_or_none).
    Unknown pairs remain in accounting; missing-outcome cases stay in the
    frozen sampling frame. Intervals describe this fixed synthetic workload.
    """
    cases = sorted(set(cases))
    if repeats < 2:
        raise ValueError("At least two bootstrap draws are required")
    grouped = defaultdict(lambda: [0, 0])
    unknown = improvements = regressions = both_correct = both_incorrect = 0
    differences = set()
    for case, incremental, clean in observations:
        if case not in cases:
            raise ValueError("Paired observation has no frozen case")
        if any(value is not None and not isinstance(value, bool) for value in (incremental, clean)):
            raise ValueError("Paired correctness values must be bool or None")
        if incremental is None or clean is None:
            unknown += 1
            continue
        difference = int(incremental) - int(clean)
        grouped[case][0] += difference
        grouped[case][1] += 1
        differences.add(difference)
        improvements += incremental and not clean
        regressions += clean and not incremental
        both_correct += incremental and clean
        both_incorrect += not incremental and not clean
    numerator = sum(value[0] for value in grouped.values())
    denominator = sum(value[1] for value in grouped.values())
    result = {"difference": numerator / denominator if denominator else None,
              "net_correct_pairs": numerator, "known_pairs": denominator, "unknown_pairs": unknown,
              "population_pairs": denominator + unknown, "improvements": improvements,
              "regressions": regressions, "both_correct": both_correct, "both_incorrect": both_incorrect,
              "candidate_cases": len(cases), "contributing_cases": len(grouped),
              "case_cluster_percentile_95": None, "bootstrap_requested_draws": repeats,
              "bootstrap_valid_draws": 0, "bootstrap_zero_denominator_draws": 0, "bootstrap_seed": seed}
    if not inference_ready:
        result["interval_status"] = "withheld_until_model_scope_complete"
        return result
    if len(grouped) < 2:
        result["interval_status"] = "insufficient_contributing_cases"
        return result
    if len(differences) < 2:
        result["interval_status"] = "withheld_no_observed_paired_variation"
        return result
    values = [grouped.get(case, (0, 0)) for case in cases]
    rng = random.Random(seed)
    draws = []
    for _ in range(repeats):
        draw = [rng.choice(values) for _ in cases]
        total = sum(value[1] for value in draw)
        if total:
            draws.append(sum(value[0] for value in draw) / total)
        else:
            result["bootstrap_zero_denominator_draws"] += 1
    result["bootstrap_valid_draws"] = len(draws)
    if len(draws) < 2:
        result["interval_status"] = "insufficient_nonempty_bootstrap_draws"
        return result
    draws.sort()
    result["case_cluster_percentile_95"] = [draws[round((len(draws) - 1) * p)] for p in (.025, .975)]
    result["interval_status"] = "descriptive_percentile_conditional_on_nonempty_draws"
    return result

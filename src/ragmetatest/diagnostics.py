"""Observation-only layer differences; no access to planted fault labels."""

from __future__ import annotations

from ragmetatest.stateful import compare_states, normalized_state


def state_signals(reference, actual):
    comparison = compare_states(reference, actual)
    expected_state, actual_state = normalized_state(reference), normalized_state(actual)
    expected_ids = {chunk[0] for chunk in expected_state.chunks}
    actual_ids = {chunk[0] for chunk in actual_state.chunks}
    signals = []
    if not comparison.logical_source_equal:
        signals.append("logical_source")
    if expected_ids != actual_ids:
        signals.append("index_membership")
    elif not comparison.index_equal:
        signals.append("index_payload_or_lineage")
    if not comparison.cache_coherent:
        signals.append("cache_coherence")
    return {"signals": signals, "missing_chunk_ids": sorted(expected_ids - actual_ids),
            "extra_chunk_ids": sorted(actual_ids - expected_ids)}

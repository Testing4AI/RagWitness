from collections import Counter
from dataclasses import asdict

import pytest

from ragmetatest.supplemental_histories import (
    FAULTS,
    STRATEGIES,
    ViolationSignature,
    execute_history,
    generate_continuous_history,
    initial_documents,
    legal_history,
    shrink_signature,
)
from ragmetatest.types import Event
from ragmetatest.workload import confirmation_cases


def test_continuous_generators_match_operation_payload_multiset_and_legality():
    case = confirmation_cases()[0]
    traces = [generate_continuous_history(case, strategy, 71) for strategy in STRATEGIES]
    for trace in traces:
        assert len(trace) == 32
        assert legal_history(initial_documents(case), trace)
        assert Counter(event.action for event in trace) == {
            "warm_cache": 4, "update": 8, "query": 4, "replay": 4,
            "rollback": 4, "delete": 4, "insert": 4,
        }
        assert len({event.doc_id for event in trace}) == 4
    assert sorted(map(str, (asdict(e) for e in traces[0]))) == sorted(map(str, (asdict(e) for e in traces[1])))
    assert traces[0] != traces[1]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_clean_continuous_reducer_and_sut_converge_at_every_prefix(strategy):
    case = confirmation_cases()[10]
    result = execute_history(initial_documents(case), generate_continuous_history(case, strategy, 71), "clean")
    assert not result["detected"]
    assert result["executed_event_count"] == 32
    assert result["effective_source_changes"] >= 8


def test_illegal_replay_and_insert_are_rejected():
    case = confirmation_cases()[0]
    initial = initial_documents(case)
    assert not legal_history(initial, [Event("replay", "r", document=case.doc_v2, replay_of="unknown")])
    assert not legal_history(initial, [Event("insert", "i", doc_id=case.doc_v1.doc_id, document=case.doc_v1)])


@pytest.mark.parametrize("fault", FAULTS)
def test_shrinker_preserves_concrete_signature_and_removes_writes(fault):
    case = confirmation_cases()[11]
    initial = initial_documents(case)
    trace = generate_continuous_history(case, "relation_guided", 71)
    original = execute_history(initial, trace, fault)
    first = next(row for row in original["checkpoints"] if row["signatures"])
    signature = ViolationSignature(**first["signatures"][0])
    result = shrink_signature(initial, trace, fault, signature)
    assert result["signature_preserved"] and result["one_minimal"]
    assert result["signature"] == asdict(signature)
    assert result["removed_mutating_operations"] > 0
    assert result["minimized_mutating_operations"] >= 1


def test_shrinker_rejects_different_target_even_when_input_fails():
    case = confirmation_cases()[11]
    with pytest.raises(ValueError, match="requested violation signature"):
        shrink_signature(initial_documents(case), generate_continuous_history(case, "relation_guided", 71),
                         "dropped_update", ViolationSignature("index", "unrelated-document", "missing_chunk"))

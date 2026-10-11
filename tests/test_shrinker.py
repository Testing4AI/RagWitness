from ragmetatest.fixtures import versioned_fact_cases
from ragmetatest.retrievers import BM25Retriever
from ragmetatest.shrinker import shrink_trace
from ragmetatest.stateful import FaultProfile
from ragmetatest.traces import replay_trace, update_trace


def test_stale_cache_trace_keeps_warm_update_query():
    case = versioned_fact_cases(1)[0]
    trace = update_trace(case, warm_cache=True)
    result = shrink_trace(
        initial_documents=[case.doc_v1],
        trace=trace,
        fault=FaultProfile("stale_query_cache"),
        retriever_factory=lambda _name: BM25Retriever(),
    )
    assert [event.action for event in result.minimized] == [
        "warm_cache",
        "update",
        "query",
    ]


def test_replay_dependency_is_preserved_during_state_shrinking():
    case = versioned_fact_cases(1)[0]
    result = shrink_trace(
        initial_documents=[case.doc_v1],
        trace=replay_trace(case),
        fault=FaultProfile("non_idempotent_replay"),
        retriever_factory=lambda _name: BM25Retriever(),
        top_k=1,
        include_state_relation=True,
    )
    assert [event.action for event in result.minimized] == ["update", "replay", "query"]


from ragmetatest.fixtures import versioned_fact_cases
from ragmetatest.retrievers import BM25Retriever
from ragmetatest.stateful import FaultProfile, StatefulRAG, history_independent
from ragmetatest.traces import update_trace
from ragmetatest.types import QueryObservation


def _last_observation(outputs):
    return next(item for item in reversed(outputs) if isinstance(item, QueryObservation))


def test_clean_incremental_update_matches_clean_rebuild():
    case = versioned_fact_cases(1)[0]
    incremental = StatefulRAG(BM25Retriever())
    incremental.build([case.doc_v1])
    observed = _last_observation([incremental.apply(event) for event in update_trace(case)])

    clean = StatefulRAG(BM25Retriever())
    clean.build([case.doc_v2])
    expected = clean.query(case.question, use_cache=False)

    assert history_independent(expected, observed)
    assert incremental.audit().exact_ids_ok


def test_dropped_update_is_acknowledged_but_behaviorally_detected():
    case = versioned_fact_cases(1)[0]
    incremental = StatefulRAG(BM25Retriever(), fault=FaultProfile("dropped_update"))
    incremental.build([case.doc_v1])
    observed = _last_observation([incremental.apply(event) for event in update_trace(case)])

    clean = StatefulRAG(BM25Retriever())
    clean.build([case.doc_v2])
    expected = clean.query(case.question, use_cache=False)
    audit = incremental.audit()

    assert audit.status_ok
    assert not audit.exact_ids_ok
    assert not history_independent(expected, observed)


def test_stale_cache_can_evade_exact_index_audit():
    case = versioned_fact_cases(1)[0]
    incremental = StatefulRAG(BM25Retriever(), fault=FaultProfile("stale_query_cache"))
    incremental.build([case.doc_v1])
    observed = _last_observation(
        [incremental.apply(event) for event in update_trace(case, warm_cache=True)]
    )

    clean = StatefulRAG(BM25Retriever())
    clean.build([case.doc_v2])
    expected = clean.query(case.question, use_cache=False)
    audit = incremental.audit()

    assert audit.exact_ids_ok
    assert observed.cache_hit
    assert not history_independent(expected, observed)


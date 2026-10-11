from ragmetatest.diagnostics import state_signals
from ragmetatest.retrievers import BM25Retriever
from ragmetatest.stateful import FaultProfile, StatefulRAG
from ragmetatest.traces import update_trace
from ragmetatest.workload import confirmation_cases


def test_cache_layer_signal_uses_observable_state_only():
    case = confirmation_cases()[0]
    ref = StatefulRAG(BM25Retriever(), top_k=5)
    ref.build([case.doc_v2, *case.background_documents])
    sut = StatefulRAG(BM25Retriever(), top_k=5, fault=FaultProfile("stale_query_cache"))
    sut.build([case.doc_v1, *case.background_documents])
    for event in update_trace(case, warm_cache=True):
        sut.apply(event)
    result = state_signals(ref, sut)
    assert result["signals"] == ["cache_coherence"]
    assert not result["missing_chunk_ids"] and not result["extra_chunk_ids"]
    assert state_signals(ref, ref)["signals"] == []

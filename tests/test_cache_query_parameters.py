from ragmetatest.retrievers import BM25Retriever
from ragmetatest.stateful import StatefulRAG
from ragmetatest.workload import confirmation_cases


def test_default_query_uses_configured_top_k_and_audit_matches_query():
    case = confirmation_cases()[0]
    sut = StatefulRAG(BM25Retriever(), top_k=5)
    sut.build([case.doc_v1, *case.background_documents])
    assert len(sut.query(case.question).hits) == 5
    assert sut.audit().cache_coherent
    assert len(sut.query(case.question, top_k=1).hits) == 1
    assert sut.audit().cache_coherent
    assert len(sut.query(case.question).hits) == 5
    assert sut.audit().cache_coherent

from ragmetatest.history import RELATIONS, generate_history, reduce_source, schedule
from ragmetatest.retrievers import BM25Retriever
from ragmetatest.stateful import StatefulRAG
from ragmetatest.workload import confirmation_cases


def test_source_model_and_history_generation():
    case = confirmation_cases()[0]
    initial = [case.doc_v1, *case.background_documents]
    for relation in RELATIONS:
        trace = generate_history(case, relation, 42)
        assert len(trace) == 6
        expected = reduce_source(initial, trace)
        sut = StatefulRAG(BM25Retriever())
        sut.build(initial)
        for event in trace:
            sut.apply(event)
        assert sorted(sut.source_documents.values(), key=lambda doc: doc.doc_id) == expected


def test_schedulers_have_equal_budget():
    assert set(schedule("systematic", 42)) == set(RELATIONS)
    assert all(len(schedule(strategy, 42)) == 7 for strategy in ("systematic", "random", "single_step"))

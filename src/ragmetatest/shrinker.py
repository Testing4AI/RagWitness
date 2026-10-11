"""One-minimal failure-trace reduction for stateful RAG counterexamples."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ragmetatest.history import reduce_source
from ragmetatest.retrievers import MutableRetriever
from ragmetatest.stateful import (
    FaultProfile,
    StatefulRAG,
    compare_states,
    history_independent,
)
from ragmetatest.types import Document, Event, QueryObservation

RetrieverFactory = Callable[[str], MutableRetriever]


@dataclass(frozen=True)
class ShrinkResult:
    original: tuple[Event, ...]
    minimized: tuple[Event, ...]
    evaluations: int

    @property
    def shrink_ratio(self) -> float:
        if not self.original:
            return 0.0
        return 1.0 - (len(self.minimized) / len(self.original))


def _valid_trace(trace: list[Event]) -> bool:
    if not trace or trace[-1].action != "query":
        return False
    seen: set[str] = set()
    for event in trace:
        if event.action == "replay" and event.replay_of not in seen:
            return False
        seen.add(event.event_id)
    return True


def _fails(
    *,
    initial_documents: list[Document],
    trace: list[Event],
    fault: FaultProfile,
    retriever_factory: RetrieverFactory,
    top_k: int,
    run_label: str,
    include_state_relation: bool,
) -> bool:
    if not _valid_trace(trace):
        return False
    try:
        final_documents = reduce_source(initial_documents, trace)
    except (KeyError, ValueError):
        return False
    sut = StatefulRAG(
        retriever_factory(f"{run_label}-sut"),
        fault=fault,
        top_k=top_k,
    )
    sut.build(initial_documents)
    outputs = [sut.apply(event) for event in trace]
    observed = next(
        output for output in reversed(outputs) if isinstance(output, QueryObservation)
    )

    reference = StatefulRAG(
        retriever_factory(f"{run_label}-reference"),
        top_k=top_k,
    )
    reference.build(final_documents)
    expected = reference.query(observed.question, top_k=top_k, use_cache=False)
    evidence_failure = not history_independent(expected, observed)
    state_failure = include_state_relation and not compare_states(reference, sut).equivalent
    return evidence_failure or state_failure


def shrink_trace(
    *,
    initial_documents: list[Document],
    trace: list[Event],
    fault: FaultProfile,
    retriever_factory: RetrieverFactory,
    top_k: int = 1,
    include_state_relation: bool = False,
) -> ShrinkResult:
    """Greedily remove events until every remaining event is failure-essential."""

    if not _valid_trace(trace):
        raise ValueError("Trace must end in query and contain valid replay dependencies")
    evaluations = 1
    if not _fails(
        initial_documents=initial_documents,
        trace=trace,
        fault=fault,
        retriever_factory=retriever_factory,
        top_k=top_k,
        run_label="shrink-initial",
        include_state_relation=include_state_relation,
    ):
        raise ValueError("Input trace does not reproduce a failure")

    current = list(trace)
    changed = True
    attempt = 0
    while changed:
        changed = False
        # Preserve the final query as the observation point.
        for index in range(len(current) - 1):
            candidate = current[:index] + current[index + 1 :]
            if not _valid_trace(candidate):
                continue
            attempt += 1
            evaluations += 1
            if _fails(
                initial_documents=initial_documents,
                trace=candidate,
                fault=fault,
                retriever_factory=retriever_factory,
                top_k=top_k,
                run_label=f"shrink-{attempt}",
                include_state_relation=include_state_relation,
            ):
                current = candidate
                changed = True
                break

    return ShrinkResult(
        original=tuple(trace),
        minimized=tuple(current),
        evaluations=evaluations,
    )

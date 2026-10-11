"""Deterministic trace templates for the smoke and pilot experiments."""

from __future__ import annotations

from ragmetatest.types import Event, FactCase


def update_trace(case: FactCase, *, warm_cache: bool = False) -> list[Event]:
    events = []
    if warm_cache:
        events.append(
            Event(
                action="warm_cache",
                event_id=f"{case.case_id}-warm-v1",
                question=case.question,
            )
        )
    events.extend(
        [
            Event(
                action="update",
                event_id=f"{case.case_id}-update-v2",
                document=case.doc_v2,
                doc_id=case.doc_v1.doc_id,
            ),
            Event(
                action="query",
                event_id=f"{case.case_id}-query-v2",
                question=case.question,
            ),
        ]
    )
    return events


def replay_trace(case: FactCase) -> list[Event]:
    update_id = f"{case.case_id}-update-v2"
    return [
        Event(
            action="update",
            event_id=update_id,
            document=case.doc_v2,
            doc_id=case.doc_v1.doc_id,
        ),
        Event(
            action="replay",
            event_id=f"{case.case_id}-replay-v2",
            replay_of=update_id,
            document=case.doc_v2,
            doc_id=case.doc_v1.doc_id,
        ),
        Event(
            action="query",
            event_id=f"{case.case_id}-query-after-replay",
            question=case.question,
        ),
    ]


def rollback_trace(case: FactCase) -> list[Event]:
    return [
        Event(
            action="update",
            event_id=f"{case.case_id}-update-v2",
            document=case.doc_v2,
            doc_id=case.doc_v1.doc_id,
        ),
        Event(
            action="rollback",
            event_id=f"{case.case_id}-rollback-v1",
            document=case.doc_v1,
            doc_id=case.doc_v1.doc_id,
        ),
        Event(
            action="query",
            event_id=f"{case.case_id}-query-after-rollback",
            question=case.question,
        ),
    ]


def delete_trace(case: FactCase) -> list[Event]:
    return [
        Event(
            action="delete",
            event_id=f"{case.case_id}-delete",
            doc_id=case.doc_v1.doc_id,
        ),
        Event(
            action="query",
            event_id=f"{case.case_id}-query-after-delete",
            question=case.question,
        ),
    ]


def rename_trace(case: FactCase) -> list[Event]:
    return [
        Event(
            action="rename",
            event_id=f"{case.case_id}-rename",
            doc_id=case.doc_v1.doc_id,
            new_doc_id=f"{case.doc_v1.doc_id}-renamed",
        ),
        Event(
            action="query",
            event_id=f"{case.case_id}-query-after-rename",
            question=case.question,
        ),
    ]


def out_of_order_trace(case: FactCase) -> list[Event]:
    return [
        Event(
            action="update",
            event_id=f"{case.case_id}-update-v2",
            document=case.doc_v2,
            doc_id=case.doc_v1.doc_id,
        ),
        Event(
            action="update",
            event_id=f"{case.case_id}-delayed-v1",
            document=case.doc_v1,
            doc_id=case.doc_v1.doc_id,
        ),
        Event(
            action="query",
            event_id=f"{case.case_id}-query-after-out-of-order",
            question=case.question,
        ),
    ]

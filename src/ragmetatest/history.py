"""Specification-only source reduction and fault-agnostic history generation."""

from __future__ import annotations

import random
from dataclasses import replace

from ragmetatest.traces import (
    delete_trace,
    out_of_order_trace,
    rename_trace,
    replay_trace,
    rollback_trace,
    update_trace,
)
from ragmetatest.types import Document, Event, FactCase

RELATIONS = ("update", "warm_update", "replay", "rollback", "delete", "rename", "out_of_order")


def reduce_source(initial: list[Document], events: list[Event]) -> list[Document]:
    """Reduce the public event contract without consulting SUT state/fault flags."""
    documents = {doc.doc_id: doc for doc in initial}
    seen = set()
    for event in events:
        if event.action in {"query", "warm_cache"}:
            pass
        elif event.action == "delete":
            documents.pop(event.doc_id, None)
        elif event.action == "rename":
            if event.doc_id not in documents or not event.new_doc_id:
                raise ValueError("Invalid rename dependency")
            documents[event.new_doc_id] = replace(documents.pop(event.doc_id), doc_id=event.new_doc_id)
        elif event.action in {"insert", "update", "rollback", "replay"}:
            if event.document is None:
                raise ValueError("Document event missing payload")
            doc = event.document
            prior = documents.get(doc.doc_id)
            replay_seen = event.action == "replay" and event.replay_of in seen
            stale_update = event.action == "update" and prior and prior.version > doc.version
            if not replay_seen and not stale_update:
                documents[doc.doc_id] = doc
        else:
            raise ValueError("Unknown event action")
        seen.add(event.event_id)
    return sorted(documents.values(), key=lambda doc: doc.doc_id)


def generate_history(case: FactCase, relation: str, seed: int, *, event_budget=6) -> list[Event]:
    templates = {
        "update": lambda: update_trace(case),
        "warm_update": lambda: update_trace(case, warm_cache=True),
        "replay": lambda: replay_trace(case),
        "rollback": lambda: rollback_trace(case),
        "delete": lambda: delete_trace(case),
        "rename": lambda: rename_trace(case),
        "out_of_order": lambda: out_of_order_trace(case),
    }
    events = templates[relation]()
    rng = random.Random(f"{seed}:{case.case_id}:{relation}")
    questions = [
        f"What is the operational owner of {doc.doc_id}?" for doc in case.background_documents
    ] or ["What is the unrelated operational owner?"]
    prefix = [Event(
        "query", f"{case.case_id}-probe-{seed}-{index}", question=rng.choice(questions),
    ) for index in range(event_budget - len(events))]
    return prefix + events


def schedule(strategy: str, seed: int, budget=7) -> list[str]:
    rng = random.Random(seed)
    if strategy == "systematic":
        order = list(RELATIONS)
        rng.shuffle(order)
        return [order[index % len(order)] for index in range(budget)]
    if strategy == "random":
        return [rng.choice(RELATIONS) for _ in range(budget)]
    if strategy == "single_step":
        return ["update"] * budget
    raise ValueError("Unknown generation strategy")

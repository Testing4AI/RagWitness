"""Dependency-free lifecycle oracle shared by historical framework adapters.

Adapters execute public APIs and expose normalized observations. Only this module
reduces desired source knowledge, compares layers, and minimizes histories. It
does not import fault labels, issue IDs, or bug-specific expected states.
"""
from __future__ import annotations

import copy
import json
import time
from collections import defaultdict
from typing import Callable, Protocol


class LifecycleAdapter(Protocol):
    def build(self, documents: list[dict]) -> None: ...
    def apply(self, event: dict) -> None: ...
    def snapshot(self) -> dict[str, list[dict]]: ...
    def close(self) -> None: ...


def reduce_documents(initial: list[dict], events: list[dict]) -> list[dict]:
    """Desired source changes even when a downstream ingestion attempt fails."""
    documents = {str(doc["id"]): copy.deepcopy(doc) for doc in initial}
    active_fault = False
    pending_delivery = set()
    for event in events:
        action = event["action"]
        if action == "put":
            doc = copy.deepcopy(event["document"])
            documents[str(doc["id"])] = doc
            target = str(doc["id"])
            if active_fault:
                pending_delivery.add(target)
            else:
                pending_delivery.discard(target)
        elif action == "delete":
            target = str(event["id"])
            documents.pop(target, None)
            if active_fault:
                pending_delivery.add(target)
            else:
                pending_delivery.discard(target)
        elif action == "fault_on":
            if active_fault:
                raise ValueError("Nested fault interval")
            active_fault = True
        elif action == "fault_off":
            if not active_fault:
                raise ValueError("Recovery without a fault interval")
            active_fault = False
        elif action != "query":
            raise ValueError(f"Unsupported lifecycle action: {action}")
    if active_fault:
        raise ValueError("Terminal observation requires recovered fault injection")
    if pending_delivery:
        raise ValueError("Terminal observation requires a post-recovery delivery for each affected logical document")
    return [documents[key] for key in sorted(documents)]


def canonical_snapshot(snapshot: dict[str, list[dict]]) -> dict[str, list[dict]]:
    result = {}
    for layer, records in sorted(snapshot.items()):
        if not isinstance(records, list) or any("id" not in row for row in records):
            raise ValueError("Every observed layer requires records with logical id")
        result[layer] = sorted(copy.deepcopy(records), key=lambda row: json.dumps(row, sort_keys=True))
    return result


def difference_signatures(expected: dict, observed: dict) -> list[dict]:
    """Compare multisets by logical ID; preserve duplicates, metadata and text."""
    signatures = []
    for layer in sorted(set(expected) | set(observed)):
        buckets = []
        for snapshot in (expected, observed):
            grouped = defaultdict(list)
            for row in snapshot.get(layer, []):
                grouped[str(row["id"])].append(json.dumps(row, sort_keys=True))
            buckets.append(grouped)
        left, right = buckets
        for logical_id in sorted(set(left) | set(right)):
            if sorted(left[logical_id]) == sorted(right[logical_id]):
                continue
            kind = "missing" if not right[logical_id] else "unexpected" if not left[logical_id] else "changed"
            signatures.append({"layer": layer, "id": logical_id, "kind": kind})
    return signatures


def run_history(factory: Callable[[], LifecycleAdapter], initial: list[dict], events: list[dict]) -> dict:
    start = time.perf_counter()
    result = {
        "valid": False, "detected": False, "signatures": [], "errors": [],
        "events": copy.deepcopy(events), "event_count": len(events),
        "write_attempt_count": sum(e["action"] in {"put", "delete"} for e in events),
    }
    incremental = reference = None
    try:
        terminal = reduce_documents(initial, events)
        result["terminal_documents"] = terminal
        incremental = factory()
        incremental.build(copy.deepcopy(initial))
        for index, event in enumerate(events):
            try:
                incremental.apply(copy.deepcopy(event))
            except Exception as error:
                allowed = bool(event.get("allow_error", False))
                result["errors"].append({
                    "event_index": index, "type": type(error).__name__,
                    "message": str(error), "allowed": allowed,
                })
                if not allowed:
                    raise
        # An explicitly allowed failed delivery is not itself a successful
        # convergence barrier. The same document must receive a later successful
        # write, in addition to the syntactic recovered-interval contract above.
        for error in result["errors"]:
            failed = events[error["event_index"]]
            if failed["action"] not in {"put", "delete"}:
                continue
            target = str(failed["document"]["id"] if failed["action"] == "put" else failed["id"])
            failed_indices = {item["event_index"] for item in result["errors"]}
            delivered = any(
                index not in failed_indices and event["action"] in {"put", "delete"}
                and str(event["document"]["id"] if event["action"] == "put" else event["id"]) == target
                for index, event in enumerate(events) if index > error["event_index"]
            )
            if not delivered:
                raise ValueError("Allowed write failure was never successfully redelivered")
        observed = canonical_snapshot(incremental.snapshot())
        reference = factory()
        reference.build(copy.deepcopy(terminal))
        expected = canonical_snapshot(reference.snapshot())
        result.update(
            valid=True, expected=expected, observed=observed,
            signatures=difference_signatures(expected, observed),
        )
        result["detected"] = bool(result["signatures"])
    except Exception as error:
        result["invalid_reason"] = f"{type(error).__name__}: {error}"
    finally:
        for adapter in (incremental, reference):
            if adapter is not None:
                adapter.close()
        result["seconds"] = time.perf_counter() - start
    return result


def shrink_history(
    factory: Callable[[], LifecycleAdapter], initial: list[dict], events: list[dict],
    *, signature: dict | None = None,
) -> dict:
    """Deletion-based delta debugging preserving layer, logical ID and kind.

    Only recovered histories with post-recovery delivery are accepted. The final one-deletion pass
    certifies 1-minimality for this signature, not global shortest length.
    """
    start = time.perf_counter()
    original = run_history(factory, initial, events)
    if not original["valid"] or not original["detected"]:
        raise ValueError("Shrinking requires a valid observed divergence")
    signature = copy.deepcopy(signature or original["signatures"][0])
    if signature not in original["signatures"]:
        raise ValueError("Requested signature absent from original execution")
    current = copy.deepcopy(events)
    attempts = []
    replays = 1

    def preserves(candidate):
        nonlocal replays
        replay = run_history(factory, initial, candidate)
        replays += 1
        keep = replay["valid"] and signature in replay["signatures"]
        attempts.append({"event_count": len(candidate), "valid": replay["valid"], "preserved": keep})
        return keep

    granularity = 2
    while len(current) >= 2:
        size = (len(current) + granularity - 1) // granularity
        changed = False
        for offset in range(0, len(current), size):
            candidate = current[:offset] + current[offset + size:]
            if preserves(candidate):
                current = candidate
                granularity = max(2, granularity - 1)
                changed = True
                break
        if not changed:
            if granularity >= len(current):
                break
            granularity = min(len(current), granularity * 2)
    index = 0
    while index < len(current):
        if preserves(current[:index] + current[index + 1:]):
            current = current[:index] + current[index + 1:]
            index = 0
        else:
            index += 1
    final = run_history(factory, initial, current)
    replays += 1
    return {
        "signature": signature, "original": original, "minimal": final,
        "signature_preserved": final["valid"] and signature in final["signatures"],
        "one_minimal": True, "replay_count": replays, "attempts": attempts,
        "removed_events": len(events) - len(current),
        "removed_write_attempts": original["write_attempt_count"] - final["write_attempt_count"],
        "seconds": time.perf_counter() - start,
    }

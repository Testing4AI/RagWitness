#!/usr/bin/env python3
"""Execute the real portable core, then export a dependency-free browser replay."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragmetatest.history import reduce_source
from ragmetatest.retrievers import BM25Retriever
from ragmetatest.stateful import FaultProfile, StatefulRAG
from ragmetatest.supplemental_histories import (
    ViolationSignature, execute_history, signatures, shrink_signature,
)
from ragmetatest.types import Document, Event


def execute(fault):
    old = Document("policy", 1, "The Atlas policy owner is Atlas. The service fee is $10.",
                   {"entity": "Atlas", "attribute": "owner"})
    new = Document("policy", 2, "The Atlas policy owner is Atlas. The service fee is $20.", old.metadata)
    # Background documents keep lexical retrieval non-degenerate.
    initial = [old, Document("garden", 1, "Garden plants need water and sun."),
               Document("museum", 1, "The museum opens at noon on Sundays.")]
    question = "What is the current owner for Atlas?"
    events = [
        Event("warm_cache", "warm-owner", question=question),
        Event("update", "install-v2", doc_id="policy", document=new),
        Event("query", "ask-owner", question=question),
        Event("rollback", "restore-v1", doc_id="policy", document=old),
        Event("query", "ask-again", question=question),
    ]
    sut = StatefulRAG(BM25Retriever(), fault=FaultProfile(fault), top_k=1)
    sut.build(initial)
    snapshots = []
    ledger = set()
    for position in range(len(events) + 1):
        if position:
            sut.apply(events[position - 1])
        reference = StatefulRAG(BM25Retriever(), top_k=1)
        prescribed = reduce_source(initial, events[:position])
        reference.build(prescribed)
        current = signatures(reference, sut, {question: "policy"})
        ledger.update(current)
        # Read-only retrieval probe: it must not alter the execution's cache.
        actual_chunks = sorted((c for c in sut.retriever.all_chunks() if c.doc_id == "policy"),
                               key=lambda c: (c.version, c.chunk_id))
        cached = sut.query_cache.get(question)
        answer = "Atlas" if any("owner is Atlas" in c.text for c in actual_chunks) else "Unavailable"
        snapshots.append({
            "position": position,
            "event": "Initial build" if not position else events[position - 1].action,
            "source_version": next(d.version for d in prescribed if d.doc_id == "policy"),
            "expected": [{"version": c.version, "fee": re.search(r"\$\d+", c.text)[0]}
                         for c in reference.retriever.all_chunks() if c.doc_id == "policy"],
            "actual": [{"version": c.version, "fee": re.search(r"\$\d+", c.text)[0]}
                       for c in actual_chunks],
            "cache_versions": [h.chunk.version for h in cached.hits] if cached else [],
            "answer": answer, "answer_same": answer == "Atlas",
            "signatures": [asdict(s) for s in sorted(current)],
            "ledger": [asdict(s) for s in sorted(ledger)],
        })
    replay = execute_history(initial, events, fault, top_k=1)
    assert [bool(s["signatures"]) for s in snapshots[1:]] == [bool(c["signatures"]) for c in replay["checkpoints"]]
    witness = None
    if fault != "clean":
        first = next(c for c in replay["checkpoints"] if c["signatures"])
        target = ViolationSignature(**first["signatures"][0])
        witness = shrink_signature(initial, events, fault, target, top_k=1)
        assert witness["signature_preserved"] and witness["one_minimal"]
        # Timings are not part of a deterministic teaching replay.
        witness.pop("seconds", None)
        assert replay["detected"] and not replay["terminal_detected"]
    else:
        assert not replay["detected"]
    assert all(s["answer_same"] for s in snapshots)
    return {"fault": fault, "snapshots": snapshots, "events": [asdict(e) for e in events],
            "witness": witness, "prefix_detected": replay["detected"],
            "endpoint_detected": replay["terminal_detected"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "verification-output/demo")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = {"schema": 1, "execution": "Fresh BM25 core execution; deterministic answer-field illustration, no LLM judge",
            "scenarios": [execute(f) for f in ("dropped_update", "orphan_old_chunks", "stale_query_cache", "clean")]}
    (output / "trace.json").write_text(json.dumps(data, indent=2) + "\n")
    (output / "data.js").write_text("window.RAGWITNESS_DEMO = " + json.dumps(data) + ";\n")
    for name in ("index.html", "styles.css", "app.js"):
        source = Path(__file__).parent / name
        if source.resolve() != (output / name).resolve():
            shutil.copyfile(source, output / name)
            if name == 'index.html':
                prefix = Path(os.path.relpath(ROOT, output)).as_posix()
                page = (output / name).read_text().replace('href="../README.md"', f'href="{prefix}/README.md"')
                page = page.replace('href="../docs/', f'href="{prefix}/docs/')
                page = page.replace('href="run_demo.py"', f'href="{prefix}/demo/run_demo.py"')
                (output / name).write_text(page)
    print(json.dumps({"valid": True, "scenarios": 4, "faults_detected": 3,
                      "clean_flags": 0, "signature_preserving_witnesses": 3,
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()

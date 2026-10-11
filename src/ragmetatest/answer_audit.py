"""Read-only answer provenance checks, independent of the result producer.

Bind saved projections to frozen retrieval rows and the authoritative call
ledger. Request hashes attest harness identity, not provider-side execution.
Run against a quiescent answer checkpoint; projection writes are not atomic
across the three JSON files and SQLite.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections import Counter

from ragmetatest.confirmation_scope import answer_unit, expected_answer_units

PROMPT_VERSION = "ragmetatest-answer-equivalence-v1"
SOURCE_FIELDS = ("case_id", "retriever", "top_k", "trace", "fault")


def request_key(*parts):
    return hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def same_value(actual, expected, label):
    assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True), f"Mismatched {label}"


class AnswerLedgerAudit:
    def __init__(self, sources, cases, fingerprints, calls):
        self.cases = {case.case_id: case for case in cases}
        self.models = sorted(fingerprints)
        assert len(self.models) == 2, "Cross-model audit requires exactly two targets"
        assert all(value["prompt_version"] == PROMPT_VERSION for value in fingerprints.values())
        self.fingerprints = fingerprints
        self.calls = calls
        self.sources = {tuple(row[field] for field in SOURCE_FIELDS): row for row in sources}
        assert len(self.sources) == len(sources), "Duplicate frozen retrieval unit"
        expected_sources = {key[:-1] for key in expected_answer_units(self.cases, self.models)}
        assert set(self.sources) == expected_sources, "Incomplete frozen answer retrieval scope"
        self.repeat_cases = list(self.cases)[:10]
        self.answer_jobs = {}
        self.judge_jobs = {}
        for source in sources:
            assert source["question"] == self.cases[source["case_id"]].question, "Wrong frozen question"
            for model in self.models:
                clean = self.answer_key(source, model, "expected_hits")
                observed = self.answer_key(source, model, "observed_hits")
                for key in (clean, observed):
                    self.answer_jobs[key] = (model, 0)
                self.plan_judge(source, model, clean, observed)
        for case_id in self.repeat_cases:
            source = self.repeat_source(case_id)
            for model in self.models:
                base = self.answer_key(source, model, "expected_hits")
                for replicate in (1, 2):
                    key = self.answer_key(source, model, "expected_hits", replicate)
                    self.answer_jobs[key] = (model, replicate)
                    self.plan_judge(source, model, base, key)
        self.verify_calls()

    def repeat_source(self, case_id):
        return self.sources[(case_id, "chroma-contriever", 1, "update", "clean")]

    def answer_key(self, source, model, field, replicate=0):
        identity = [tuple(hit[name] for name in ("doc_id", "version", "chunk_id", "content_hash", "rank"))
                    for hit in source[field]]
        return request_key("answer", self.fingerprints[model], source["question"], identity, replicate)

    def plan_judge(self, source, model, clean_key, observed_key):
        clean, observed = self.calls.get(clean_key), self.calls.get(observed_key)
        if not clean or not observed:
            return
        if clean["answer"].strip().casefold() == observed["answer"].strip().casefold():
            return
        judge = self.models[1 - self.models.index(model)]
        key = request_key("judge", self.fingerprints[judge], source["question"], clean["answer"], observed["answer"])
        self.judge_jobs[key] = judge

    def verify_calls(self):
        for key, call in self.calls.items():
            assert isinstance(call["usage"], dict), "Invalid call usage payload"
            assert isinstance(call["seconds"], (int, float)) and math.isfinite(call["seconds"]) and call["seconds"] >= 0
            if call["kind"] == "answer":
                assert key in self.answer_jobs, "Unplanned answer request key"
                model, replicate = self.answer_jobs[key]
                same_value(call["model"], model, "answer target")
                same_value(call["replicate"], replicate, "answer replicate")
                assert isinstance(call["answer"], str) and call["answer"].strip(), "Empty cached answer"
            else:
                assert call["kind"] == "judge" and key in self.judge_jobs, "Unplanned judge request key"
                same_value(call["model"], self.judge_jobs[key], "judge target")
                assert isinstance(call["text"], str) and call["text"].strip(), "Empty cached judgment"
                match = re.search(r"SCORE\s*:\s*([1-5])", call["text"], re.IGNORECASE)
                same_value(call["score"], int(match.group(1)) if match else None, "parsed judge score")

    def comparison(self, source, model, base, candidate):
        if not base or not candidate:
            return None, None
        if base["answer"].strip().casefold() == candidate["answer"].strip().casefold():
            return True, None
        judge = self.models[1 - self.models.index(model)]
        key = request_key("judge", self.fingerprints[judge], source["question"], base["answer"], candidate["answer"])
        judgment = self.calls.get(key)
        equivalent = judgment["score"] >= 3 if judgment and judgment["score"] is not None else None
        return equivalent, judgment

    def verify_answers(self, rows):
        keys = [answer_unit(row) for row in rows]
        expected_keys = expected_answer_units(self.cases, self.models)
        assert len(keys) == len(set(keys)) and set(keys) <= expected_keys, "Duplicate or out-of-scope answer unit"
        for row in rows:
            source = self.sources[tuple(row[field] for field in SOURCE_FIELDS)]
            model = row["target_model"]
            same_value(row["fault_seeded"], source["fault"] != "clean", "fault provenance")
            same_value(row["state_diverged"], source["ragmetatest_state_relation_detected"], "state signal")
            same_value(row["evidence_diverged"], source["ragmetatest_evidence_detected"], "evidence signal")
            clean_key = self.answer_key(source, model, "expected_hits")
            observed_key = self.answer_key(source, model, "observed_hits")
            same_value(row["clean_call_key"], clean_key, "clean request key")
            same_value(row["observed_call_key"], observed_key, "incremental request key")
            clean, observed = self.calls.get(clean_key), self.calls.get(observed_key)
            if not clean or not observed:
                same_value(row["status"], "missing_answer", "missing answer status")
                assert not any(field in row for field in (
                    "clean_answer", "incremental_answer", "clean_correct", "incremental_correct", "judgment",
                    "clean_exact_answer_match", "incremental_exact_answer_match", "semantic_answer_divergence",
                    "answer_correctness_regression", "exact_answer_match_regression")), "Missing answer carries outcomes"
                continue
            case = self.cases[row["case_id"]]
            expected = ("UNKNOWN" if source["trace"] == "delete" else case.answer_v1
                        if source["trace"] in {"rollback", "rename"} else case.answer_v2)
            same_value(row["expected_value"], expected, "expected source value")
            equivalent, judgment = self.comparison(source, model, clean, observed)
            status = "complete" if equivalent is not None else "unparsed_judgment" if judgment else "missing_judgment"
            same_value(row["status"], status, "answer status")
            same_value(row["judgment"], judgment, "answer judgment ledger payload")
            same_value(row["semantic_answer_divergence"], None if equivalent is None else not equivalent, "semantic divergence")
            for prefix, call in (("clean", clean), ("incremental", observed)):
                same_value(row[f"{prefix}_answer"], call["answer"], "answer ledger text")
                normalized = " ".join(call["answer"].casefold().split())
                target = " ".join(expected.casefold().split())
                presence = normalized == target if target == "unknown" else bool(re.search(r"(?<!\w)" + re.escape(target) + r"(?!\w)", normalized))
                exact = " ".join(call["answer"].casefold().strip().rstrip(".").split()) == target
                same_value(row[f"{prefix}_correct"], presence, "value-presence proxy")
                same_value(row[f"{prefix}_exact_answer_match"], exact, "exact-match proxy")
            same_value(row["answer_correctness_regression"], row["clean_correct"] and not row["incremental_correct"], "proxy regression")
            same_value(row["exact_answer_match_regression"], row["clean_exact_answer_match"] and not row["incremental_exact_answer_match"], "exact regression")
        return {"valid": True, "rows": len(rows), "expected_rows": len(expected_keys),
                "complete": set(keys) == expected_keys and not any(row["status"].startswith("missing_") for row in rows),
                "all_judgments_parsed": bool(rows) and all(row["status"] == "complete" for row in rows),
                "statuses": dict(Counter(row["status"] for row in rows)),
                "provenance": "Every saved answer and judgment is bound to its frozen question, evidence identity, target preset and SQLite request key."}

    def verify_repeats(self, rows):
        keys = [(row["case_id"], row["model"]) for row in rows]
        expected_keys = {(case_id, model) for case_id in self.repeat_cases for model in self.models}
        assert len(keys) == len(set(keys)) and set(keys) <= expected_keys, "Duplicate or out-of-scope fresh unit"
        for row in rows:
            source = self.repeat_source(row["case_id"])
            samples = [self.calls.get(self.answer_key(source, row["model"], "expected_hits", replicate)) for replicate in range(3)]
            same_value(row["samples"], samples, "fresh replicate ledger payloads")
            assert [c["replicate"] for c in row["comparisons_to_first"]] == [1, 2], "Wrong fresh comparisons"
            for comparison, candidate in zip(row["comparisons_to_first"], samples[1:], strict=True):
                equivalent, judgment = self.comparison(source, row["model"], samples[0], candidate)
                same_value(comparison["equivalent"], equivalent, "fresh equivalence")
                same_value(comparison["judgment"], judgment, "fresh judgment ledger payload")
        successful = sum(sample is not None for row in rows for sample in row["samples"])
        missing_judgments = sum(c["equivalent"] is None and c["judgment"] is None for row in rows for c in row["comparisons_to_first"])
        return {"valid": True, "rows": len(rows), "successful_draws": successful,
                "complete": set(keys) == expected_keys and successful == 3 * len(expected_keys) and missing_judgments == 0,
                "unknown_comparisons": sum(c["equivalent"] is None for row in rows for c in row["comparisons_to_first"]),
                "provenance": "All three draws and cross-model comparisons are checked against replicate-specific SQLite request keys."}


def verify_answer_artifacts(directory, sources, cases, presets):
    path = directory / "answer_calls.sqlite"
    calls = {}
    if path.exists():
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
            db.execute("BEGIN")
            calls = {key: json.loads(payload) for key, payload in db.execute("SELECT key,payload FROM calls")}
    fingerprints = {name: {"model": preset.model, "base_url": preset.base_url, "request_args": preset.request_args,
                           "prompt_version": PROMPT_VERSION} for name, preset in presets.items()}
    audit = AnswerLedgerAudit(sources, cases, fingerprints, calls)
    def read(name):
        path = directory / name
        return json.loads(path.read_text()) if path.exists() else []
    exported = Counter(json.dumps(call, sort_keys=True, ensure_ascii=False) for call in read("answer_call_export.json"))
    ledger = Counter(json.dumps(call, sort_keys=True, ensure_ascii=False) for call in calls.values())
    assert exported <= ledger, "Call export contains fabricated or duplicated ledger payloads"
    return {
        "answers": audit.verify_answers(read("answer_evaluation.json")),
        "fresh_answers": audit.verify_repeats(read("answer_repeatability.json")),
        "answer_call_provenance": {"valid": True, "unique_calls": len(calls), "exported_calls": sum(exported.values()),
                                   "planned_unique_answer_requests": len(audit.answer_jobs),
                                   "currently_required_unique_judge_requests": len(audit.judge_jobs),
                                   "complete": bool(calls) and exported == ledger,
                                   "scope": "Frozen request identities, recorded responses and export integrity; not independent provider execution or billing verification."},
    }

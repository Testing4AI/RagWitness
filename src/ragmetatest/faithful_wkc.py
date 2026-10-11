"""WKC core-protocol reproduction with pinned prompts and same-case transitions."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import asdict, replace
from pathlib import Path

from ragmetatest.chat_stream import collect_chat_stream, usage_counts
from ragmetatest.chunking import content_hash
from ragmetatest.llm import TargetLLM, parse_correctness, parse_wkc_score
from ragmetatest.stateful import (
    FaultProfile,
    StatefulRAG,
    compare_states,
    evidence_signature,
)
from ragmetatest.types import Document, Event
from ragmetatest.upstream_contract import parse_preservation
from ragmetatest.wkc_adapter import _upstream_module

OPERATORS = ("semantic_noise", "datatype_noise", "illegal_sentence_noise", "supportive_noise", "orthographic_noise", "counterfactual_noise")
PARAMS = {"semantic_noise": {"noise_ratio": 0.2}, "datatype_noise": {"noise_ratio": 0.1},
          "illegal_sentence_noise": {"noise_ratio": 0.15}, "supportive_noise": {"noise_ratio": 0.2},
          "orthographic_noise": {"error_rate": 0.01}, "counterfactual_noise": {"noise_ratio": 0.1}}
MAX_RECORDED_CALL_ERRORS = 5


class RepeatedCallFailure(RuntimeError):
    """A logical provider call exhausted its recorded retry budget."""

    def __init__(self, key, kind, error_count):
        super().__init__(
            f"logical call {key} ({kind}) has {error_count} recorded failures"
        )
        self.key = key
        self.kind = kind
        self.error_count = error_count


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


class CachedCalls:
    """One process owns one SQLite ledger; independent repeats never share keys."""

    def __init__(self, path, presets, *, stream=False):
        self.stream = stream
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS calls(key TEXT PRIMARY KEY,payload TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS errors(key TEXT,stamp REAL,error_type TEXT)")
        # WKC's temperature is 1.0; other transport options retain gateway presets.
        self.presets = {name: replace(p, request_args={**p.request_args, "temperature": 1.0}) for name, p in presets.items()}
        self.clients = {name: TargetLLM(p) for name, p in self.presets.items()}
        self.completed = self.db.execute("SELECT COUNT(*) FROM calls").fetchone()[0]

    def call(self, model, kind, prompt, *, repeat=0):
        preset = self.presets[model]
        identity = {"model": preset.model, "base_url": preset.base_url, "request_args": preset.request_args,
                    "kind": kind, "prompt": prompt, "repeat": repeat, "max_tokens": None}
        key = content_hash(json.dumps(identity, sort_keys=True))
        existing = self.db.execute("SELECT payload FROM calls WHERE key=?", (key,)).fetchone()
        if existing:
            return json.loads(existing[0])
        error_count = self.db.execute(
            "SELECT COUNT(*) FROM errors WHERE key=?", (key,)
        ).fetchone()[0]
        if error_count >= MAX_RECORDED_CALL_ERRORS:
            raise RepeatedCallFailure(key, kind, error_count)
        started = time.perf_counter()
        try:
            request_args = {k: v for k, v in preset.request_args.items() if k not in {"timeout", "max_tokens", "max_completion_tokens"}}
            # Match upstream call_llm: do not impose a short output-token cap.
            transport = {"transport_streaming": self.stream}
            stream_args = {"stream": True, "stream_options": {"include_usage": True}} if self.stream else {}
            response = self.clients[model].client.chat.completions.create(
                model=preset.model, messages=[{"role": "user", "content": prompt}], timeout=600,
                **request_args, **stream_args)
            if self.stream:
                collected = collect_chat_stream(response, kind=kind)
                text, usage = collected.pop("text"), collected.pop("usage")
                transport.update(collected)
            else:
                choices = getattr(response, "choices", None)
                returned_model = getattr(response, "model", None)
                if not isinstance(choices, list) or len(choices) != 1 or not isinstance(returned_model, str) or not returned_model:
                    raise ValueError("Malformed non-streaming chat-completion response")
                message = getattr(choices[0], "message", None)
                content = getattr(message, "content", None)
                if content is not None and not isinstance(content, str):
                    raise ValueError("Malformed non-streaming chat-completion content")
                text = (content or "").strip()
                try:
                    usage = usage_counts(getattr(response, "usage", None))
                except (AttributeError, TypeError, ValueError) as error:
                    raise ValueError("Malformed non-streaming chat-completion usage") from error
                transport["returned_model"] = returned_model
            if not text.strip():
                raise ValueError("Empty response in faithful WKC request")
        except Exception as error:  # Checkpoint type only, never secrets.
            with self.db:
                self.db.execute("INSERT INTO errors VALUES(?,?,?)", (key, time.time(), type(error).__name__))
            raise
        result = {"key": key, "preset": model, "model": preset.model, "kind": kind,
                  "text": text, "usage": usage, "seconds": time.perf_counter() - started,
                  "repeat": repeat, "prompt": prompt, "request_args": preset.request_args, "max_tokens": None,
                  "transport_timeout_seconds": 600, **transport}
        with self.db:
            self.db.execute("INSERT INTO calls VALUES(?,?)", (key, json.dumps(result)))
        self.completed += 1
        if self.completed % 25 == 0:
            print("FAITHFUL_CALLS", self.completed, flush=True)
        return result


def prepare_inputs(case, frozen, path, caller, contract, generator="antalchemy-kimi-k3"):
    """Freeze generated counterfactual before either target sees this case."""
    path = Path(path)
    if path.exists():
        payload = json.loads(path.read_text())
        digest = content_hash(json.dumps(payload["input"], sort_keys=True, ensure_ascii=False))
        if (digest != payload["input_sha256"] or payload["input"]["question"] != case.question
                or payload["input"]["document"] != asdict(case.document)):
            raise ValueError("Faithful input manifest mismatch")
        if payload["upstream_source_hashes"] != contract.source_hashes:
            raise ValueError("Upstream source changed after input freeze")
        return payload["input"]
    generated = caller.call(generator, "counterfactual_generation", contract.counterfactual_prompt(case.question, case.document.text))
    mo = _upstream_module("third_party/RAG-metamorphic-mutation")
    operator = mo.CounterfactualNoiseOperator(random_seed=42)
    # Only the provider transport is replaced: insertion/rechunking are upstream.
    operator._generate_counterfactual = lambda text, question=None: generated["text"]
    corpus = mo.Corpus(documents=[mo.Document(doc_id=case.document.doc_id, content=case.document.text)], chunk_size=256, chunk_overlap=20)
    mutation = operator.apply_pre_chunk(corpus, noise_ratio=0.1, question=case.question)
    if mutation.mutation_log[0]["generated_counterfactual"] != generated["text"]:
        raise ValueError("Counterfactual generator fell back instead of using recorded LLM output")
    doc = replace(case.document, version=2, text=mutation.mutated_corpus.documents[0].content)
    variants = {name: {"document": frozen["mutations"][name]["document"],
                       "mutation_log": frozen["mutations"][name]["mutation_log"], "params": PARAMS[name]}
                for name in OPERATORS if name != "counterfactual_noise"}
    variants["counterfactual_noise"] = {"document": asdict(doc), "mutation_log": mutation.mutation_log,
                                        "params": mutation.mutation_params, "generator_call_key": generated["key"]}
    inputs = {"dataset_index": case.dataset_index, "question_id": case.question_id, "question": case.question,
              "document": asdict(case.document), "mutations": variants,
              "generator_preset": generator, "generator_policy": "fixed Kimi generator, same frozen mutation for both targets; no heuristic fallback"}
    save_json(path, {"input": inputs, "input_sha256": content_hash(json.dumps(inputs, sort_keys=True, ensure_ascii=False)),
                     "upstream_source_hashes": contract.source_hashes})
    return inputs


def hit_rows(hits):
    return [{"doc_id": h.chunk.doc_id, "version": h.chunk.version, "chunk_id": h.chunk.chunk_id,
             "content_hash": h.chunk.content_hash, "text": h.chunk.text, "rank": h.rank} for h in hits]


def text_signature(hits):
    return [h.chunk.content_hash for h in hits]


def run_candidate(case, model, inputs, caller, contract, reference_backend, actual_backend, row, checkpoint):
    if row.get("complete"):
        return
    trials = row.setdefault("closed_trials", [])
    for repeat in range(len(trials), contract.constants["NUM_CLOSED_BOOK_RUNS"]):
        if any(t["correct"] is True for t in trials):
            break
        generated = None
        try:
            generated = caller.call(model, "closed_answer", contract.constants["CLOSED_BOOK_PROMPT_TEMPLATE"].format(question=case.question), repeat=repeat)
            judged = caller.call(model, "closed_correctness", contract.correctness_prompt(generated["text"], case), repeat=repeat)
        except RepeatedCallFailure as error:
            trials.append({
                "repeat": repeat,
                "answer": generated["text"] if generated else None,
                "judge": None,
                "correct": None,
                "answer_key": generated["key"] if generated else None,
                "judge_key": None,
                "call_error": {
                    "kind": error.kind,
                    "key": error.key,
                    "recorded_failures": error.error_count,
                    "terminal_policy": f"unknown after {MAX_RECORDED_CALL_ERRORS} recorded failures",
                },
            })
            checkpoint()
            break
        trials.append({"repeat": repeat, "answer": generated["text"], "judge": judged["text"],
                       "correct": parse_correctness(judged["text"]), "answer_key": generated["key"], "judge_key": judged["key"]})
        checkpoint()
    if any(t["correct"] is True for t in trials):
        row.update(eligible=False, eligibility_reason="closed_book_correct", complete=True)
        checkpoint()
        return
    if any(t["correct"] is None for t in trials):
        row.update(eligible=None, eligibility_reason="unknown_closed_book_judgment", complete=True)
        checkpoint()
        return
    reference_backend.build([case.document])
    baseline_hits = reference_backend.search(case.question, top_k=1)
    answer = caller.call(model, "baseline_answer", contract.answer_prompt(case.question, [h.chunk.text for h in baseline_hits]))
    judge = caller.call(model, "baseline_correctness", contract.correctness_prompt(answer["text"], case))
    correct = parse_correctness(judge["text"])
    row.update(baseline_answer=answer["text"], baseline_judge=judge["text"], baseline_correct=correct,
               baseline_answer_key=answer["key"], baseline_judge_key=judge["key"], baseline_hits=hit_rows(baseline_hits),
               eligible=True if correct is True else False if correct is False else None,
               eligibility_reason="eligible" if correct is True else "baseline_incorrect" if correct is False else "unknown_baseline_judgment")
    checkpoint()
    if correct is not True:
        row["complete"] = True
        checkpoint()
        return
    completed = {m["operator"] for m in row.setdefault("mutations", [])}
    for name in OPERATORS:
        if name in completed:
            continue
        variant = inputs["mutations"][name]
        mutated_doc = Document(**variant["document"])
        reference = StatefulRAG(reference_backend, top_k=1)
        reference.build([mutated_doc])
        expected = reference.query(case.question, top_k=1, use_cache=False)
        same_context = text_signature(baseline_hits) == text_signature(expected.hits)
        description = contract.describe(name, variant["params"], variant["mutation_log"])
        semantic = judged = generated = None
        preserving = score = None
        if same_context:
            status = "skipped_identical_context"
            mutated_answer = answer["text"]
        else:
            semantic = caller.call(model, "semantic_preservation", contract.semantic_prompt(case.question, case.document.text, mutated_doc.text, description))
            preserving = parse_preservation(semantic["text"])
            generated = caller.call(model, "mutated_answer", contract.answer_prompt(case.question, [h.chunk.text for h in expected.hits]))
            mutated_answer = generated["text"]
            if preserving is None:
                status = "unknown_semantics"
            elif contract.is_abstention(mutated_answer):
                score = 1 if preserving else 5
                status = "complete_deterministic_abstention"
            else:
                prompt = contract.mr_prompt(case.question, answer["text"], mutated_answer, description, preserving,
                                            "context", retrieved_content_original=[h.chunk.text for h in baseline_hits])
                judged = caller.call(model, "metamorphic_judgment", prompt)
                score = parse_wkc_score(judged["text"])
                status = "complete" if score is not None else "unparsed_judgment"
        transitions = []
        for fault in ("clean", "dropped_update", "stale_query_cache"):
            sut = StatefulRAG(actual_backend, fault=FaultProfile(fault), top_k=1)
            sut.build([case.document])
            old = sut.query(case.question, top_k=1)
            sut.apply(Event("update", "faithful-mutation-update", document=mutated_doc))
            observed = sut.query(case.question, top_k=1)
            comparison = compare_states(reference, sut)
            extra = None
            if text_signature(observed.hits) == text_signature(expected.hits):
                incremental_answer = mutated_answer
            elif text_signature(observed.hits) == text_signature(old.hits):
                incremental_answer = answer["text"]
            else:
                extra = caller.call(model, "incremental_answer", contract.answer_prompt(case.question, [h.chunk.text for h in observed.hits]))
                incremental_answer = extra["text"]
            violation = not comparison.equivalent
            quadrant = ("not_applicable" if same_context else "unknown" if score is None else
                        "both" if score < 3 and violation else "wkc_only" if score < 3 else
                        "transition_only" if violation else "neither")
            transitions.append({"fault": fault, "state_comparison": asdict(comparison), "transition_violated": violation,
                                "evidence_diverged": evidence_signature(expected) != evidence_signature(observed),
                                "observed_hits": hit_rows(observed.hits), "incremental_answer": incremental_answer,
                                "extra_call_key": extra["key"] if extra else None, "quadrant": quadrant})
        row["mutations"].append({"operator": name, "status": status, "same_retrieved_text": same_context,
                                 "preserving": preserving, "preservation_judgment": semantic["text"] if semantic else None,
                                 "preservation_call_key": semantic["key"] if semantic else None,
                                 "mutated_answer": mutated_answer, "mutated_answer_key": generated["key"] if generated else None,
                                 "mutated_hits": hit_rows(expected.hits), "score": score,
                                 "judge": judged["text"] if judged else None, "judge_call_key": judged["key"] if judged else None,
                                 "joint_transitions": transitions})
        checkpoint()
    row["complete"] = True
    checkpoint()


def source_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

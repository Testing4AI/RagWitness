"""Credential-safe OpenAI-compatible target-LLM adapter."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI

from ragmetatest.types import RetrievalHit

_ENV_REF = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


@dataclass(frozen=True)
class ModelPreset:
    name: str
    base_url: str
    model: str
    api_key_ref: str
    request_args: dict[str, Any] = field(default_factory=dict)


def load_preset(path: str | Path) -> ModelPreset:
    preset_path = Path(path)
    data = json.loads(preset_path.read_text(encoding="utf-8"))
    return ModelPreset(
        name=data.get("preset_name") or preset_path.stem,
        base_url=data["llm_url"],
        model=data["model"],
        api_key_ref=data.get("api_key", ""),
        request_args=dict(data.get("llm_arg") or {}),
    )


def resolve_api_key(preset: ModelPreset, *, override_env: str | None = None) -> str:
    if override_env:
        value = os.getenv(override_env)
        if value:
            return value
    match = _ENV_REF.fullmatch(preset.api_key_ref.strip())
    if match:
        value = os.getenv(match.group(1))
        if value:
            return value
        raise RuntimeError(f"Required API-key environment variable {match.group(1)} is unset")
    if preset.api_key_ref:
        return preset.api_key_ref
    raise RuntimeError(f"No API key configured for preset {preset.name}")


class TargetLLM:
    def __init__(self, preset: ModelPreset, *, api_key_env: str | None = None):
        self.preset = preset
        self.client = OpenAI(
            base_url=preset.base_url,
            api_key=resolve_api_key(preset, override_env=api_key_env),
            timeout=float(preset.request_args.get("timeout", 90)),
            max_retries=1,
        )

    def complete(self, prompt: str, *, max_tokens: int = 160) -> tuple[str, dict[str, int]]:
        total_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "attempts": 0,
        }
        budgets = [max_tokens]
        if max_tokens < 1024:
            budgets.append(min(1024, max_tokens * 2))
        content = ""
        for budget in budgets:
            args = dict(self.preset.request_args)
            args.pop("timeout", None)
            args.setdefault("temperature", 0)
            args["max_tokens"] = budget
            response = self.client.chat.completions.create(
                model=self.preset.model,
                messages=[{"role": "user", "content": prompt}],
                **args,
            )
            content = (response.choices[0].message.content or "").strip()
            usage = response.usage
            total_usage["attempts"] += 1
            if usage:
                total_usage["prompt_tokens"] += int(usage.prompt_tokens)
                total_usage["completion_tokens"] += int(usage.completion_tokens)
                total_usage["total_tokens"] += int(usage.total_tokens)
            if content:
                break
        return content, total_usage

    def answer(self, question: str, hits: list[RetrievalHit]) -> tuple[str, dict[str, int]]:
        context = "\n\n".join(
            f"[Evidence {hit.rank}; source={hit.chunk.doc_id}; "
            f"version={hit.chunk.version}]\n{hit.chunk.text}"
            for hit in hits
        )
        prompt = f"""Answer the question using only the evidence below.
If the evidence does not contain the answer, output exactly UNKNOWN.
Return only the shortest direct answer, with no explanation.

Evidence:
{context or "(none)"}

Question: {question}
Answer:"""
        # Both target families may consume part of this budget as hidden
        # reasoning tokens before emitting the short visible answer.
        return self.complete(prompt, max_tokens=512)

    def judge_wkc(
        self,
        *,
        question: str,
        original_answer: str,
        mutated_answer: str,
        mutation: str,
        preserving: bool,
    ) -> tuple[str, dict[str, int]]:
        expectation = (
            "The mutation preserves the answer-relevant facts. The two answers "
            "should be semantically equivalent; abstention is a violation."
            if preserving
            else "The mutation changes or contradicts answer-relevant facts. The "
            "second answer should adapt, differ meaningfully, or explicitly abstain."
        )
        prompt = f"""You are reproducing the metamorphic oracle from When Knowledge Changes.

Question: {question}
Mutation: {mutation}
Original answer: {original_answer}
Mutated answer: {mutated_answer}
Expected relation: {expectation}

Rate satisfaction from 1 to 5. Output exactly two lines:
SCORE: <integer 1-5>
EXPLANATION: <one short sentence>"""
        return self.complete(prompt, max_tokens=768)

    def closed_book_answer(self, question: str) -> tuple[str, dict[str, int]]:
        prompt = f"""Answer the following question directly and concisely.
If you do not know, output exactly UNKNOWN.

Question: {question}
Answer:"""
        return self.complete(prompt, max_tokens=512)

    def judge_correctness(
        self,
        *,
        generated_answer: str,
        ground_truth_short: str,
        ground_truth_long: str,
    ) -> tuple[str, dict[str, int]]:
        prompt = f"""Decide whether the generated answer conveys the key information in the ground truth.
It may be shorter or use different wording. Output exactly CORRECT or INCORRECT.

Generated answer: {generated_answer}
Ground-truth short answer: {ground_truth_short}
Ground-truth long answer: {ground_truth_long}

Verdict:"""
        return self.complete(prompt, max_tokens=512)

    def judge_equivalence(
        self,
        *,
        question: str,
        reference_answer: str,
        candidate_answer: str,
    ) -> tuple[str, dict[str, int]]:
        prompt = f"""Judge whether two answers to the same question are semantically equivalent.
Wording, brevity, and harmless extra detail may differ. A changed factual claim,
unsupported commitment, or answer-versus-UNKNOWN difference is not equivalent.

Question: {question}
Reference answer: {reference_answer}
Candidate answer: {candidate_answer}

Rate equivalence from 1 to 5. Output exactly two lines:
SCORE: <integer 1-5>
EXPLANATION: <one short sentence>"""
        return self.complete(prompt, max_tokens=768)


def parse_wkc_score(text: str) -> int | None:
    match = re.search(r"SCORE\s*:\s*([1-5])", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def parse_correctness(text: str) -> bool | None:
    upper = text.strip().upper()
    if "INCORRECT" in upper:
        return False
    if "CORRECT" in upper:
        return True
    return None

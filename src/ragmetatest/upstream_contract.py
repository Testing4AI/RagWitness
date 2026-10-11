"""Reuse pinned WKC prompts/helpers without initializing its API or RAGAS stack."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any, Optional


class UpstreamContract:
    def __init__(self, root="third_party/RAG-metamorphic-mutation"):
        root = Path(root)
        config_path = root / "pipeline_config.py"
        campaign_path = root / "step3_mutation_campaign.py"
        operator_path = root / "mutation_operators.py"
        self.source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in (config_path, campaign_path, operator_path)}
        wanted = {"CLOSED_BOOK_PROMPT_TEMPLATE", "JUDGE_CORRECTNESS_PROMPT_TEMPLATE", "RAG_PROMPT_TEMPLATE",
                  "METAMORPHIC_JUDGE_PRESERVING_CONTEXT_PROMPT", "METAMORPHIC_JUDGE_NON_PRESERVING_CONTEXT_PROMPT",
                  "METAMORPHIC_JUDGE_PRESERVING_DOCUMENT_PROMPT", "METAMORPHIC_JUDGE_NON_PRESERVING_DOCUMENT_PROMPT",
                  "SEMANTIC_PRESERVATION_CHECK_DOCUMENT_PROMPT", "NUM_CLOSED_BOOK_RUNS"}
        self.constants = {}
        for node in ast.parse(config_path.read_text()).body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in wanted:
                        self.constants[target.id] = ast.literal_eval(node.value)
        if set(self.constants) != wanted:
            raise ValueError("Pinned WKC contract no longer matches required constants")
        namespace = {**self.constants, "Any": Any, "Dict": dict, "List": list, "Optional": Optional}
        selected = []
        for node in ast.parse(campaign_path.read_text()).body:
            if isinstance(node, ast.FunctionDef) and node.name in {"_format_mutation_description", "_build_mr_prompt", "_is_abstention"}:
                selected.append(node)
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_ABSTENTION_PATTERNS" for t in node.targets):
                namespace["_ABSTENTION_PATTERNS"] = ast.literal_eval(node.value)
        if len(selected) != 3:
            raise ValueError("Missing upstream helper")
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(campaign_path), "exec"), namespace)  # noqa: S102 - selected trusted pinned source.
        self.describe = namespace["_format_mutation_description"]
        self.mr_prompt = namespace["_build_mr_prompt"]
        self.is_abstention = namespace["_is_abstention"]
        operator_tree = ast.parse(operator_path.read_text())
        cls = next(n for n in operator_tree.body if isinstance(n, ast.ClassDef) and n.name == "CounterfactualNoiseOperator")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_generate_counterfactual")
        query_if = next(n for n in ast.walk(method) if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "question")
        assignment = next(n for n in query_if.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "prompt" for t in n.targets))
        self._counterfactual_expression = compile(ast.Expression(assignment.value), str(operator_path), "eval")

    def counterfactual_prompt(self, question, document):
        return eval(self._counterfactual_expression, {"text": document, "question": question})  # Exact pinned prompt expression.

    def answer_prompt(self, question, texts):
        context = "\n\n".join(f"[Chunk {i}]\n{text}" for i, text in enumerate(texts, 1))
        return self.constants["RAG_PROMPT_TEMPLATE"].format(question=question, context=context)

    def correctness_prompt(self, answer, case):
        return self.constants["JUDGE_CORRECTNESS_PROMPT_TEMPLATE"].format(
            generated_answer=answer, ground_truth_short=case.answer, ground_truth_long=case.long_answer)

    def semantic_prompt(self, question, original, mutated, description):
        return self.constants["SEMANTIC_PRESERVATION_CHECK_DOCUMENT_PROMPT"].format(
            question=question, original_text=original, mutated_text=mutated, mutation_description=description)


def parse_preservation(text):
    value = text.strip().upper().rstrip(".")
    return True if value == "PRESERVING" else False if value == "NON-PRESERVING" else None

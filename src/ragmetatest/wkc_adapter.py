"""Adapter for the pinned WKC mutation operators."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

from ragmetatest.types import Document


@dataclass(frozen=True)
class MutatedDocument:
    name: str
    document: Document
    preserving: bool
    description: str
    mutation_log: tuple[dict, ...]


def _upstream_module(upstream_dir: str | Path):
    directory = str(Path(upstream_dir).resolve())
    if directory not in sys.path:
        sys.path.insert(0, directory)
    return importlib.import_module("mutation_operators")


def apply_upstream_mutation(
    document: Document,
    *,
    operator_name: str,
    question: str,
    upstream_dir: str | Path,
    seed: int = 42,
) -> MutatedDocument:
    mo = _upstream_module(upstream_dir)
    corpus = mo.Corpus(
        documents=[mo.Document(doc_id=document.doc_id, content=document.text)],
        chunk_size=256,
        chunk_overlap=20,
    )
    operators = {
        "semantic_noise": (mo.SemanticNoiseOperator(seed), {"noise_ratio": 0.2}, True),
        "datatype_noise": (mo.DatatypeNoiseOperator(seed), {"noise_ratio": 0.1}, True),
        "illegal_sentence_noise": (
            mo.IllegalSentenceNoiseOperator(seed),
            {"noise_ratio": 0.15},
            True,
        ),
        "supportive_noise": (
            mo.SupportiveNoiseOperator(seed),
            {"noise_ratio": 0.2},
            True,
        ),
        "orthographic_noise": (
            mo.OrthographicNoiseOperator(seed),
            {"error_rate": 0.01},
            True,
        ),
        "counterfactual_noise": (
            mo.CounterfactualNoiseOperator(seed),
            {"noise_ratio": 0.1, "question": question},
            False,
        ),
    }
    if operator_name not in operators:
        raise ValueError(f"Unsupported WKC reproduction operator: {operator_name}")
    operator, kwargs, preserving = operators[operator_name]
    result = operator.apply_pre_chunk(corpus, **kwargs)
    mutated_text = result.mutated_corpus.documents[0].content
    return MutatedDocument(
        name=operator_name,
        document=Document(
            doc_id=document.doc_id,
            version=document.version,
            text=mutated_text,
            metadata={**document.metadata, "mutation": operator_name},
        ),
        preserving=preserving,
        description=result.description,
        mutation_log=tuple(result.mutation_log),
    )

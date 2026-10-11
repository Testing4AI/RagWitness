"""Deterministic versioned facts with exact answer and lineage oracles."""

from __future__ import annotations

from ragmetatest.types import Document, FactCase

_FACTS = [
    ("asterion", "refund window", "7 days", "14 days"),
    ("brindle", "telemetry port", "4127", "8359"),
    ("caligo", "access badge color", "amber", "violet"),
    ("dovetail", "archive retention", "18 months", "30 months"),
    ("emberline", "backup weekday", "Tuesday", "Friday"),
    ("farsight", "maximum payload", "24 megabytes", "48 megabytes"),
    ("gossamer", "support tier", "bronze", "platinum"),
    ("halcyon", "review interval", "45 days", "60 days"),
    ("isotope", "escalation code", "IX-204", "IX-731"),
    ("juniper", "service region", "North Quay", "West Haven"),
    ("keystone", "approval threshold", "three reviewers", "five reviewers"),
    ("lattice", "maintenance hour", "02:00 UTC", "05:00 UTC"),
]


def _document_text(entity: str, attribute: str, value: str, version: int) -> str:
    # The repeated but harmless operational prose creates multiple chunks while
    # keeping one exact answer-bearing sentence. Entity-specific tokens make the
    # retrieval task realistic without relying on model memorisation.
    intro = (
        f"The {entity.title()} Operations Handbook is an internal fictional "
        f"reference. The official {attribute} field is defined only in this "
        f"section. In policy version {version}, the {attribute} for Project "
        f"{entity.title()} is {value}. The current {attribute} is therefore "
        f"{value}. This statement supersedes any informal conversation and is "
        "the authoritative value for answering questions."
    )
    filler_sentences = []
    for index in range(36):
        filler_sentences.append(
            f"Section {index + 1} of the {entity.title()} handbook describes "
            "routine documentation, audit logging, ownership review, and "
            "operational hygiene without changing the authoritative policy value."
        )
    # Put the fact near the middle so chunking mutations can change which chunk
    # contains it, rather than always placing it in the first few tokens.
    return " ".join(filler_sentences[:18] + [intro] + filler_sentences[18:])


def versioned_fact_cases(limit: int | None = None) -> list[FactCase]:
    facts = _FACTS if limit is None else _FACTS[:limit]
    cases = []
    for entity, attribute, old, new in facts:
        doc_id = f"policy-{entity}"
        cases.append(
            FactCase(
                case_id=entity,
                doc_v1=Document(
                    doc_id=doc_id,
                    version=1,
                    text=_document_text(entity, attribute, old, 1),
                    metadata={"entity": entity, "attribute": attribute, "value": old},
                ),
                doc_v2=Document(
                    doc_id=doc_id,
                    version=2,
                    text=_document_text(entity, attribute, new, 2),
                    metadata={"entity": entity, "attribute": attribute, "value": new},
                ),
                question=(
                    f"According to the current {entity.title()} Operations Handbook, "
                    f"what is the {attribute} for Project {entity.title()}?"
                ),
                answer_v1=old,
                answer_v2=new,
            )
        )
    return cases

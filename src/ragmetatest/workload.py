"""Frozen multi-document confirmation corpus; no model generation or fault labels."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, replace
from pathlib import Path

from ragmetatest.chunking import content_hash
from ragmetatest.types import Document, FactCase

DOMAINS = (
    ("refund period", "days", 7, 21),
    ("telemetry port", "", 4100, 8300),
    ("archive retention", "months", 12, 36),
    ("maximum request size", "megabytes", 24, 80),
    ("review interval", "days", 30, 90),
    ("maintenance start", "UTC", 2, 5),
    ("approval threshold", "reviewers", 2, 6),
    ("support response limit", "hours", 48, 12),
    ("backup frequency", "hours", 24, 6),
    ("cache retention", "minutes", 15, 60),
)


def confirmation_cases(count: int = 100, seed: int = 20260903) -> list[FactCase]:
    rng = random.Random(seed)
    bare = []
    for index in range(count):
        attribute, unit, old_base, new_base = DOMAINS[index % len(DOMAINS)]
        entity = f"Project {chr(65 + index // 10)}{index:03d}"
        old = f"{old_base + index // 10} {unit}".strip()
        new = f"{new_base + index // 10} {unit}".strip()
        section_count = (30, 42, 54)[index % 3]
        insertion = (3, section_count // 2, section_count - 3)[(index // 3) % 3]

        def document(
            value: str, version: int, *, entity=entity, attribute=attribute,
            section_count=section_count, insertion=insertion, index=index,
        ) -> Document:
            filler = [
                f"Section {part + 1} covers {entity} operational ownership, routine "
                "audits, incident documentation, release coordination, and service "
                "reviews. These procedures do not specify the requested setting."
                for part in range(section_count)
            ]
            statement = (
                f"The approved {attribute} for {entity} is {value}. "
                f"This is the {attribute} to use for this project under policy "
                f"revision {version}; other projects have separate settings."
            )
            filler.insert(insertion, statement)
            return Document(
                doc_id=f"confirmation-doc-{index:03d}",
                version=version,
                text=" ".join(filler),
                metadata={"entity": entity, "attribute": attribute, "value": value},
            )

        questions = [
            f"What is the current {attribute} for {entity}?",
            f"According to the latest policy, what {attribute} applies to {entity}?",
            f"For {entity}, give the approved {attribute}.",
        ]
        bare.append(FactCase(
            case_id=f"confirm-{index:03d}",
            doc_v1=document(old, 1),
            doc_v2=document(new, 2),
            question=questions[rng.randrange(len(questions))],
            answer_v1=old,
            answer_v2=new,
        ))

    return [
        replace(
            case,
            background_documents=tuple(
                bare[(index + offset) % count].doc_v1
                for offset in (1, 10, 37)
                if count > offset
            ),
        )
        for index, case in enumerate(bare)
    ]


def write_manifest(path: str | Path, count: int = 100) -> None:
    cases = confirmation_cases(count)
    records = [asdict(case) for case in cases]
    canonical = json.dumps(records, sort_keys=True, ensure_ascii=False)
    payload = {
        "schema_version": 1,
        "seed": 20260903,
        "provenance": "synthetic controlled versioned knowledge",
        "case_count": len(cases),
        "case_content_sha256": content_hash(canonical),
        "cases": records,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def load_manifest(path: str | Path) -> list[FactCase]:
    payload = json.loads(Path(path).read_text())
    canonical = json.dumps(payload["cases"], sort_keys=True, ensure_ascii=False)
    if payload["case_content_sha256"] != content_hash(canonical):
        raise ValueError("Frozen confirmation manifest content hash mismatch")
    if payload["case_count"] != len(payload["cases"]):
        raise ValueError("Frozen confirmation manifest case count mismatch")
    return [FactCase(
        **{k: v for k, v in row.items() if k not in {
            "doc_v1", "doc_v2", "background_documents",
        }},
        doc_v1=Document(**row["doc_v1"]),
        doc_v2=Document(**row["doc_v2"]),
        background_documents=tuple(Document(**doc) for doc in row["background_documents"]),
    ) for row in payload["cases"]]

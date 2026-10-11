"""RepliQA loading by immutable sample indices."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset

from ragmetatest.chunking import content_hash
from ragmetatest.types import Document

REPLIQA_REVISION = "bc880adc948fd3a70d5f8b2b3a1d1ee90d820dbd"


@dataclass(frozen=True)
class RepliQACase:
    dataset_index: int
    question_id: str
    document: Document
    question: str
    answer: str
    long_answer: str


def load_repliqa_cases(indices: list[int], *, use_frozen: bool = True) -> list[RepliQACase]:
    manifest = Path("data/manifests/wkc_repliqa_confirmation.json")
    if use_frozen and manifest.exists():
        payload = json.loads(manifest.read_text())
        canonical = json.dumps(payload["cases"], sort_keys=True, ensure_ascii=False)
        if content_hash(canonical) != payload["content_sha256"]:
            raise ValueError("Frozen RepliQA input hash mismatch")
        by_index = {row["dataset_index"]: row for row in payload["cases"]}
        if all(index in by_index for index in indices):
            return [RepliQACase(
                **{key: by_index[index][key] for key in (
                    "dataset_index", "question_id", "question", "answer", "long_answer",
                )}, document=Document(**by_index[index]["document"]),
            ) for index in indices]
    dataset = load_dataset("ServiceNow/repliqa", revision=REPLIQA_REVISION)["repliqa_0"]
    cases = []
    for index in indices:
        item = dataset[index]
        cases.append(
            RepliQACase(
                dataset_index=index,
                question_id=item["question_id"],
                document=Document(
                    doc_id=item["document_id"],
                    version=1,
                    text=item["document_extracted"],
                    metadata={
                        "document_topic": item["document_topic"],
                        "document_path": item["document_path"],
                    },
                ),
                question=item["question"],
                answer=item["answer"],
                long_answer=item["long_answer"],
            )
        )
    return cases

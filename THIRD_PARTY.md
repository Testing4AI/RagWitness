# Licenses and provenance

Original framework and release code retain the existing MIT license, copyright
2026 The Authors. Third-party material keeps its own terms.

- **RAG-MT / When Knowledge Changes.** The pinned source metadata is recorded in
  `evidence/reference/WKC_PROVENANCE.json`. The upstream checkout is not included:
  the recorded snapshot has no explicit redistribution license. Internal filenames
  retaining `wkc` are provenance identifiers, not a second comparison method.
- **Historical case records.** Public issue, patch and revision identifiers are
  retained for attribution. This review package includes experiment records, not
  vendored LangChain/LlamaIndex/Haystack source trees or their Git histories.
- **Dependencies.** NumPy, rank-bm25 and pytest are installed separately under
  their own licenses. No dependencies or model weights are vendored.
- **Data.** The selected history examples use synthetic versioned documents.
  The broader study also uses ServiceNow RepLiQA (CC BY 4.0), but its corpus and
  raw model-call cache are not redistributed in this review package.
- **Model records.** Model aliases and judgment scores in the matched comparison
  are research records. Alias strings identify requested configurations, not an
  independent verification of provider-served weights. No access credential is
  included.

This package avoids author-identifying project URLs, provider service addresses,
local absolute paths, review correspondence and account-linked website metadata.
Public upstream links and third-party attribution are intentionally retained.

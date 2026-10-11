# Claims, comparators and evidence

Run `python3 bin/check_results.py` to recompute the README results. This is saved-data
verification, not a rerun of the original models or databases.

| Claim | Included record | Unit |
| :--- | :--- | :--- |
| RAGWitness 39/40; RAG-MT 28/40 and 24/40 | `evidence/matched/history-verdicts.json` | One continuing history under one seeded configuration |
| Dense-backend endpoint comparison | `evidence/crossbackend/rows/` | 40 seeded + 10 clean histories per backend |
| BM25 history detection | `evidence/continuous/results.json` | 360 seeded + 60 clean executions |
| Signature retention 358 vs 54 | `evidence/reduction/paired-reductions.csv` | Paired reduction of an observed failing history |
| Historical repair validation | `evidence/historical/validation.json` and settled records | Three repeats per variant, not three distinct bugs |

## Baseline protocol

RAG-MT is the closest published metamorphic testing framework for RAG under
knowledge changes. The matched study adapts its follow-up execution to the
continuing RAG instance, retaining its context-related answer relationships,
preservation labels, abstention policy and score threshold (`score < 3`). It is
not a verbatim rerun of the original paper's experimental setting. The frozen
protocol, analysis and upstream provenance are included under `evidence/matched/`
and `evidence/reference/`; the study records retain model request aliases.

All methods observe the same 50 Chroma/Contriever histories, with 32 checkpoints
and four question probes per checkpoint. RAG-MT has 6/10 clean-history alerts for
DeepSeek and 5/10 for Kimi; state and evidence checks have 0/10. An answer-relationship
alert on a clean maintenance execution is not automatically a false statement
about answer quality: these checks test different properties. We therefore show
clean counts separately from seeded-condition detection.

The query-evidence control compares ordered retrieved document/revision/content
projections. RAGWitness checks exposed source, index and cache consistency and
keeps checkpoint violations. A complete mechanical audit with matching state
access is a white-box control, not an answer-focused baseline. Endpoint-only
checking and any-failure reduction are ablations of the testing workflow.

## Interpreting the numbers

- The two target models share the same 40 seeded histories: they do not create
  80 independent cases. The reported 65% is the mean of 70% and 60%.
- Seeded faults test sensitivity to controlled maintenance mechanisms. They do
  not measure production incidence. The one unflagged dense condition leaves
  no observed discrepancy under the executed history.
- The reduction study starts with 360 seeded histories; 358 exhibit a discrepancy.
  All 358 are reduced. The remaining two are retained as non-detections, not
  relabeled as successful reductions.
- A fresh reference can share defects with the maintained implementation.
  Consistency checks detect differences in their declared observation surfaces;
  they do not establish arbitrary answer correctness or inspect hidden state.

`SOURCE_MAP.json` records source and packaged SHA-256 values. Sanitized paths or
service addresses may change a file's byte hash, but not the recorded verdicts.
Use the release manifest for packaged integrity; do not compare sanitized files
against pre-sanitization workspace hashes embedded in historical receipts.

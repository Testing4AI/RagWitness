# From update failures to repair tests

The broader historical investigation covers 18 failure scenarios in 16 issue
families across LangChain, LlamaIndex and Haystack. Four representative cases
were brought into the shared history/reduction/repair-check workflow below.
These are historical reproductions, not claims of vulnerabilities in the latest
upstream releases. Each faulty and patched arm has three repeat executions.

| Case | Failure mechanism | History → witness | Patched-history violations |
| :--- | :--- | ---: | ---: |
| [LangChain #28447](https://github.com/langchain-ai/langchain/issues/28447) / [#28702](https://github.com/langchain-ai/langchain/pull/28702) | Same-source deletion leaves inconsistent bookkeeping | 12 → 1 | 0/3 |
| [LangChain #28103](https://github.com/langchain-ai/langchain/pull/28103) | Failed deletion loses the key needed for retry | 14 → 4 | 0/3 |
| [LlamaIndex #17871](https://github.com/run-llama/llama_index/issues/17871) / [#18303](https://github.com/run-llama/llama_index/pull/18303) | Metadata-only update is not propagated correctly | 9 → 1 | 0/3 |
| [LlamaIndex #19849](https://github.com/run-llama/llama_index/pull/19849) / [#19868](https://github.com/run-llama/llama_index/pull/19868) | Failed ingestion publishes a document prematurely; recovery retry remains incomplete | 11 → 4 | 3/3 |

All four faulty arms are detected in 3/3 repeats. The 24 matched clean controls
have no flags. Here **3/3 means three repeat runs of one case**, not three bugs.

## A repair that satisfies the workflow: 14 → 4

In the LangChain deletion case, a vector-store deletion fails, but bookkeeping
can be removed before deletion succeeds. A retry then lacks the record needed
to find the remaining vector. RAGWitness reduces the larger history to four
events while retaining the selected stale-vector mismatch. Replaying the original
history on the patched version validates the fix in its surrounding workflow.

## A local repair with a remaining retry failure: 11 → 4

The LlamaIndex patch prevents the failed new document from being published too
early. That local rollback succeeds. In the tested historical pipeline, however,
a retained deduplication hash makes the same-document retry look already processed,
so the new revision is still not installed. RAGWitness exposes this remaining
failure after recovery, and source inspection confirms the mechanism.

The experiment uses historical pipeline source overlaid on a LlamaIndex 0.14.15
runtime. It does not establish that the patch introduced the residual failure or
that the current release is affected. This distinction is part of the case's
meaning: **a local fix can be correct while the complete update workflow is not**.

RAGWitness supplies replayable tests and repair checks; it does not synthesize
patches or automatically prove root causes. The settled records and the unified
validation summary are in `evidence/historical/`.

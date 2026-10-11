# Reproduction levels

Choose the smallest level needed for the question being reviewed.

| Level | Command / entry point | Dependencies | What it establishes |
| :--- | :--- | :--- | :--- |
| Integrity | `python3 bin/verify_release.py` | Python standard library | Packaged files match the release manifest; known identity/credential guards pass |
| Saved-result recomputation | `python3 bin/check_results.py` | Python standard library | Headline counts agree with packaged per-history/per-reduction records |
| Interactive replay | `demo/index.html` | Browser | An inspectable example of the testing workflow, using recorded core execution |
| Fresh core execution | `python demo/run_demo.py --output verification-output/demo` | `requirements-demo.txt` | The shipped implementation reproduces the illustrated checkpoint failures and witnesses |
| Continuous-history execution | `python bin/run_histories.py --cases 1 --reduce` | Same environment | Fresh 32-event histories across two schedules, six fault profiles and clean controls; optional signature reduction |
| Core regression suite | `PYTHONPATH=src python -m pytest -q tests` | Same environment | Portable tests exercise history legality, state checks and signature reduction |

All execution commands should be run from the repository root. Regenerated output
is separate from the shipped demo. To view it, serve the repository and visit
`/verification-output/demo/`. Nothing in the quick start contacts a model service.

## Included scope

This is a **curated anonymous review repository**, not a mirror of the research
workspace. It includes the framework source, selected regression tests, a fresh
executable example, matched comparison verdicts, dense-backend history records,
the original continuous-history records, paired reduction rows and historical
repair records. It excludes multi-gigabyte database working directories, raw
provider request caches, local environments, manuscript drafts and private reviews.
The paired reduction CSV supports recounting outcomes; it does not itself rerun
every minimization attempt.

The Python modules retain the research implementation's namespace and semantics.
The shipped tests intentionally cover the portable core, not every integration
test in the research workspace. The live demo uses a synchronous BM25 backend;
the paper's database and model comparisons are represented by frozen records.

## Larger integrations

`src/ragmetatest/retrievers.py` contains the Chroma and Qdrant adapters. Their
recorded Contriever path depends on the original `sentinelrag.utils.embeddings`
loader, which is **not bundled here**. The source is included for inspection, not
presented as a one-command clean-room dense rerun. A full dense rerun additionally
needs that loader, compatible model weights and a service-backed Qdrant instance.

`faithful_wkc.py`, `wkc_adapter.py` and the answer experiment modules preserve
the baseline integration logic. Running them requires their original experiment
inputs, licensed upstream dependencies and user-supplied model access. No key or
private gateway is required or included in this repository. Inspect the reference
provenance before fetching the upstream RAG-MT implementation; it is not vendored.

Historical case records preserve public issue/revision information and observed
repair behavior. They are not a packaged replacement for the upstream source
overlays and version-specific dependency environments. See [REAL_WORLD.md](REAL_WORLD.md).

## Release checks

`MANIFEST.sha256` excludes itself and generated verification output. The verifier
ignores standard local environment/cache directories, so installing dependencies
or running tests does not invalidate the release. Editing a tracked source or
evidence file does invalidate it. A failed check is reported as a failure; missing
or unknown outcomes are never treated as successful tests.

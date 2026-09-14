# Validation results — September 14, 2026 UTC

| Check | Result | What it establishes |
| --- | --- | --- |
| Offline unit / app regression suite | 96 passed | Software contracts, isolation, retry bounds, source expansion, batch failure handling and benchmark negative controls |
| Recorded financial / follow-up / failure benchmark | 27 passed, 17 explicitly skipped | Deterministic paths; no network or model accuracy claim |
| Fresh historical SEC benchmark | 20 / 20 passed | Five companies × four supported metrics, 2025-10-31 cutoff, correct values, source identity, periods, units, displayed numbers and comparisons; cache reads disabled |
| Current SEC readiness | 5 / 5 verified live | Current annual records for AAPL, MSFT, GOOGL, AMZN and NVDA as of 2026-09-14; no fallback |
| Local-model development benchmark | **43 / 44 passed** | 20 financial cases, 11/12 document cases, 8 routing/follow-up cases and 4 deterministic failure injections |
| Browser | Passed | Current AAPL/NVDA comparison, correctly dated SEC facts, and NVIDIA chip opening its own original 2026 filing |
| Packaging checks | Passed | Compose configuration and Git whitespace checks; not a new Docker image build or hosted release |

## Remaining substantive failure

`document-capex`: the local model answered **$12.715 million**, but the reported
outflow is **$12,715 million ($12.715 billion)**. The source row was retrieved and
cited, yet the magnitude in the answer was wrong. The `numeric_unit` check failed,
and the benchmark exits nonzero. See Apple's cash flow statement, PDF page 36 /
report page 33, in the [original filing](https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm).

The production document-answer path still allows this class of error. Source-page
expansion improves evidence availability; citation-ID checks do not validate
financial claims. Claim-level numerical verification and a separate held-out
filing set are the next accuracy work. Do not represent these results as analyst
certification or general financial correctness.

## Changes supported by observed failures

- Real operator identity resolved the initial SEC 403 on this network. The contact
  remains only in an ignored, mode-600 local file, excluded from Docker images.
- HTTP retries honor `Retry-After` and share cooldowns across sessions. Typed
  transport errors prevent semantic errors from accidentally enabling snapshots.
- Exact user-supplied tickers are retained when the planner expands their names;
  validated four-digit fiscal-year strings are normalized. Explicit ticker-pair
  subject order is preserved if the model reverses it.
- Ranked PDF chunks expand to bounded source pages, restoring statement headings
  and omitted rows. The earlier total-liabilities answer was wrong; the final
  run answered it correctly with the expected evidence.
- The local Ollama embedding runner reset during the original whole-filing batch.
  It was unloaded and recovered; indexing then succeeded with 16-chunk batches.
  No model downloads or dependency changes occurred. This is a single successful
  recovery/run, not a concurrency or endurance benchmark.

## Reproducibility and scoring revisions

Runtime: Python 3.13, local `llama3`, `mxbai-embed-large`, cached
`cross-encoder/ms-marco-MiniLM-L-6-v2`, temperature 0, k=8, top_n=4, source-page
limits 6,000 characters per page / 18,000 total. The model benchmark uses recorded
SEC inputs to separate language-model behavior from changing network responses.
The separate live benchmark exercises fresh network calls.

The final generation run initially scored 41/44. Two citations were correct
references to matching tables elsewhere in Apple's filing: services on PDF page
26 and R&D on page 27. After checking the original pages, dataset version 2 accepts
those locations. The **same saved answers** were re-scored to 43/44; no expected
financial values, question wording or tolerances changed, and the remaining
capital-expenditure failure was retained. These are development-set results, not
a held-out test.

Compact audit records are versioned in
[model results](results/model-2026-09-14.json) and
[live SEC results](results/live-sec-2026-09-14.json).
Full answers, request traces and source excerpts are saved locally under the
ignored `evaluation-results/` directory. The original and rescored model reports
are both preserved. CI runs the offline benchmark and uploads its report; a
GitHub CI run and a new hosted deployment were not verified in this change.

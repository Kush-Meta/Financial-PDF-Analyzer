# Validation results — September 14, 2026 UTC

## Numerical-evidence milestone

| Check | Result | What it establishes |
| --- | --- | --- |
| Offline unit/app suite | **142 passed** | Includes 32 independently authored synthetic adversarial tests, public statement regressions, formula/source records, scope protection, index bypass and UI/export checks |
| Recorded offline benchmark | 27 passed, 17 explicitly skipped | Existing deterministic financial/follow-up/failure paths; no new model or network claim |
| Full development benchmark with local planner | **44/44 passed** | 20 peer-financial cases, 12 document cases, 8 routing/follow-up cases and 4 failure injections |
| Separate document audit run | **12/12 passed** | All twelve answers retain `verification.status=verified`, exact row/year/currency/scale records and deterministic output; current-code replay matches every recorded answer and audit |
| Final browser preview | Passed | Correct capex currency text, statement-row caption, expanded original-row audit and PDF page 36 evidence; checked in the existing narrow preview |

The original capex answer is preserved in
[`capex-model-failure.json`](../tests/fixtures/capex-model-failure.json). The new
statement path renders **$12,715 million ($12.715 billion)** from `(12,715)` on
Apple PDF page 36, fiscal 2025, with the negative source cell and explicit absolute
value transformation in the audit. No expected amount, accepted source label or
scoring tolerance changed for this milestone.

Browser inspection initially caught paired dollar signs rendering as inline math.
Currency signs are now escaped for display. The retained preview process required
a restart to load that imported-module change; the final actual browser result
was inspected after restarting, not inferred from a health endpoint.

This is an improvement in the application's answer path, **not a new claim of
model arithmetic accuracy**. The local `llama3` planner still routes questions;
supported statement answers read the complete recognized statement pages and use
Decimal arithmetic without embeddings, reranking or answer generation. The twelve
document cases all use that path. The configured embedding/reranker names remain
in runner metadata but were not exercised by these document questions. Peer
answers still use deterministic comparisons over recorded SEC inputs.

The full 44-case run exposed a reporting omission: document cases retained answers
and sources but omitted research metadata. The logger was fixed and the separate
12-case document run captured the full numerical records. These runs and their
timestamps are retained separately in
[full development results](results/model-numeric-2026-09-14.json) and
[document audit results](results/document-numeric-audit-2026-09-14.json); they are
not combined into a fabricated single run. Raw reports remain local under
`evaluation-results/`. The original 43/44 report below is unchanged.

The benchmark still uses page-level evidence-hit proxies. Exact row/year checks
and negative controls are tested separately; the score alone does not prove
claim-level entailment. This is an Apple-centered development set plus synthetic
safety cases, not a held-out multi-filing evaluation or analyst certification.
Unsupported layouts/questions and broader model prose remain explicitly unchecked.
No new dependency, model download, live SEC test, GitHub CI run, Docker image build
or hosted deployment is claimed for this milestone.

## Previous milestone: live-data reliability

| Check | Result | What it establishes |
| --- | --- | --- |
| Offline unit / app regression suite | 96 passed | Software contracts, isolation, retry bounds, source expansion, batch failure handling and benchmark negative controls |
| Recorded financial / follow-up / failure benchmark | 27 passed, 17 explicitly skipped | Deterministic paths; no network or model accuracy claim |
| Fresh historical SEC benchmark | 20 / 20 passed | Five companies × four supported metrics, 2025-10-31 cutoff, correct values, source identity, periods, units, displayed numbers and comparisons; cache reads disabled |
| Current SEC readiness | 5 / 5 verified live | Current annual records for AAPL, MSFT, GOOGL, AMZN and NVDA as of 2026-09-14; no fallback |
| Local-model development benchmark | **43 / 44 passed** | 20 financial cases, 11/12 document cases, 8 routing/follow-up cases and 4 deterministic failure injections |
| Browser | Passed | Current AAPL/NVDA comparison, correctly dated SEC facts, and NVIDIA chip opening its own original 2026 filing |
| Packaging checks | Passed | Compose configuration and Git whitespace checks; not a new Docker image build or hosted release |

### Preserved earlier substantive failure

`document-capex`: the local model answered **$12.715 million**, but the reported
outflow is **$12,715 million ($12.715 billion)**. The source row was retrieved and
cited, yet the magnitude in the answer was wrong. The `numeric_unit` check failed,
and the benchmark exits nonzero. See Apple's cash flow statement, PDF page 36 /
report page 33, in the [original filing](https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm).

At that milestone, the document-answer path still allowed this class of error. The subsequent supported statement path above fixes this recorded case; general model interpretations remain outside its numerical checks. Source-page
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

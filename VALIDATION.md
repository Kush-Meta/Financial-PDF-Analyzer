# Validation record

Verified locally on September 12, 2026, starting from canonical GitHub main commit
`bb37d946060c36d04e81c97259e278ca873b3efc` on branch `codex/finread-reliable-chat`.

## Passed

- 32 deterministic tests: `python -m unittest discover -s tests -v`.
- Real local-model smoke test: `python scripts/smoke_models.py`.
  - Built a FAISS index with `mxbai-embed-large`.
  - Retrieved and reranked synthetic evidence using cached MiniLM weights.
  - `llama3` answered fiscal 2024 revenue as $42 million with `[S1]` pointing to page 1.
  - LangExtract returned five entities; every returned span matched its source offsets.
  - Total smoke test elapsed time: 24.1 seconds. This is not a performance benchmark.
- Streamlit browser preview rendered successfully on `http://127.0.0.1:8502`.
- `docker compose config --quiet` and `git diff --check`.
- Existing virtual environment: `pip check` reported no broken requirements.

## UI overhaul — September 13, 2026

- 38 deterministic tests passed after the redesign, including sample open/close,
  one-shot starter and follow-up submission, citation-to-page navigation, retained
  indexing after starting a new conversation, retry without retyping, and complete
  UI state reset when replacing a source.
- AppTest exercised both the empty workspace and the fictional sample workspace.
- Python compilation and `git diff --check` passed.
- The existing local preview health endpoint returned `ok` on port 8502.
- No new dependencies were installed. The underlying model pipeline is unchanged
  from the earlier live smoke test; that result is not a new financial benchmark.
- Browser screenshots and responsive browser interaction testing were not performed
  for this redesign. The earlier browser-render check above applies to the prior UI.

Local runtime: Python 3.13, Streamlit 1.54.0, LangChain 0.3.25,
langchain-ollama 0.3.3, LangExtract 1.1.1, pypdf 6.7.0.
Requirements were not changed and no model downloads were performed for the smoke test.

## Real SEC example — September 13, 2026

- Replaced the fictional onboarding briefing with Apple's complete FY2025 Form 10-K
  downloaded from its as-filed PDF distribution URL and matched to SEC accession
  0000320193-25-000079. Provenance and checksum are recorded in `assets/filings/README.md`.
- Parsed all 80 PDF pages through the same reader used for uploads. Visually checked
  the consolidated statements page and confirmed PDF page 32 maps to report page 29.
- Updated the sample interaction regression to verify Apple, the 2025 filing, all
  80 pages, and the printed-page mapping. No model call occurs just by opening it.
- All 38 tests passed after the replacement.
- Added narrow Git and Docker ignore exceptions for this public filing only.

## Preview recovery — September 13, 2026

The retained Streamlit process served a stale imported `sample_document` module
after the SEC example changed its exports. Fresh-process tests passed while the
browser failed with `ImportError: cannot import name 'SAMPLE_SEC_URL'`.
Restarted the preview and enabled the polling file watcher so imported local
modules are monitored without an optional watchdog dependency. For future module
interface changes, verify the running browser session: an HTTP health response or
a separate test process does not prove that a retained preview has reloaded.
Verified the repaired browser session: the landing page rendered without errors,
and opening the Apple example displayed all 80 pages, both original-source links,
the question starters, and the enabled chat composer.

## Competitor research — September 13, 2026

- 73 offline tests passed with the new planner, SEC adapter, snapshot fallback,
  comparison arithmetic, original-accession and annual-period checks, unit/conflict
  handling, entity verification, HTTP retry bounds, explicit follow-up state edits,
  source inspection and exports.
- `scripts/smoke_research.py` passed against the installed local `llama3`: document
  questions, implicit margin benchmarking, explicit named peers, quarterly scope
  clarification, credit-rating scope handling, and an end-to-end comparison with
  no embedding/index call. Apple and Microsoft snapshot margins round to 31.97%
  and 45.62% from independently checked report inputs.
- Live-model follow-up experiments initially failed: the model lost the peer plan
  on short metric, peer and cutoff edits. Added deterministic state transitions and
  regression tests for those explicit edits. Free-form planning remains model-dependent.
- Browser testing caught an unsupported qualitative sentence despite valid source
  IDs. Removed model-generated narration from the peer tool. The final table and
  comparison conclusions are deterministic; a test verifies that this tool does
  not call an answer model. Another verifies that Microsoft's operating income is
  lower than Apple's while its margin and operating cash flow are higher in the
  example periods. Document-only answers remain model-generated.
- Direct SEC requests from this network returned HTTP 403. No access restriction
  was bypassed. Live JSON behavior is covered with offline fixtures; successful
  live SEC retrieval is **not verified**. The real-report fallback is explicitly
  dated and cannot fulfill current-data requests.
- Docker Compose configuration and `git diff --check` passed. Requirements were
  unchanged; no additional package or model was installed.
- Final browser verification: the named Apple/Microsoft margin question displayed
  the compact sourced table and deterministic comparison, without generated prose.
  Selecting the Microsoft source showed its figures and linked to Microsoft's
  report. "And operating cash flow?" retained Microsoft and the historical cutoff,
  changed the comparison metric, and completed without another planning/model call.
  "Use Alphabet instead" then replaced Microsoft while retaining the cash-flow
  metric. The final narrow browser layout displayed a compact table and stacked
  evidence panel without a page-wide horizontal scrollbar.

## Remaining validation limits

The subsequent interface simplification was verified in the retained browser on
September 13: a plain wordmark, no starter/follow-up cards, a prominent composer,
and collapsed document/settings tools. All 38 tests passed, including consecutive
questions, source navigation, and failure recovery. The layout retains stable
widget positions while CSS changes the presentation after the first answer.

- Financial accuracy on real filings, retrieval recall, or resistance to malicious PDF content.
- Multi-user load, durability, access control, or hosted deployment readiness.
- A built/running Docker image or the GitHub CI job. Compose syntax was checked locally;
  CI has been added but has not run on GitHub.
- Compatibility of every alternative model or a clean install of the unpinned dependencies.

See the production milestones in README.md before treating this as a hosted service.

## Live-data reliability and accuracy benchmark — September 14, 2026 UTC

This section supersedes the earlier live-connectivity limitation. Configuring the
actual operator contact resolved the SEC 403 on the development network. Fresh
current annual financials were retrieved for AAPL, MSFT, GOOGL, AMZN and NVDA.
The live historical benchmark passed all 20 supported metric comparisons against
separately transcribed report values, with HTTP cache reads disabled.

The new 44-case development benchmark separates financial facts, document answers,
routing/follow-ups and explicitly injected failures. The final result is **43/44**
after a documented correction to two accepted source locations. One material
capital-expenditure unit error remains in generated document answers and fails
the benchmark. It is not hidden by the 96 passing offline tests. See
[full results and limitations](benchmarks/RESULTS.md), including original/rescored
report provenance, and [running instructions](benchmarks/README.md).

Source-page expansion recovered missing statement rows in document research.
Local Ollama indexing failed during a whole-filing embedding batch; the embedding
model was unloaded/recovered, and the new 16-chunk batches successfully indexed
the full Apple filing. Atomic publication, bounded batches and incomplete/failing
batch behavior have regression coverage. This is not a load/endurance result.

Fresh browser verification on port 8502: the Apple example opened successfully;
"Compare AAPL with NVDA on operating margin using current data" returned SEC facts
at the 2026-09-14 cutoff, including NVIDIA's year ended 2026-01-25, without a dated
snapshot notice. The NVDA chip showed the correct company and its original 2026
filing URL. Compose configuration, whitespace checks and contact-file exclusion
from Git-visible files passed. The operator contact is stored locally with mode
600 and excluded from Git/Docker; no dependencies were changed. CI benchmark
artifact collection is configured but has not been verified on GitHub in this run.

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

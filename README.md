# FinRead

A local financial research app built with Streamlit, Ollama, FAISS, and LangExtract. Ask questions about a filing or compare annual company financials, then inspect the evidence behind the answer.

## Documentation

| Guide | Purpose |
| --- | --- |
| [Developer and operator guide](DEVELOPMENT.md) | Request flow, configuration, limits, evidence contracts, session ownership and failure recovery |
| [Roadmap](ROADMAP.md) | Delivered scope, remaining work and release acceptance criteria |
| [Changelog](CHANGELOG.md) | Completed milestones and implementation commits |
| [UI design](DESIGN.md) | Interaction decisions and design references |
| [Intelligence design](INTELLIGENCE.md) | Competitor architecture, research references and future credit methodology |
| [Numerical evidence](NUMERICAL_EVIDENCE.md) | Supported statement figures, formulas, source-cell checks and explicit limits |
| [Open-source survey](OPEN_SOURCE.md) | Comparable projects and which user-facing patterns FinRead adopted |
| [Validation history](VALIDATION.md) | Checks actually performed, failures and remaining limitations |
| [Benchmark guide](benchmarks/README.md) | Dataset provenance, commands, scoring and label corrections |
| [Benchmark results](benchmarks/RESULTS.md) | Current checks, preserved earlier failures and audit records |
| [Apple filing provenance](assets/filings/README.md) / [peer snapshot sources](assets/filings/PEER_SOURCES.md) | Original documents, dates and fallback restrictions |

## Run locally

Use Python 3.10 or newer and an existing Ollama installation:

```bash
ollama pull mxbai-embed-large
ollama pull llama3
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.address=127.0.0.1
```

Open [FinRead](http://localhost:8501). The default reranker downloads its public model weights on first use. Disable **Rerank evidence** to use vector search alone.

Alternatively:

```bash
docker compose up --build
```

Compose binds the app to localhost and routes model calls to the host's Ollama server. The container runs as a non-root user, has an HTTP health check, and excludes uploaded PDFs from its build context. Set `OLLAMA_BASE` for other Ollama endpoints; that endpoint receives document text.

## Research workflow

1. Add a text-based PDF in the source sidebar, or choose **Open Apple’s 2025 10-K** to use the complete 80-page public SEC filing. Links to the SEC original and as-filed PDF remain visible in the source panel. PDFs can contain up to 25 MB and 1,000 pages. Invalid, encrypted, and textless files produce actionable errors.
2. Type a question in the main conversation box. A validated planner chooses document or competitor research. Supported annual-figure questions read statement rows directly. Broader document questions build or reuse the search index; peer comparisons do not require embeddings. Ask follow-up questions naturally in the same box.
3. Click a source chip such as **S1 · p. 2** or **S2 · MSFT** to inspect evidence in the adjacent **Evidence** panel. Uploaded excerpts open the correct PDF page; external figures link to the corresponding company's original report.
4. Open **Evidence** and switch to **Pages** to browse text, or **Figures** to extract entities from up to five selected pages. Model and retrieval controls live under **Settings** in the sidebar.
5. Use **More** to download JSON research or readable Markdown notes with source excerpts. Entity results have their own JSON download.

**More → New conversation** clears chat while retaining the document index. Replacing or removing the source clears its associated research. Failed questions offer a retry button.

Follow-up questions are rewritten into standalone retrieval queries using the last two conversation turns. Answers are instructed to preserve periods, currencies, and units and to show inputs and formulas for calculations. Citation validation checks IDs, not whether a claim is true.

## Checked financial figures

Try **“What were capital expenditures in fiscal 2025?”**, **“Calculate operating margin in fiscal 2025”**, or **“What was revenue growth from fiscal 2024 to fiscal 2025?”** with the Apple filing. Supported questions read complete statement pages and calculate answers with decimal arithmetic, without embeddings or answer-generation calls. The research planner still uses the local model.

The small **Figures checked against statement rows** caption identifies this path. Open **Figure checks** to inspect the exact row, year column, signed cell, scale, currency and formula inputs. The same record is preserved in JSON and Markdown downloads. Missing or conflicting evidence produces an explicit explanation instead of a guessed value. Broader model interpretations are labeled **figures not numerically checked**.

This fixes the observed capex scale error by producing **$12,715 million ($12.715 billion)** from the original `(12,715)` cell. It is a narrow supported-layout feature, not a validator of all model-written claims. Explicit years and supported annual metrics are required; unfamiliar tables, narrative explanations, forecasts, adjusted figures, quarterly results and per-share amounts remain outside its scope. Uploaded PDFs must establish their own currency; a `$` sign alone is not assumed to mean USD. See the [full contract](NUMERICAL_EVIDENCE.md).

## Competitor research

Try **“Is this company's operating margin strong?”** or **“Compare Apple with Microsoft on operating margin.”** in the Apple example. The planner recognizes the need for outside evidence, invokes the annual-financials tool, and produces a sourced comparison. A follow-up can change the metric or named peers. Research details retain the plan, chosen peers, tool outcomes, and data origins.

The current external-data scope is nonfinancial US 10-K filers, one annual period per company, a focal company and up to three named peers. Metrics are consolidated USD revenue, operating income, operating margin, and operating cash flow. The comparison table, margins, and higher/lower conclusions are produced in Python from source values. The model plans research; it does not generate the peer tool's factual answer. Missing/conflicting facts remain missing. Quarterly, segment, multi-year and other metric requests require a supported narrower question. These totals do not establish why companies differ or determine their creditworthiness.

Automatic benchmarks currently exist only for the Apple example: Microsoft and Alphabet, explicitly labeled as broad technology benchmarks with different business mixes. Other companies require user-named peers. This is not an automated industry-wide competitor discovery service.

The default cutoff for the Apple example is **October 31, 2025**, its filing date. Other uploads default to today's date because the app cannot reliably infer their publication date. The cutoff is visible in every comparison; ask for an explicit date or current data to change it. Fiscal years can differ, and exact periods are preserved in the table and sources. Data selection uses the original 10-K accession, excluding later filings' restated facts; amendments are flagged for review rather than silently applied.

Live retrieval uses the official SEC submissions and Company Facts JSON APIs. Configure an application identity and actual operator contact for SEC access:

```bash
export SEC_USER_AGENT='Your application name your-real-contact@your-domain.com'
streamlit run app.py --server.address=127.0.0.1
```

For local development, an ignored `.sec-user-agent` file beside `sec_data.py` can contain the same single-line identity; `SEC_USER_AGENT` takes precedence. Keep the file private (mode 600). It is excluded from Git and Docker images. Containers require the environment variable. The default development identity identifies this repository.

Requests stay on allowlisted SEC endpoints, verify TLS, reject redirects, and use process-wide spacing, a bounded one-hour public-data cache, request/response limits and limited retries. HTTP `Retry-After` cooldowns are respected across sessions; a cooldown longer than the remaining research budget returns an actionable error. Transport failures have explicit categories, so data/identity errors cannot accidentally select fallback data based on error-message wording. An operator contact does not guarantee SEC network access. HTTP 403 is reported and never bypassed. Multi-process deployment will need a shared rate limiter.

If retrieval is unavailable, only requests matching the bundled **2025-10-31** cutoff and supported companies/years may use the small [verified annual-report snapshot](assets/filings/PEER_SOURCES.md). Snapshot use is conspicuous in the answer and exports; current-data requests never silently receive those old figures. Initial requests returned HTTP 403; configuring an actual operator contact resolved access on this development network. On September 14, 2026 UTC, fresh SEC retrieval succeeded for Apple, Microsoft, Alphabet, Amazon and NVIDIA. All 20 dated financial benchmark cases matched the independently transcribed report values using live requests with cache reads disabled. This verifies those cases, not universal SEC availability or all accounting treatments.

Credit/S&P methodology scoring is the next phase, **not implemented**. Credit-rating requests state this limitation. No S&P documents or rating matrices are bundled. See [INTELLIGENCE.md](INTELLIGENCE.md) for the researched design, scope, and next steps.

## What changed

- Uploaded files are parsed in memory, without a shared `uploaded.pdf` file.
- Documents, indexes, chat, and entity results belong to one Streamlit session. Replacing or removing a document clears all associated state.
- Index construction is atomic: a failed embedding call cannot publish a partially built index or silently substitute an incompatible one.
- Questions run only on submission. Settings changes and downloads do not submit them again.
- Model failures preserve prior research and present a retry message without exposing provider payloads.
- Source metadata is included in the model's evidence, and full excerpts remain available for review.
- Extraction uses the selected model and full selected pages, retains results across reruns, and reports whether returned offsets exactly match the source.
- Public reranker resources, immutable example bytes, and bounded public SEC responses may be globally cached. Private document indexes and research remain session-scoped. See [Streamlit session state](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state) and [resource caching](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource).

## Architecture

The diagram below is the document-retrieval branch. Before entering it,
`intelligence.py` applies explicit follow-up edits or asks the local model for a
validated research plan. Peer questions instead call the bounded SEC adapter,
validate annual facts, calculate the comparison, and attach external evidence.

```text
PDF bytes → validate + extract pages → session-owned page documents
                                     ↓ for semantic research / embedding change
                             split (1,000 chars, 150 overlap)
                                     ↓
                           Ollama embeddings → FAISS
                                     ↓
question → resolve follow-up → retrieve → optional cross-encoder
                                     ↓
                   bounded source-page expansion → evidence JSON
                                                                      ↓
                                                        Ollama answer with [S#]
                                                                      ↓
                                                    citation checks + full excerpts
```

Before semantic retrieval, the document branch scans complete recognized statement
pages within a separate bounded budget. Supported figures and formulas return
directly with cell-level evidence; other questions take the diagrammed model path
and are explicitly marked numerically unchecked.

- `app.py`: Streamlit UI and user-triggered actions.
- `ui.py` and `assets/finread.css`: reusable conversation controls and the responsive visual system.
- `sample_document.py` and `assets/filings/`: the complete Apple FY2025 Form 10-K, original URLs, and a checksum for provenance. The bundled example works without fetching SEC data at runtime.
- `finread.py`: document validation, session lifecycle, evidence and answer contracts.
- `numeric_evidence.py`: conservative statement/column parsing, exact decimal amounts, supported formulas and numerical audit records.
- `intelligence.py`: structured query planning, peer research, deterministic comparison and trace.
- `sec_data.py`: bounded SEC JSON adapter, company resolution, annual-period and fact validation.
- `peer_snapshot.py`: explicitly dated, verified public annual-report fallback; no invented API responses.
- `services.py`: local embedding in batches of at most 16 chunks, retrieval, reranking, generation, and extraction. Indexes remain unpublished until every embedding batch succeeds.
- `finread.PageContextRetriever`: expands ranked chunks to their original source pages when within a 6,000-character page / 18,000-character total budget, preserving statement headings and rows. Oversized pages retain ranked excerpts.
- `benchmarks/`: 44 source-checked development cases and recorded live SEC response subsets. See [benchmark instructions and limits](benchmarks/README.md).
- `tests/`: deterministic unit and Streamlit interaction tests without model calls.
- `scripts/smoke_models.py`: opt-in smoke test against installed local models.
- `.github/workflows/tests.yml`: CI test job using the existing requirements.

## Verification

```bash
python -m unittest discover -s tests -v
# Optional: requires Ollama, both default models, and cached reranker weights
python scripts/smoke_models.py
# Optional: local Ollama planner + public SEC/snapshot comparison (no embeddings)
python scripts/smoke_research.py
```

The offline suite covers document/session behavior plus research-plan validation, source navigation, comparison arithmetic, data cutoff/period/unit checks, conflicting tags, snapshot restrictions, model failures and exports. The opt-in research smoke script exercises real local-model routing and the comparison flow. It explicitly reports snapshot use; passing it does not prove live SEC connectivity or semantic correctness across all financial questions. See `DESIGN.md` for interaction principles and `INTELLIGENCE.md` for research references.

## Accuracy and live-data checks

```bash
# Network-free regression benchmark; model-dependent cases are explicitly skipped
python scripts/evaluate_financials.py --output evaluation-results/offline.json
# Current operational readiness; no cache reads or fallback
python scripts/check_sec.py
# Fresh historical financial comparisons checked against annual reports
python scripts/evaluate_financials.py --mode live --only financial --output evaluation-results/live.json
# All 44 cases with installed local models and recorded SEC inputs
python scripts/evaluate_financials.py --mode model --output evaluation-results/model.json
```

Reports distinguish live retrieval, recorded response replay and injected failure
scenarios. They include per-case failures and separate numeric, period, source,
retrieval and routing checks. This is a development set, not a held-out or
analyst-certified accuracy claim. Document evidence-hit checks do not prove every
claim is supported. See [benchmark documentation](benchmarks/README.md) and
[validation record](VALIDATION.md) for measured outcomes and limitations.

## Production status and next milestones

This is a tested local application foundation, **not yet a hosted multi-user production service**.

1. **Deployment foundation:** define users and hosting, add authentication/authorization, tenant-scoped durable storage, explicit retention/deletion, background indexing jobs, quotas, and deployment observability.
2. **Financial accuracy:** independently evaluate additional filings and layouts, extend supported source-cell extraction, and measure coverage as well as correctness. Add OCR, broader metric definitions, and analyst-reviewed evaluation. Checked statement figures do not validate every claim in model interpretations.
3. **Product depth:** validated peer discovery, richer cross-filing evidence, credit methodology implementation, saved document libraries, and side-by-side original PDF navigation.
4. **Release hardening:** lock all dependency versions, add dependency/security scans and isolated PDF parsing with resource limits, load-test concurrent model work, and exercise deployment/backup recovery.

Current limitations: refreshes or restarts may clear session data; PDF text extraction may flatten tables; scanned pages require external OCR; financial interpretation is model-generated; prompts are not a security boundary against malicious documents. Timeouts bound model HTTP calls, but there is no durable job queue, cancellation, or per-user rate limiting. Dependencies remain unchanged from the original project.

A legacy `uploaded.pdf` is already tracked in the original repository. The app no longer reads or writes it; build exclusions prevent it entering new container images. Review it separately before publishing a release.

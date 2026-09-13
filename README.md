# FinRead

A local financial document research app built with Streamlit, Ollama, FAISS, and LangExtract. Upload a filing, ask questions, and inspect the page-level evidence behind each answer.

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
2. Type a question in the main conversation box. The first question builds the index; subsequent questions reuse it. Ask follow-up questions naturally in the same box.
3. Click a source chip such as **S1 · p. 2** to read the complete excerpt in the adjacent **Document details** panel. **Read full page** opens the corresponding page without regenerating the answer.
4. Open **Document details** and switch to **Pages** to browse text, or **Figures** to extract entities from up to five selected pages. Model and retrieval controls live under **Settings** in the sidebar.
5. Use **More** to download JSON research or readable Markdown notes with source excerpts. Entity results have their own JSON download.

**More → New conversation** clears chat while retaining the document index. Replacing or removing the source clears its associated research. Failed questions offer a retry button.

Follow-up questions are rewritten into standalone retrieval queries using the last two conversation turns. Answers are instructed to preserve periods, currencies, and units and to show inputs and formulas for calculations. Citation validation checks IDs, not whether a claim is true.

## What changed

- Uploaded files are parsed in memory, without a shared `uploaded.pdf` file.
- Documents, indexes, chat, and entity results belong to one Streamlit session. Replacing or removing a document clears all associated state.
- Index construction is atomic: a failed embedding call cannot publish a partially built index or silently substitute an incompatible one.
- Questions run only on submission. Settings changes and downloads do not submit them again.
- Model failures preserve prior research and present a retry message without exposing provider payloads.
- Source metadata is included in the model's evidence, and full excerpts remain available for review.
- Extraction uses the selected model and full selected pages, retains results across reruns, and reports whether returned offsets exactly match the source.
- Only the public reranker model is globally cached. See [Streamlit session state](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state) and [resource caching](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource).

## Architecture

```text
PDF bytes → validate + extract pages → session-owned page documents
                                     ↓ on first question / embedding change
                             split (1,000 chars, 150 overlap)
                                     ↓
                           Ollama embeddings → FAISS
                                     ↓
question → resolve follow-up → retrieve → optional cross-encoder → evidence JSON
                                                                      ↓
                                                        Ollama answer with [S#]
                                                                      ↓
                                                    citation checks + full excerpts
```

- `app.py`: Streamlit UI and user-triggered actions.
- `ui.py` and `assets/finread.css`: reusable conversation controls and the responsive visual system.
- `sample_document.py` and `assets/filings/`: the complete Apple FY2025 Form 10-K, original URLs, and a checksum for provenance. The bundled example works without fetching SEC data at runtime.
- `finread.py`: document validation, session lifecycle, evidence and answer contracts.
- `services.py`: local embedding, retrieval, reranking, generation, and extraction.
- `tests/`: deterministic unit and Streamlit interaction tests without model calls.
- `scripts/smoke_models.py`: opt-in smoke test against installed local models.
- `.github/workflows/tests.yml`: CI test job using the existing requirements.

## Verification

```bash
python -m unittest discover -s tests -v
# Optional: requires Ollama, both default models, and cached reranker weights
python scripts/smoke_models.py
```

The 38-test suite covers page provenance, invalid/encrypted/scanned PDFs, upload bounds, session isolation, index invalidation and failed rebuilds, follow-up retrieval, citation checks, duplicate submissions, settings reruns, retries, persistent extraction results, sample onboarding, direct conversational follow-ups, and source-to-page navigation. See `DESIGN.md` for the redesign's interaction principles and reference projects.

## Production status and next milestones

This is a tested local application foundation, **not yet a hosted multi-user production service**.

1. **Deployment foundation:** define users and hosting, add authentication/authorization, tenant-scoped durable storage, explicit retention/deletion, background indexing jobs, quotas, and deployment observability.
2. **Financial accuracy:** build a labeled filing evaluation set; measure retrieval recall, citation support, abstention, and period/unit accuracy. Add table-aware extraction, OCR, and deterministic financial calculations before offering computed KPIs.
3. **Product depth:** saved document libraries, multi-filing comparisons, side-by-side original PDF navigation, and structured financial exports.
4. **Release hardening:** lock all dependency versions, add dependency/security scans and isolated PDF parsing with resource limits, load-test concurrent model work, and exercise deployment/backup recovery.

Current limitations: refreshes or restarts may clear session data; PDF text extraction may flatten tables; scanned pages require external OCR; financial interpretation is model-generated; prompts are not a security boundary against malicious documents. Timeouts bound model HTTP calls, but there is no durable job queue, cancellation, or per-user rate limiting. Dependencies remain unchanged from the original project.

A legacy `uploaded.pdf` is already tracked in the original repository. The app no longer reads or writes it; build exclusions prevent it entering new container images. Review it separately before publishing a release.

# Financial accuracy benchmark

This is a small, public **development benchmark**, not a financial accuracy
certification or a held-out evaluation. Labels were checked against company
statements by the implementation author. They have not received independent
analyst review. Failures found here informed the implementation; a separate,
blinded test set is still required before making broader quality claims.

## Dataset and provenance

`financial_cases.json` contains 44 cases:

- 20 financial questions: four metrics for each of Apple, Microsoft, Alphabet,
  Amazon and NVIDIA, with a named comparison company and the 2025-10-31 cutoff.
- 12 document questions: reported amounts, previous-year figures, balance-sheet
  totals and cash-flow figures in Apple's bundled FY2025 filing.
- 8 routing questions: implicit benchmarking, credit, unsupported comparisons,
  document-only requests and conversational edits to metric, peer and cutoff.
- 4 failure injections: unavailable current data, unavailable requested year,
  missing revenue and an unresolved peer. These are deliberately altered scenarios,
  not additional observations from company reports.

Expected financial amounts are separately transcribed in USD millions from the
original statements below, then explicitly converted to USD for comparisons.
Margins are independently calculated and recorded to five decimal places; their
absolute tolerance is 0.0001 percentage points. Reported USD values must match
exactly. The evaluator imports neither the production snapshot values nor the
production margin function to construct expected answers.

| Company / annual period | Reference statements |
| --- | --- |
| Apple FY2025 | [10-K](https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm), consolidated operations (report p. 29 / PDF p. 32), balance sheet (31 / 34), cash flows (33 / 36) |
| Microsoft FY2025 | [Annual report](https://www.microsoft.com/investor/reports/ar25/index.html), Income Statements and Cash Flows Statements |
| Alphabet FY2024 | [10-K](https://www.sec.gov/Archives/edgar/data/1652044/000165204425000014/goog-20241231.htm), consolidated income and cash flow statements |
| Amazon FY2024 | [10-K](https://www.sec.gov/Archives/edgar/data/1018724/000101872425000004/amzn-20241231.htm), consolidated operations and cash flow statements |
| NVIDIA FY2025 | [10-K](https://www.sec.gov/Archives/edgar/data/1045810/000104581025000023/nvda-20250126.htm), consolidated income and cash flow statements |

`fixtures/` contains **subsets of actual successful SEC JSON responses**, captured
on 2026-09-14 UTC. They are not responses synthesized from the expected answers.
Each file records endpoint URLs, capture time, hashes of the normalized full JSON,
and selection rules. The subsets retain original observation metadata, supported
concepts across currencies and accessions for the reference period end, and recent
annual filing rows. They are intentionally narrow; they do not emulate all SEC
responses. Later restatements remain in the inputs so original-accession selection
is exercised. The application does not load these evaluation fixtures.

Refresh candidate inputs without changing the reference answers:

```bash
python scripts/capture_sec_fixtures.py --output-dir evaluation-results/new-fixtures
```

Review diffs before replacing any versioned fixtures. The capture script will not
overwrite existing files. Check publication and period metadata against the filing
when changing a reference case. Document cases use one-based PDF pages; explicitly
accepted alternate statement pages are listed for repeated income/equity figures.

## Running and interpreting results

```bash
# No models or network: 20 financial cases, 3 follow-ups, 4 failure injections.
python scripts/evaluate_financials.py --output evaluation-results/offline.json

# Fresh SEC requests; all 20 financial cases must use live data, never a snapshot.
python scripts/evaluate_financials.py --mode live --only financial --output evaluation-results/live.json

# All 44 cases, using installed local Ollama models and recorded SEC responses.
python scripts/evaluate_financials.py --mode model --output evaluation-results/model.json

# Separate operational probe: current annual records for five companies.
python scripts/check_sec.py --output evaluation-results/sec-current.json
```

`--only financial`, `--only document`, `--only routing` or `--only abstention`
selects a subset. The model runner uses the same research entry point as the app,
with llama3, mxbai-embed-large, the cached MiniLM reranker, k=8 and top_n=4. Set
`FINREAD_MODEL` and `OLLAMA_BASE` to change the model/server. It does not download
models. Index construction occurs only for semantic document research. Supported
statement questions now use the complete statement scan and deterministic answers;
the model still selects the research route. Reports retain `research.verification`
so this path is distinguishable from model-generated prose.

JSON reports contain per-case outcomes, timings, source evidence, plans and
per-dimension scores. JSONL checkpoints preserve completed cases if interrupted.
Failures and errors produce a nonzero exit code. **Skipped cases never count as
passes**, and offline success does not establish model or live-network correctness.
CI runs only the offline benchmark and saves the report as an artifact.

Financial checks cover exact values, margin arithmetic, displayed table cells,
higher/lower comparisons, company identity, periods, cutoff, units, citation IDs
and source URLs/figures. Live mode additionally rejects any snapshot result.

Document checks measure expected amount/unit presence, year mention, retrieval of
an accepted statement row, and whether an actual cited excerpt contains that row.
The amount tolerance is half a USD million, allowing explicitly scaled answers.
Only the payments/outflow-magnitude case accepts an absolute signed amount.
These are **evidence-hit proxies**, not claim entailment: a correct amount can
coexist with an unsupported explanation. Extra claims, contradictions, correct
year-to-column binding and abstention for arbitrary document questions still need
human review or a stronger claim verifier. Errors are included in case outcomes;
per-dimension denominators include only cases where an answer could be scored.

The numerical-evidence milestone keeps these original labels and scoring rules
unchanged. Its separate regression tests check exact row/year/base-unit records,
formulas and unsupported cases; a page-level evidence hit does not substitute for
those checks. `tests/fixtures/capex-model-failure.json` preserves the original
wrong answer and public source excerpts. Synthetic adversarial tests establish
software behavior, not accuracy on an independent filing set.

## Design references

- [SEC API documentation](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
  and [automated access FAQ](https://www.sec.gov/about/webmaster-frequently-asked-questions):
  official data, declared operator identity, bounded requests and cache-aware access.
- [EdgarTools](https://github.com/dgunning/edgartools): considered for data access;
  the existing narrow adapter is sufficient here, with no new dependency.
- [Ragas metrics](https://docs.ragas.io/en/v0.2.9/concepts/metrics/available_metrics/):
  separate retrieval, answer and reference-based measures. This benchmark uses
  explicit financial checks rather than a paid or model-based grading service.
- [LangChain parent-document retrieval](https://reference.langchain.com/python/langchain-classic/retrievers/parent_document_retriever/ParentDocumentRetriever):
  retrieve small chunks, then recover larger context. FinRead uses its existing
  session-owned PDF pages, capped at 6,000 characters per expanded page and 18,000
  characters total, avoiding a new document store or re-indexing requirement.

### Reviewed label corrections

Dataset version 2 also accepts the product/services table on Apple PDF page 26
for services revenue, and the operating-expense table on PDF page 27 for R&D.
Both contain the exact amount, units and annual columns; restricting citations
to the consolidated statement on page 32 created false negatives. This changes
accepted evidence locations only, not financial values, tolerances or questions.
Saved model observations can be re-scored without re-generating answers:

```bash
python scripts/evaluate_financials.py --rescore evaluation-results/model-verified.json --output evaluation-results/model-rescored.json
```

The original report is preserved; the new report records the old/new dataset
versions and rescore time. Runtime errors and observed route failures stay failed.

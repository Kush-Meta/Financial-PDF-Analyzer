# Roadmap and release criteria

Status after the numerical-evidence milestone. This is a prioritized engineering
backlog, not a claim that the application is already production-ready.

## Delivered

- Simplified conversational UI, actual Apple SEC filing example, evidence
  inspection, exports and session isolation.
- Bounded competitor research with official SEC data, explicit periods/source
  identity and deterministic comparisons; narrow, clearly dated fallback.
- Verified live access for five companies, reproducible captured SEC inputs,
  a 44-case development benchmark, 96 offline tests and CI replay/report setup.
- Planner ticker/year handling, bounded original-page context and 16-chunk
  embedding batches, with regression coverage.
- Direct answers for supported annual statement figures, operating margin and
  revenue growth, with exact row/year/unit records and deterministic arithmetic.
  Complete statement lookup skips index creation for these questions. The UI and
  exports distinguish checked figures, insufficient evidence and unchecked prose.

The development benchmark now passes **44/44** using the real local planner and
deterministic statement/peer answers. The earlier 43/44 observation and incorrect
capex answer are preserved. See [results](benchmarks/RESULTS.md). CI configuration exists; successful
execution of the new job on GitHub was not verified during that milestone.

## 1. Expand numerical coverage

**Delivered first scope:** the recorded capex regression now renders $12,715
million from the statement cell, and supported annual figures/formulas have
reviewable source records. Independent synthetic adversarial tests cover wrong
years, units, rows, currencies, scope, conflicting data and zero denominators.
See [Numerical evidence](NUMERICAL_EVIDENCE.md).

**Delivered second-issuer slice:** an offline Alphabet FY2024 statement fixture
covers consolidated income and cash-flow layouts (`Revenues`, `Income from
operations`, `Net cash provided by operating activities`), including operating
margin from those cells. Operating-cash-flow aliases now include Microsoft-style
`net cash from operations`, with counterexamples so investing cash and
unsupported segment/adjusted income-statement headings still abstain. See
`tests/fixtures/alphabet-2024-statements.json`.

**Delivered third-issuer + formula slice:** Microsoft FY2025 Income Statements /
Cash Flows Statements headings are recognized; net margin and free cash flow are
checked formulas (`net_income / revenue * 100` and
`operating_cash_flow - abs(capital_expenditure)`). Capex aliases include
`additions to property and equipment`. See
`tests/fixtures/microsoft-fy2025-statements.json`.

**Remaining gap:** this still does not validate arbitrary model-written numerical
claims or reconstruct unfamiliar PDF tables. Extend source extraction and question
coverage using additional filings and held-out layouts, while retaining explicit
unsupported cases. Keep reported data distinct from inferred or adjusted values;
do not turn the current caption into a blanket guarantee about an interpretation.

**Acceptance evidence:** check additional filings and held-out questions, reporting
coverage and abstention as well as correctness. Preserve all original failures.
The Apple development set, Alphabet/Microsoft fixtures and synthetic tests alone
do not meet that broader bar.

## 2. Establish a held-out financial evaluation set

**Problem:** the current development set was used to improve the implementation
and covers five technology-oriented companies, with document QA on Apple only.

**Next deliverable:** independently reviewed questions and source labels spanning
additional nonfinancial sectors, fiscal calendars, table layouts and unsupported
requests. Separate reported values, calculations, retrieval evidence and
interpretive claims. Add document-only abstention and adversarial-input cases.

**Acceptance evidence:** publish per-category counts, failures, coverage,
retrieval/source support, numerical checks, abstention and latency. Freeze the
held-out labels before evaluation; log justified label corrections and retain
original reports. Agree release thresholds before using scores to approve a beta.

## 3. Prepare a hosted beta

**Next deliverable:** choose the target users and hosting model, then implement
sign-in/authorization, tenant-scoped durable document/research storage, explicit
retention/deletion, background jobs, cancellation, quotas and operational metrics.
Keep the current single question box and expandable evidence interface.

**Acceptance evidence:** a reproducible dependency set and Docker build; CI runs;
separation between users' uploads, indexes, research and downloads; concurrent
model-load tests; job recovery after restarts; storage/backup recovery; shared SEC
rate limiting if using multiple processes. Define PDF parsing resource limits
and evaluate untrusted-document behavior. Review the legacy tracked PDF before
publishing a release. None of these deployment guarantees follows from local
unit-test success.

## 4. Expand competitor research

**Next deliverable:** sourced peer discovery, multi-year comparisons and deeper
filing evidence where supported. Distinguish direct competitors from broader
financial benchmarks. Preserve explicit fiscal windows and report availability
cutoffs; explain missing or incomparable data.

**Acceptance evidence:** reviewed peer-selection cases, period alignment and
restatement tests, additional metric definitions and traceable inputs. Quarterly,
segment, debt and adjusted credit metrics need their own accounting contracts.
Do not quietly map unsupported requests onto the existing four annual metrics.

## 5. Add a methodology-led credit assessment

**Next deliverable:** select one nonfinancial sector, version the applicable
methodology documents, and encode reviewed formulas, thresholds, adjustments,
exceptions and required judgments. Link every rule to its source. Verify rights
before bundling third-party methodology content in the repository.

Start with cited strengths/weaknesses, reported-to-adjusted reconciliations,
missing inputs and downside scenarios. Qualitative judgments should remain
explicit and reviewable.

**Acceptance evidence:** rule-boundary and exception tests, incomplete-input
handling, version-change tests and analyst-reviewed cases. Only introduce an
indicative rating when mandatory inputs and framework rules are complete and
validated. Identify it as FinRead's assessment, distinct from a published S&P
rating. There is no methodology scoring engine or bundled S&P matrix today.

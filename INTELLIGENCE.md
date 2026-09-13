# FinRead intelligence layer

Researched on September 13, 2026. The first bounded competitor-research release is
implemented in `intelligence.py` and `sec_data.py`; the credit methodology sections
below remain a roadmap. See README.md for the implemented scope and limitations.
The first release uses model-based planning with deterministic peer comparisons;
free-form comparative interpretation is deferred until claim-level validation is
available. Valid citation IDs alone did not prevent contradictory live-model prose.

## Product direction

Build toward a credit research assistant whose conclusions can be checked against
company disclosures and explicit analytical rules. Implement competitor research
first because it supplies the external evidence foundation required by both paths.
Keep the existing conversation interface: users ask a question; the application
selects the necessary tools and explains material assumptions within the answer.

| First release | User outcome | Main difficulty |
| --- | --- | --- |
| Competitor research | Compare a company's performance with relevant peers using cited filings | Defensible peer selection and genuinely comparable figures |
| Methodology-led credit analysis | Explain credit strengths, weaknesses, missing inputs, and sensitivity to assumptions | Correctly implementing a versioned methodology, including adjustments and judgment |

These are complementary capabilities. Peer results can inform competitive-position
analysis; comparable revenue or leverage alone cannot determine creditworthiness.

## Existing implementation and extension points

`finread.answer_question` currently rewrites follow-up questions, retrieves uploaded
PDF excerpts, and generates an answer. `services.py` provides local Ollama and
retrieval integrations. The new competitor tool wraps that existing document path;
there is no methodology scoring engine yet.

The new orchestrator runs before answer generation and preserves the document-only
path for questions that need no external research. Extend the evidence contract,
source inspector, and exports together: external filings must open their own source,
not a similarly numbered page in the uploaded PDF.

## Bounded research workflow

1. Resolve the question using conversation context, the document's company identity,
   reporting period, and an explicit research cutoff date.
2. Produce a validated plan: document research, peer research, credit analysis, or a
   combination. An ambiguous identity, sector, or comparison should trigger one
   targeted question. A failed planner must not silently answer an external question
   as if it had researched peers.
3. Retrieve filing evidence and execute only the tools required by the plan. Limit
   peers, requests, elapsed time, and retries; report useful partial results on failure.
4. Normalize facts and calculate comparisons in deterministic Python code.
5. Generate an explanation from those results. Separate reported facts, calculated
   metrics, analyst assumptions, and unresolved judgments.
6. Validate source references and numerical claims. A valid citation ID alone does
   not establish that a claim is entailed by its source.

The model selects tools and explains evidence. It cannot invent tool results,
execute arbitrary code, choose arbitrary network destinations, or modify scoring
rules. Retrieved filings and methodology passages remain untrusted data.

## Patterns evaluated in existing projects

- [EdgarTools tool reference](https://github.com/dgunning/edgartools/blob/main/docs/ai/mcp-tools.md)
  separates company discovery, filing reading and company comparison. FinRead adopts
  those boundaries and carries reporting periods and source identity across them.
- [Dexter](https://github.com/virattt/dexter) describes explicit planning, tool execution,
  validation and step limits. FinRead uses a smaller fixed workflow with at most
  three peers and bounded requests, rather than an open-ended loop.
- [Agentic FinSearch](https://fingpt.readthedocs.io/en/latest/introduction.html) combines
  local-file search with external financial tools. FinRead similarly routes by the
  evidence needed while preserving its existing conversational UI.

Explicit metric/peer/cutoff edits to a previous comparison are handled as validated
state transitions before model planning. Live tests found that relying exclusively
on the small local model lost peer context on short follow-ups.

These are architectural references, not copied implementations or evidence of
FinRead's financial accuracy. The SEC JSON APIs already supply the narrowly scoped
data this first release needs, so the implementation uses them directly rather
than adding a filing parser or agent framework. Requirements remain unchanged.
EdgarTools remains a candidate when the scope grows to filing sections and broader
metrics. No paid market-data API is needed for the implemented release.

## Competitor research: proposed first release

Start with US-listed nonfinancial companies, annual filings, a focal company and at
most three peers. Support explicitly named peers first. Automatic suggestions must
carry a sourced business rationale; industry codes or a model's recollection are
candidate discovery signals, not proof that companies are comparable. Distinguish
direct competitors from broader financial benchmarks and permit a conversational
correction to the peer set.

Initial comparisons: revenue, operating income, operating margin, and operating
cash flow. Add debt and credit ratios only once their definitions and accounting
adjustments are implemented. Preserve fiscal start/end dates, currency, units,
consolidation scope, taxonomy tags, filing accession, filing date, retrieval date,
and source links with each fact. Show the formula and inputs for derived metrics.

Default the analysis cutoff to the focal filing's publication date, display it, and
allow a request for current data to change it. Select only filings available by that
cutoff; do not leak later restatements into a historical answer. Show exact fiscal
dates when years do not align. Do not silently substitute a quarter for a year,
combine currencies, use missing data as zero, or compare segment results with
whole-company figures. Missing or conflicting facts remain explicitly unresolved.

The SEC provides submissions and XBRL Company Facts APIs without API keys. Its
documentation also warns that financial calendars differ; standard aggregate facts
do not cover every custom or segment disclosure. Use filing-level evidence when
the aggregate facts are insufficient. [SEC API documentation](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)

Prefer evaluating the free, MIT-licensed EdgarTools library as the retrieval adapter
before writing an EDGAR parser. It already exposes financial statements, filing
objects, and caching/rate-limit support. It is not currently installed in FinRead;
dependency compatibility and a pinned version must be checked before adoption.
The app must still own its peer-selection, comparability, and provenance checks.
[EdgarTools](https://github.com/dgunning/edgartools)

Configure an actual operator identity for SEC requests, respect fair-access limits,
and use bounded retries, caching, timeouts, and an explicit unavailable-data result.
[SEC developer FAQ](https://www.sec.gov/about/webmaster-frequently-asked-questions)

## Methodology-led credit analysis

Select one supported nonfinancial sector before implementing scoring. Version the
applicable methodology documents and record original publication, revision and
effective dates, source location, content fingerprint, and applicable sector.
Check the current criteria registry rather than treating a downloaded PDF as
permanently authoritative. S&P's registry lists Corporate Methodology, Ratios And
Adjustments, sector criteria, liquidity criteria, and management/governance criteria
as related components. [S&P corporate criteria](https://www.spglobal.com/ratings/en/regulatory/ratings-criteria/-/articles/criteria/corporates/filter/all)

S&P describes its own corporate model as accepting analytical assessments and
applying methodology matrices; its indicative outputs support a recommendation to
a credit committee. This supports separating rule execution from the judgments
that supply its inputs. [S&P corpengine description](https://spratings.spglobal.com/ratings/en/regulatory/article/-/view/sourceId/12885054)

Implement three connected components:

- **Methodology retrieval:** find the applicable definitions, exceptions and criteria
  with section/paragraph citations. This helps explain the rules; it is not scoring.
- **Reviewed rules:** encode formulas, thresholds, applicability, required inputs,
  adjustment treatment, and exceptions in version-controlled code/data. Each rule
  references its methodology source. Boundary tests verify the implementation.
- **Credit assessment:** assemble business-risk, financial-risk and liquidity
  evidence; expose qualitative judgments and scenario assumptions for review.
  Carry unknowns through the calculation rather than defaulting them to favorable
  scores or reporting unsupported precision.

Maintain reported and adjusted values separately, including a reconciliation.
S&P's adjustment methodology addresses debt, earnings, cash flow and interest;
reported accounting values should not be relabeled as methodology-adjusted metrics.
[S&P Ratios And Adjustments](https://spratings.spglobal.com/ratings/en/regulatory/article/-/view/sourceId/10906146)

The first deliverable should be a credit assessment with cited strengths, weaknesses,
computed metrics, missing inputs and downside scenarios. Introduce an indicative
rating only when the selected framework and mandatory inputs are complete and
validated. Identify it as FinRead's assessment, distinct from a published S&P rating.
Verify rights for the intended document ingestion and distribution before bundling
methodology content in the public repository; link to sources in the meantime.

## Evidence and evaluation

Use one evidence model with types for uploaded passages, external filing passages,
structured financial facts, calculations, and methodology rules. Record a dependency
chain from conclusion to calculation/rule to source inputs. Retain it in exports.

Evaluate routing on explicit comparisons, implicit benchmarking ("Is this margin
strong?"), follow-ups, ambiguous entities, document-only questions and credit
questions. Evaluate data handling on mismatched fiscal periods, stale filings,
restatements, different units, missing metrics, taxonomy changes and network failure.
For credit rules, test thresholds, exceptions, negative denominators, adjustments,
missing judgments and methodology-version changes.

Use dated public-filing fixtures with independently checked expected facts. Run
integration checks separately from offline tests. Evaluate answers for claim-level
support and coverage; fluent text and citation presence are insufficient. Benchmark
credit judgments against analyst-reviewed cases before representing them as reliable.

## UI behavior

Keep one question box. During research, show a compact status such as "Checking
peer filings". The answer should state the chosen peers and dates, provide a small
comparison table when useful, explain the conclusion, and link directly to evidence.
Keep calculation details and the research trace expandable. Ask for clarification
only when the answer depends materially on an unresolved choice.

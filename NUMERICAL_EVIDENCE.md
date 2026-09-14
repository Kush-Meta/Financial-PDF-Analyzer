# Numerical evidence

FinRead can answer a small set of financial questions directly from supported statement rows and record how each displayed amount was obtained. The app enters through [`finread.statement_answer`](finread.py), which passes complete recognized statement pages to [`numeric_evidence.quantitative_answer`](numeric_evidence.py). The numerical path uses a conservative question grammar, exact row aliases, and Python `Decimal`; it does not ask a model to repair another model's numbers.

The motivating failure was a cited Apple FY2025 capex answer that reported **$12.715 million**. The statement cell was `(12,715)` under an **in millions** header: its cash outflow magnitude is **$12,715 million ($12.715 billion)**. A valid page citation had concealed a 1,000× scale error. The new path constructs supported answers from the cell, year column, unit header, and currency evidence. It does not validate arbitrary financial prose or certify the underlying filing.

## Questions covered

Use explicit fiscal years and a direct request for reported figures. With the bundled Apple filing, examples include:

- “What was Apple's revenue in fiscal 2025?”
- “What were capital expenditures in fiscal 2025? Give the outflow magnitude in USD millions.”
- “What were total assets at fiscal year end 2025?”
- “Calculate operating margin in fiscal 2025.”
- “What was revenue growth from fiscal 2024 to fiscal 2025?”

The grammar accepts one or two distinct years and up to three recognized metrics. The table parser can read up to three year columns. Growth requests must name the earlier year first. The supported metric IDs are:

| Statement | Metrics |
| --- | --- |
| Annual operations/income | `revenue`, `operating_income`, `net_income`, `research_and_development`, `services_revenue` |
| Balance sheet | `cash_and_equivalents`, `total_assets`, `total_liabilities`, `shareholders_equity` |
| Annual cash flow | `operating_cash_flow`, `capital_expenditure`; a matching `net_income` row can also supply reported net income |

Exact aliases live in `ALIASES`. For example, total liabilities must not match the combined liabilities-and-equity row. A bare “Services” row is accepted as services revenue only under a recognized sales/revenue section, so a services cost row cannot substitute for it.

Operating margin is `operating_income / revenue * 100`. Revenue growth is `(current_revenue - prior_revenue) / prior_revenue * 100`. Both require compatible inputs from the same statement and currency, with a positive revenue denominator. Percentages display two decimal places; the research record retains the decimal result and inputs. Capital expenditure output is explicitly an outflow magnitude, with `abs(reported_cash_flow)` recorded separately from the original signed cell.

Recognition is deliberately limited. Explanations, forecasts, adjusted or organic growth, constant-currency results, geographic/segment questions, quarters, per-share figures, and availability cutoffs such as “as of” are outside this contract. Unsupported words are not silently discarded to turn a broader request into a reported-total answer. A named entity must be covered by the source metadata; arbitrary company names are not inferred from the question.

## Request flow

1. `intelligence.py` plans document or competitor research. Its rewritten query can guide later semantic retrieval, but the original user question controls numerical scope.
2. For the document route, `finread.statement_answer` examines every session page containing a recognized statement heading before index creation. The scan includes complete candidate pages, with limits of **12 pages, 12,000 characters per page, and 60,000 characters in total**. Exceeding any limit skips the entire scan; it never drops a candidate just to fit a budget and risk hiding a conflicting row.
3. `finread.evidence_context` assigns source IDs to those statement pages and carries their text and available provenance into the numerical path. This scan is session-local and introduces no shared cache.
4. `quantitative_answer` recognizes the question and calls `table_facts`. Each accepted row must have an explicit statement context, usable year columns, a local monetary unit header, and a complete set of well-formed cells.
5. Supported facts, formulas, and explicit evidence abstentions return before embeddings, reranking, or answer generation. The planner still runs; this path does not eliminate every model dependency.
6. An unsupported question, an absent statement, or a scan exceeding its budget continues to normal semantic retrieval and model analysis. `PageContextRetriever` expands ranked excerpts within its existing context budget. The app passes `check_numbers=False` to this fallback: a ranked subset cannot earn a checked status after the complete scan was skipped. Any generated interpretation is marked `not_checked`.
7. `intelligence.py` preserves the numerical record while adding the research plan. The conversation, source inspector, and exports retain the same evidence.

The lower-level `quantitative_answer` function can also be used with explicitly supplied sources, and `answer_question` exposes a `check_numbers` option. Such callers are responsible for the completeness of their supplied evidence; the app's complete-statement scan contract comes from `research_answer` and `statement_answer` together.

## Evidence requirements and abstention

Flow statements need explicit annual context; a bare year does not establish an annual period. Month and quarter headers are rejected, as are headers indicating forecast, projected, pro forma, adjusted, non-GAAP, segment, geographic, or supplemental tables. A balance-sheet date can establish a reported balance, but a fiscal-year-end answer also needs evidence that the date is the fiscal endpoint.

The parser binds cells to the year columns in their displayed order. It does not use the question's year to fill a missing header, guess a truncated row's missing column, or interpret a dash as zero. Parentheses preserve negative values. Exact duplicate values can be consistent; conflicting metric/year values, currencies, period kinds, or source files prevent a checked answer. Later tables and recognized supplemental sections do not inherit an earlier table's units.

Currency must be established. A bare `$` is insufficient to identify USD. Explicit local currency takes precedence over metadata; no currency conversion occurs. Monetary scale comes from the table's leading unit declaration, before share-count or per-share exceptions. An explicit currency without a multiplier can represent base units; it is not assumed to mean millions.

Only the known bundled Apple filing receives `currency`, `entity_names`, and `fiscal_year_ends` metadata from [`sample_document.py`](sample_document.py). These values document that specific sample's provenance. Ordinary uploaded PDFs do not receive Apple identity, USD, or fiscal endpoints and must establish applicable information themselves. Extending sample metadata requires verifying the new filing's provenance.

For a recognized question with a statement present but insufficient or conflicting evidence, the numerical function returns an explicit explanation with status `unavailable`; the model is not invited to fill the missing number. If the question is outside the grammar, no recognized statement is present, or the complete scan exceeds a limit, the relevant entry point returns `None` and normal model analysis remains available. That prose receives `not_checked`, even if it includes citations or happens to contain a correct amount.

## Research record and UI

The record is attached as `Answer.research["verification"]` and stored with the conversation entry. Its status has a deliberately narrow meaning:

| Status | Meaning | UI caption |
| --- | --- | --- |
| `verified` | The displayed supported figures and formulas were constructed from accepted statement cells. | Figures checked against statement rows |
| `unavailable` | The recognized request could not be supported under these rules. | Not enough evidence to check these figures |
| `not_checked` | The answer used the normal model interpretation path. | Model interpretation · figures not numerically checked |

A checked record contains the schema `version`, method, scope, `facts`, and `calculations`. Each fact preserves its metric, year, base-unit decimal value as a string, currency, scale, raw cell, raw row, statement/year header, unit header, source ID, file, page, period kind, and table ID. Formula records retain their input facts, expression, decimal result, and display precision where applicable. Unavailable records include a reason and no accepted facts. Model interpretation records retain their unchecked scope.

The collapsed **Figure checks** panel exposes accepted rows and the complete numerical record for checked or unavailable results. Source chips open the supporting excerpts. Research JSON retains the record with each answer; readable Markdown notes include it as a JSON research record alongside the answer and excerpts. These exports preserve the calculation trail but are not signed attestations or substitutes for retaining the original filing.

## Design references and limits

The lightweight existing-solutions review considered three relevant patterns:

- [FinQA](https://finqasite.github.io/) pairs financial questions with supporting facts and annotated reasoning programs. Its explicit-input approach informs recording executable formulas and their evidence here.
- [FinVerify](https://github.com/FinVerify/Finverify) describes decimal canonicalization, deterministic correction rules, and constraints between claims. Those components address numerical representation and consistency; FinRead still needs to bind a requested metric and period to a particular source row.
- [finvariant](https://github.com/arikanatakan/finvariant) checks statement relationships such as balance-sheet balancing and cash-flow tie-outs. Statement consistency is a different check from whether an answer used the right cell and year.

For this narrow feature, a local parser using the standard library avoids adding a service or dependency while retaining an inspectable source-to-answer contract. This is a scope decision, not an assessment that those projects are unsuitable for broader financial verification.

The remaining limits are substantial: flattened PDF text can misrepresent table structure; unfamiliar headings, labels, and layouts reduce coverage; continuation pages without recognized headings and disclosures outside recognized statements can contain relevant evidence; and source statements can themselves contain errors or require accounting judgment. The numerical path scans all recognized statement pages within its budget, but this does not establish that every relevant statement or note in the filing was recognized. Over-budget scans receive no numerical certification. Semantic retrieval can still omit evidence in the separate, explicitly unchecked model path. These checks establish neither accounting validity, causal explanations, peer comparability, nor creditworthiness, and do not inspect every numeric claim in model-written answers.

Keep the original erroneous model output as a regression record. Report successful deterministic rendering separately from new model accuracy, and do not treat a benchmark's page-level evidence hit as proof of row/year support. Consult [validation history](VALIDATION.md) and [benchmark results](benchmarks/RESULTS.md) for checks actually run. Expanding the grammar or row aliases should include both source-backed examples and counterexamples for period, unit, entity, and scope confusion.

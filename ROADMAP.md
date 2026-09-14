# Roadmap and release criteria

Status after implementation commit `6a7372e`. This is a prioritized engineering
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

The measured model result is **43/44**, including a remaining substantive numeric
error. See [results](benchmarks/RESULTS.md). CI configuration exists; successful
execution of the new job on GitHub was not verified during that milestone.

## 1. Validate numerical claims in document answers

**Problem:** the model can retrieve/cite the correct table and still convert its
units incorrectly. The recorded capex answer was $12.715 million instead of
$12,715 million. Citation-ID validation cannot detect this.

**Next deliverable:** represent each material numerical claim with its metric,
period, currency, scale, source row and any formula inputs. Check reported values
and calculations deterministically; reject or clearly qualify unverified claims.
Keep reported data distinct from inferred or adjusted values.

**Acceptance evidence:** the capex regression passes for the right reason, and
negative cases catch thousandfold unit errors, prior-year substitutions, wrong
signs, missing denominators and citations to unrelated rows. Check additional
filings and held-out questions; a stronger prompt alone is insufficient evidence
of reliable validation. Preserve the current failing answer as a regression input.

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

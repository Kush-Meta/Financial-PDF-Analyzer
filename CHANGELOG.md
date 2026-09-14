# Changelog

This file records completed development milestones on the working branch. A local
commit is not evidence of a merge to `main`, a GitHub release or a deployment.
See [validation history](VALIDATION.md) for the corresponding checks and limits.

## 2026-09-14 — Statement figures and numerical evidence

- Added complete statement lookup before semantic indexing for supported annual
  questions, with exact row aliases, year-column binding, local units/currency,
  signed Decimal values, and explicit incomplete/conflicting-evidence handling.
- Added deterministic operating-margin and revenue-growth calculations, retaining
  formulas and source inputs. Preserved the original capex failure as a public
  regression fixture; the supported answer now reports $12,715 million correctly.
- Kept original user scope separate from planner query rewrites. Unsupported
  model interpretations remain explicitly numerically unchecked.
- Added a quiet checked-figures caption, collapsed row/formula audit, and retained
  records in JSON/Markdown exports and document benchmark reports. Escaped currency
  signs so paired dollar amounts do not render as Markdown math.
- Added independent synthetic adversarial cases, public-filing regressions,
  no-index and planner-scope tests, UI/export checks and operator documentation.
- The 44-case development run passed with the installed local planner and
  deterministic supported answers. This does not establish held-out filing
  accuracy, a universal claim validator or hosted production readiness.

## 2026-09-14 — Live-data reliability and financial benchmark

Implementation: `6a7372e`.

- Configured local SEC operator identity and verified successful live retrieval
  for Apple, Microsoft, Alphabet, Amazon and NVIDIA.
- Added cache-bypassing readiness checks, typed transport failures, `Retry-After`
  handling and request-origin diagnostics.
- Added 44 development cases, captured actual SEC response subsets, independently
  transcribed reference values, CLI reports and CI benchmark artifact collection.
- Fixed planner ticker/name and fiscal-year normalization, explicit ticker-pair
  order, missing source-page context and whole-filing embedding batch size.
- Validated 96 offline tests, 20/20 live historical comparisons and a 43/44 model
  benchmark. Recorded the remaining capex unit error and transparent source-label
  corrections. See [benchmark results](benchmarks/RESULTS.md).
- Added developer/operations documentation, a documentation index and prioritized
  release criteria in the subsequent documentation commit.

## 2026-09-13 — Bounded competitor research

Implementation: `c9808b9`.

- Added validated research planning, supported peer follow-ups and annual SEC
  financial comparisons with source provenance, cutoffs and period checks.
- Preserved a narrow real-report fallback at an exact historical cutoff.
- Made comparative tables and conclusions deterministic after live-model prose
  contradicted available evidence. Peer research does not build a PDF index.
- Extended source inspection and exports for external company reports.
- Documented the future credit-methodology design without claiming an implemented
  S&P scoring engine. The milestone passed 73 offline tests; live SEC access was
  still unverified at that point and was resolved in the later milestone.

## 2026-09-12–13 — Reliable research workspace and UI

Implementation: `838b157`.

- Added in-memory PDF handling, session-owned indexes/research, atomic index
  construction, explicit submission/retry behavior and source-aware exports.
- Redesigned the workspace around a simple composer, evidence panel and collapsed
  controls; removed the earlier lettermark and starter-card clutter.
- Replaced the fictional onboarding example with Apple's complete FY2025 10-K,
  with provenance, checksum and printed-page mapping.
- Added offline tests, local-model smoke checks, CI configuration and Docker
  loopback binding, non-root execution and health checks.
- Recovered a stale imported-module preview and verified the actual browser.

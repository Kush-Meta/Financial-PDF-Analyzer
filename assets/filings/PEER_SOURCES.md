# Dated comparison snapshot

The optional fallback in `peer_snapshot.py` contains manually verified public
annual-report facts for research **as of October 31, 2025**. It is not live data,
an SEC Company Facts download, or an S&P-adjusted dataset. It is never substituted
for a different cutoff or a fiscal year it does not cover. Checked September 13,
2026. All figures below are reported consolidated US dollars in millions.

| Company | Period start | Period end | Revenue | Operating income | Operating cash flow |
| --- | --- | --- | ---: | ---: | ---: |
| Apple | 2024-09-29 | 2025-09-27 | 416,161 | 133,050 | 111,482 |
| Microsoft | 2024-07-01 | 2025-06-30 | 281,724 | 128,528 | 136,162 |
| Alphabet | 2024-01-01 | 2024-12-31 | 350,018 | 112,390 | 125,299 |

Sources and locations:

- [Apple FY2025 10-K](https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm),
  filed October 31, 2025: Consolidated Statements of Operations (printed page 29,
  bundled PDF page 32); Consolidated Statements of Cash Flows (printed page 33,
  bundled PDF page 36). Values verified against extracted bundled PDF text.
- [Microsoft FY2025 annual report](https://www.microsoft.com/investor/reports/ar25/index.html),
  dated July 30, 2025: Financial Statements / Income Statements (Total revenue,
  Operating income); Cash Flows Statements (Net cash from operations). Values
  verified against the issuer's HTML report. No unverified SEC accession is supplied.
- [Alphabet FY2024 10-K](https://www.sec.gov/Archives/edgar/data/1652044/000165204425000014/goog-20241231.htm),
  filed February 5, 2025: Consolidated Statements of Income (Revenues, Income from
  operations); Consolidated Statements of Cash Flows (Net cash provided by operating
  activities). Values verified against the SEC report.

The Apple example uses Microsoft and Alphabet as **broad technology benchmarks**.
This is a product-design choice, not an assertion that their consolidated businesses
are like-for-like competitors. The cited reports describe different product and
service mixes. The application explicitly discloses that limitation and the different
financial calendars. Users can name other peers in the conversation. No automatic
peer catalog is claimed for other companies in this release.

Reported operating margin is calculated as operating income / revenue × 100, using
the same company's same-period figures. It is not a credit rating or a normalized
industry profitability score.

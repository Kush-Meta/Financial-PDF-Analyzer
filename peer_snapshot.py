"""Small, explicitly dated fallback transcribed from published annual reports.

These are real reported figures, not simulated SEC API responses. See
assets/filings/PEER_SOURCES.md for the verification and selection policy.
"""
from sec_data import AnnualFinancials

SNAPSHOT_CUTOFF = "2025-10-31"
RECORDS = {
    "AAPL": {
        "company": "Apple Inc.", "cik": 320193, "start": "2024-09-29", "end": "2025-09-27",
        "filed": "2025-10-31", "accession": "0000320193-25-000079",
        "url": "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm",
        "values": (416161, 133050, 111482),
        "locators": ("Consolidated Statements of Operations, report p. 29", "Consolidated Statements of Operations, report p. 29", "Consolidated Statements of Cash Flows, report p. 33"),
    },
    "MSFT": {
        "company": "Microsoft Corporation", "cik": 789019, "start": "2024-07-01", "end": "2025-06-30",
        "filed": "2025-07-30", "accession": "",
        "url": "https://www.microsoft.com/investor/reports/ar25/index.html",
        "values": (281724, 128528, 136162),
        "locators": ("Financial Statements: Income Statements, Total revenue", "Financial Statements: Income Statements, Operating income", "Financial Statements: Cash Flows Statements, Net cash from operations"),
    },
    "GOOGL": {
        "company": "Alphabet Inc.", "cik": 1652044, "start": "2024-01-01", "end": "2024-12-31",
        "filed": "2025-02-05", "accession": "0001652044-25-000014",
        "url": "https://www.sec.gov/Archives/edgar/data/1652044/000165204425000014/goog-20241231.htm",
        "values": (350018, 112390, 125299),
        "locators": ("Consolidated Statements of Income, Revenues", "Consolidated Statements of Income, Income from operations", "Consolidated Statements of Cash Flows, Net cash provided by operating activities"),
    },
}


def snapshot(identifier, cutoff, fiscal_year=None):
    if cutoff != SNAPSHOT_CUTOFF:
        return None
    identifier = "GOOGL" if identifier.upper() == "GOOG" else identifier
    key = next((k for k, v in RECORDS.items() if identifier.upper() == k or (identifier.isdigit() and int(identifier) == v["cik"])), None)
    if key is None:
        return None
    record = RECORDS[key]
    if fiscal_year is not None and not record["end"].startswith(str(fiscal_year)):
        return None
    facts = {}
    for metric, millions, locator in zip(("revenue", "operating_income", "operating_cash_flow"), record["values"], record["locators"]):
        facts[metric] = {"value": millions * 1_000_000, "unit": "USD", "start": record["start"],
                         "end": record["end"], "locator": locator,
                         "tag": None, "original_value": millions, "original_unit": "USD millions"}
    return AnnualFinancials(record["company"], key, record["cik"], record["start"], record["end"],
                            record["filed"], record["accession"], record["url"], cutoff,
                            facts, [], "2026-09-13", origin="bundled_annual_report_snapshot")

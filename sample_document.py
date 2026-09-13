"""The complete, bundled Apple FY2025 Form 10-K, with original provenance."""
from functools import lru_cache
from pathlib import Path
import re

from finread import read_pdf

SAMPLE_NAME = "Apple · 2025 Form 10-K"
SAMPLE_SEC_URL = "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm"
SAMPLE_PDF_URL = "https://s2.q4cdn.com/470004039/files/doc_financials/2025/ar/_10-K-2025-As-Filed.pdf"
SAMPLE_PATH = Path(__file__).parent / "assets" / "filings" / "apple-2025-10k.pdf"


@lru_cache(maxsize=1)
def sample_bytes():
    # Only immutable public filing bytes are cached. Parsed documents belong to
    # their browser session, exactly as user uploads do.
    return SAMPLE_PATH.read_bytes()


def sample_pages():
    pages = read_pdf(sample_bytes(), SAMPLE_NAME)
    for page in pages:
        page.metadata["source_url"] = SAMPLE_SEC_URL
        page.metadata["pdf_url"] = SAMPLE_PDF_URL
        # The PDF's cover/contents precede the report's printed page 1.
        footer = re.search(r"Apple Inc\.\s*\|\s*2025 Form 10-K\s*\|\s*(\d+)\s*$",
                           page.page_content)
        page.metadata["page_label"] = footer.group(1) if footer else ""
    return pages

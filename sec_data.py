"""Bounded public SEC JSON retrieval and conservative annual fact selection."""
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from email.utils import parsedate_to_datetime
import json
import math
import os
from pathlib import Path
import re
import ssl
from threading import Lock
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPSHandler, HTTPRedirectHandler, Request, build_opener


class ResearchError(ValueError):
    """A safe, actionable research limitation suitable for display."""


class SecTransportError(ResearchError):
    """Transport failures may permit an explicitly dated fallback; data errors may not."""

    def __init__(self, message, code="network", retry_after=None):
        super().__init__(message)
        self.code = code
        self.retry_after = retry_after


def retry_seconds(value):
    """Support both HTTP Retry-After formats without accepting negative waits."""
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        try:
            when = parsedate_to_datetime(value)
            return max(0, (when - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return 1


def iso_date(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ResearchError("Use a date in YYYY-MM-DD format.")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ResearchError("That reporting date is invalid.") from exc


def cik_number(value):
    value = str(value)
    if not re.fullmatch(r"\d{1,10}", value) or int(value) == 0:
        raise ResearchError("A valid SEC company identifier is required.")
    return int(value)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ResearchError("SEC redirected this request; retrieval stopped.")


# Only public SEC responses are cached. Lock and request spacing are process-wide,
# so separate browser sessions do not each consume the full SEC rate allowance.
_lock = Lock()
_cache = OrderedDict()
_last_request = 0.0
_blocked_until = 0.0
MAX_RESPONSE = 25 * 1024 * 1024


def configured_identity():
    """Read the operator's local identity without putting contact details in Git."""
    if os.environ.get("SEC_USER_AGENT"):
        return os.environ["SEC_USER_AGENT"]
    path = Path(__file__).with_name(".sec-user-agent")
    return path.read_text().strip() if path.is_file() else None


class SecClient:
    def __init__(self, user_agent=None, budget=16, deadline=75, use_cache=True):
        self.user_agent = user_agent or configured_identity() or "FinRead research https://github.com/Kush-Meta/Financial-PDF-Analyzer"
        if any(c in self.user_agent for c in "\r\n"):
            raise ResearchError("SEC_USER_AGENT must be a single-line application identity.")
        self.remaining = budget
        self.deadline = time.monotonic() + deadline
        self.use_cache = use_cache
        self.events = []
        context = ssl.create_default_context()
        # Some macOS Python installations omit system certificate roots. Certifi
        # ships with our existing model dependencies; TLS verification stays on.
        try:
            import certifi
            context.load_verify_locations(certifi.where())
        except ImportError:
            pass
        self.opener = build_opener(HTTPSHandler(context=context), NoRedirect())

    def get(self, path):
        if path == "tickers":
            url = "https://www.sec.gov/files/company_tickers.json"
        elif re.fullmatch(r"submissions/CIK\d{10}(?:-submissions-\d{3})?\.json", path):
            url = "https://data.sec.gov/" + path
        elif re.fullmatch(r"api/xbrl/companyfacts/CIK\d{10}\.json", path):
            url = "https://data.sec.gov/" + path
        else:
            raise ResearchError("Unsupported SEC endpoint.")
        global _last_request, _blocked_until
        for attempt in range(2):
            with _lock:
                cached = _cache.get(url)
                if self.use_cache and cached and time.monotonic() - cached[0] < 3600:
                    _cache.move_to_end(url)
                    self.events.append({"endpoint": path, "status": "cache_hit", "age_seconds": round(time.monotonic() - cached[0], 2)})
                    return json.loads(cached[1])
                if self.remaining <= 0 or time.monotonic() >= self.deadline:
                    raise SecTransportError("SEC research reached its request or time limit. Try fewer companies.", "budget")
                delay = max(0, .25 - (time.monotonic() - _last_request), _blocked_until - time.monotonic())
                if delay >= self.deadline - time.monotonic():
                    raise SecTransportError("SEC requested a longer cooldown than this research allows. Try again later.", "cooldown", delay)
                time.sleep(delay)
                if time.monotonic() >= self.deadline:
                    raise SecTransportError("SEC research reached its request or time limit. Try fewer companies.", "budget")
                self.remaining -= 1
                _last_request = time.monotonic()
                request = Request(url, headers={"User-Agent": self.user_agent, "Accept": "application/json"})
                try:
                    with self.opener.open(request, timeout=max(.1, min(10, self.deadline - time.monotonic()))) as response:
                        raw = response.read(MAX_RESPONSE + 1)
                    if len(raw) > MAX_RESPONSE:
                        raise SecTransportError("SEC response exceeded the research size limit.", "response_size")
                    data = json.loads(raw)
                    if not isinstance(data, dict):
                        raise ResearchError("SEC returned an unexpected data format.")
                    _cache[url] = (time.monotonic(), raw)
                    while len(_cache) > 12:
                        _cache.popitem(last=False)
                    self.events.append({"endpoint": path, "status": "network_ok", "bytes": len(raw)})
                    return data
                except HTTPError as exc:
                    self.events.append({"endpoint": path, "status": "http_error", "http_status": exc.code})
                    if exc.code == 403:
                        raise SecTransportError("SEC denied access from this network. Configure SEC_USER_AGENT with your application and contact, or retry from a permitted network.", "access_denied") from exc
                    delay = retry_seconds(exc.headers.get("Retry-After")) if exc.headers else 1
                    if exc.code in (429, 503):
                        _blocked_until = max(_blocked_until, time.monotonic() + delay)
                    if exc.code not in (429, 500, 502, 503, 504) or attempt:
                        if exc.code == 404:
                            raise ResearchError("SEC has no data at this endpoint (HTTP 404). Check the company identifier.") from exc
                        raise SecTransportError(f"SEC data is unavailable (HTTP {exc.code}). Try again later.", f"http_{exc.code}", delay) from exc
                except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                    self.events.append({"endpoint": path, "status": "network_error"})
                    raise SecTransportError("SEC could not be reached or returned unreadable data. Check the connection and retry.") from exc
            if delay >= self.deadline - time.monotonic():
                raise SecTransportError("SEC requested a longer cooldown than this research allows. Try again later.", "cooldown", delay)
            time.sleep(delay)
        raise SecTransportError("SEC data is temporarily unavailable.")

    def resolve(self, identifier):
        identifier = identifier.strip()
        if identifier.isdigit():
            cik = cik_number(identifier)
        else:
            directory = self.get("tickers")
            matches = [v for v in directory.values() if
                       v.get("ticker", "").casefold() == identifier.casefold() or
                       v.get("title", "").casefold() == identifier.casefold()]
            if len(matches) != 1:
                raise ResearchError(f"Could not uniquely identify {identifier}. Please give its SEC ticker.")
            cik = cik_number(matches[0]["cik_str"])
        profile = self.get(f"submissions/CIK{cik:010d}.json")
        if cik_number(profile.get("cik", 0)) != cik:
            raise ResearchError("SEC company identity did not match the request.")
        if 6000 <= int(profile.get("sic") or 0) <= 6999:
            raise ResearchError("Financial companies need a different comparison framework; this release supports nonfinancial 10-K filers.")
        return profile

    def annual(self, profile, cutoff, fiscal_year=None):
        cutoff_date = iso_date(cutoff)
        cik = cik_number(profile["cik"])
        recent = profile.get("filings", {}).get("recent", {})
        filings = filing_rows(recent)
        # Historical submissions are bounded; do not fetch arbitrary filenames.
        archives = profile.get("filings", {}).get("files", [])
        for archive in archives[:3]:
            if not any(f["form"] == "10-K" and f["filingDate"] <= cutoff and
                       (fiscal_year is None or f["reportDate"].startswith(str(fiscal_year))) for f in filings):
                filings.extend(filing_rows(self.get("submissions/" + archive["name"])))
        candidates = [f for f in filings if f["form"] == "10-K" and
                      f["filingDate"] <= cutoff and f["reportDate"] <= cutoff and
                      (fiscal_year is None or f["reportDate"].startswith(str(fiscal_year)))]
        if not candidates:
            raise ResearchError(f"No supported annual 10-K for {profile['name']} was available by {cutoff}.")
        filing = max(candidates, key=lambda f: (f["reportDate"], f["filingDate"]))
        if fiscal_year is None and (cutoff_date - iso_date(filing["reportDate"])).days > 550:
            raise ResearchError(f"The available annual filing for {profile['name']} is too old for this cutoff.")
        facts = self.get(f"api/xbrl/companyfacts/CIK{cik:010d}.json")
        if cik_number(facts.get("cik", 0)) != cik:
            raise ResearchError("SEC financial facts did not match the company.")
        result = select_annual(profile, filing, facts, cutoff)
        if any(f["form"] == "10-K/A" and f["reportDate"] == filing["reportDate"] and f["filingDate"] <= cutoff for f in filings):
            result.warnings.append(f"{profile['name']}: an amendment exists. These metrics use the original 10-K; review the amendment before relying on them.")
        return result


def filing_rows(columns):
    keys = ("accessionNumber", "form", "filingDate", "reportDate", "primaryDocument")
    if not all(isinstance(columns.get(k), list) for k in keys):
        return []
    return [dict(zip(keys, row)) for row in zip(*(columns[k] for k in keys))]


TAGS = {
    "revenue": ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"),
    "operating_income": ("OperatingIncomeLoss",),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
}


@dataclass
class AnnualFinancials:
    company: str
    ticker: str
    cik: int
    start: str
    end: str
    filed: str
    accession: str
    url: str
    cutoff: str
    facts: dict
    warnings: list
    retrieved: str
    origin: str = "live_sec"
    currency: str = "USD"

    def to_dict(self):
        return asdict(self)


def select_annual(profile, filing, companyfacts, cutoff):
    cik = cik_number(profile["cik"])
    accession = filing["accessionNumber"]
    if not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession):
        raise ResearchError("SEC returned an invalid filing accession.")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+\.html?", filing["primaryDocument"]):
        raise ResearchError("SEC returned an invalid filing document name.")
    end = iso_date(filing["reportDate"])
    found, warnings, starts = {}, [], set()
    for metric, tags in TAGS.items():
        matches = []
        for tag in tags:
            concept = companyfacts.get("facts", {}).get("us-gaap", {}).get(tag, {})
            for fact in concept.get("units", {}).get("USD", []):
                if (fact.get("accn") != accession or fact.get("form") != "10-K" or
                    fact.get("end") != str(end) or fact.get("filed", "9999") > cutoff):
                    continue
                try:
                    start = iso_date(fact.get("start"))
                except ResearchError:
                    continue
                value = fact.get("val")
                if not 330 <= (end - start).days <= 380 or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    continue
                matches.append({"value": value, "tag": "us-gaap:" + tag, "start": str(start), "end": str(end), "unit": "USD"})
        signatures = {(m["value"], m["start"]) for m in matches}
        if len(signatures) == 1:
            found[metric] = matches[0]
            starts.add(matches[0]["start"])
        else:
            warnings.append(f"{profile['name']}: {metric.replace('_', ' ')} is {'conflicting' if matches else 'missing'} in the supported annual USD facts.")
    if len(starts) > 1:
        raise ResearchError(f"Annual metrics for {profile['name']} have inconsistent period starts.")
    if not found:
        raise ResearchError(f"No supported consolidated annual USD metrics were found for {profile['name']}.")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{filing['primaryDocument']}"
    return AnnualFinancials(profile["name"], (profile.get("tickers") or [str(cik)])[0], cik,
                            next(iter(starts)), str(end), filing["filingDate"], accession,
                            url, cutoff, found, warnings, datetime.now(timezone.utc).isoformat())


def operating_margin(financials):
    revenue = financials.facts.get("revenue")
    income = financials.facts.get("operating_income")
    if not revenue or not income or revenue["value"] <= 0:
        return None
    if (revenue["start"], revenue["end"], revenue["unit"]) != (income["start"], income["end"], income["unit"]):
        return None
    return float(Decimal(str(income["value"])) / Decimal(str(revenue["value"])) * 100)

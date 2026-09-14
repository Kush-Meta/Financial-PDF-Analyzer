"""Live-only SEC readiness check: no snapshot and no HTTP cache reads."""
import argparse
from datetime import date, datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sec_data import SecClient, ResearchError, configured_identity, operating_margin


def check(companies, cutoff):
    results = []
    client = SecClient(budget=24, deadline=150, use_cache=False)
    for ticker in companies:
        try:
            record = client.annual(client.resolve(ticker), cutoff)
            missing = set(('revenue', 'operating_income', 'operating_cash_flow')) - set(record.facts)
            results.append({'company': ticker, 'status': 'partial' if missing else 'verified_live',
                            'financials': record.to_dict(), 'operating_margin_pct': operating_margin(record),
                            'missing': sorted(missing)})
        except ResearchError as exc:
            results.append({'company': ticker, 'status': 'unavailable', 'code': getattr(exc, 'code', 'data'), 'message': str(exc)})
            if getattr(exc, 'code', '') in ('access_denied', 'cooldown', 'budget', 'http_429'):
                break
    return {'checked_at': datetime.now(timezone.utc).isoformat(), 'cutoff': cutoff,
            'operator_identity_configured': bool(configured_identity()), 'cache_reads': False,
            'snapshot_allowed': False, 'requested_companies': companies, 'results': results,
            'requests': client.events,
            'passed': len(results) == len(companies) and all(r['status'] == 'verified_live' for r in results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--companies', nargs='+', default=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'])
    parser.add_argument('--as-of', default=date.today().isoformat())
    parser.add_argument('--output', default='evaluation-results/sec-readiness.json')
    args = parser.parse_args()
    if not 1 <= len(args.companies) <= 5:
        parser.error('Choose between one and five companies.')
    report = check(args.companies, args.as_of)
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n')
    for result in report['results']:
        print(result['company'], result['status'], result.get('message', ''))
    print('Live SEC verification:', 'PASS' if report['passed'] else 'FAIL')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

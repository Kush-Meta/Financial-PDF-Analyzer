"""Source-checked financial benchmark. No LLM judge and no production fallback.

Offline replays captured SEC responses; model also measures the real router and
PDF RAG; live fetches SEC again with the HTTP cache disabled. These modes have
separate denominators and must never be presented as equivalent evidence.
"""
from collections import Counter
from dataclasses import asdict
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
import re
import time

from finread import citation_issues
from intelligence import ResearchPlan, competitor_research, plan_question, research_answer
from sec_data import ResearchError, SecTransportError, SecClient

ROOT = Path(__file__).resolve().parent


def load_dataset():
    return json.loads((ROOT / 'financial_cases.json').read_text())


class RecordedSecClient(SecClient):
    def __init__(self):
        super().__init__()
        self.fixtures = [json.loads(p.read_text()) for p in sorted((ROOT / 'fixtures').glob('*.json'))]

    def get(self, path):
        if path == 'tickers':
            return {str(i): {'ticker': p['profile']['tickers'][0], 'title': p['profile']['name'],
                             'cik_str': p['profile']['cik']} for i, p in enumerate(self.fixtures)}
        for fixture in self.fixtures:
            cik = int(fixture['profile']['cik'])
            if path == f'submissions/CIK{cik:010d}.json':
                return json.loads(json.dumps(fixture['profile']))
            if path == f'api/xbrl/companyfacts/CIK{cik:010d}.json':
                return json.loads(json.dumps(fixture['companyfacts']))
        raise ResearchError('Endpoint is absent from recorded benchmark fixtures.')


def previous_comparison():
    plan = ResearchPlan('competitors', 'AAPL', ('MSFT',), '2025-10-31', None,
                        'Compare operating margin for AAPL and MSFT.', identities={'AAPL': 'Apple', 'MSFT': 'Microsoft'})
    return [{'q': 'Compare Apple and Microsoft on operating margin.', 'a': 'Previous comparison.',
             'research': {'plan': asdict(plan)}}]


def close(actual, expected, tolerance):
    return type(actual) in (int, float) and math.isfinite(actual) and abs(actual - expected) <= tolerance


def score_financial(case, answer, companies):
    sources = {s.get('financials', {}).get('ticker'): s for s in answer.sources}
    focal = sources.get(case['company'], {}).get('financials', {})
    metric = case['metric']
    actual = focal.get('operating_margin_pct') if metric == 'operating_margin' else focal.get('facts', {}).get(metric, {}).get('value')
    expected_company = companies[case['company']]
    checks = {'numeric': close(actual, case['expected'], case['tolerance']),
              'entities': set(sources) == {case['company'], *case['peers']},
              'period': all(focal.get(k) == expected_company[k] for k in ('start', 'end', 'filed', 'accession')),
              'cutoff': focal.get('cutoff') == case['cutoff'],
              'unit': focal.get('currency') == 'USD',
              'citation_ids': not citation_issues(answer.text, answer.sources),
              'complete': answer.research.get('status') == 'complete'}
    # Every returned source must match its independently checked annual report,
    # not merely have a plausible URL or a valid citation ID.
    checks['source_facts'] = bool(sources) and all(
        ticker in companies and source['financials'].get('url') == companies[ticker]['url'] and
        all(source['financials'].get('facts', {}).get(m, {}).get('value') == val * 1_000_000
            for m, val in zip(('revenue', 'operating_income', 'operating_cash_flow'), companies[ticker]['values']))
        for ticker, source in sources.items())
    label = {'revenue': 'Revenue ($bn)', 'operating_income': 'Operating income ($bn)',
             'operating_cash_flow': 'Operating cash flow ($bn)', 'operating_margin': 'Operating margin'}[metric]
    row = next((line for line in answer.text.splitlines() if line.startswith('| ' + label + ' |')), '')
    cells = [s.strip() for s in row.split('|')][2:-1]
    expected_cells = []
    for ticker in [case['company']] + case['peers']:
        c = companies[ticker]
        expected_cells.append(f"{c['margin']:.2f}%" if metric == 'operating_margin' else
                              f"{c['values'][['revenue','operating_income','operating_cash_flow'].index(metric)] / 1000:,.2f}")
    checks['displayed_numbers'] = cells == expected_cells
    if case['peers']:
        peer = case['peers'][0]
        c = companies[peer]
        peer_value = c['margin'] if metric == 'operating_margin' else c['values'][['revenue','operating_income','operating_cash_flow'].index(metric)] * 1_000_000
        relation = 'below' if case['expected'] < peer_value else 'above' if case['expected'] > peer_value else 'equal to'
        checks['comparison'] = f"{case['company']}'s reported {metric.replace('_', ' ')} is {relation} {peer}'s" in answer.text
    return checks


def score_document(case, answer):
    """Conservative answer-value presence + cited supporting row, not entailment.

    Detect a wrong magnitude/unit and unknown citations. Additional claims and
    contradictions still require review; report that explicitly in benchmark docs.
    """
    text = answer.text.replace('**', '').replace(',', '')
    amounts = []
    for match in re.finditer(r'(?<![\w.])\$?\s*(-?\d+(?:\.\d+)?)\s*(million|billion|trillion)\b', text, re.I):
        amounts.append(float(match[1]) * {'million': 1, 'billion': 1000, 'trillion': 1_000_000}[match[2].lower()])
    cited_ids = set(re.findall(r'\[(S\d+)\]', answer.text))
    source_value = f"{case['expected']:,}"
    # The row must actually contain the label and number. A citation to the right
    # page alone is insufficient when chunking has dropped the relevant row.
    def supports(source):
        return (source.get('page') in case.get('accepted_pages', [case['page']]) and case['anchor'].lower() in source.get('text', '').lower()
                and source_value in source.get('text', ''))
    return {'numeric_unit': any(close(abs(v) if case.get('outflow_magnitude') else v, case['expected'], case['tolerance']) for v in amounts),
            'period_mentioned': str(case['year']) in text,
            'retrieval_hit': any(supports(s) for s in answer.sources),
            'cited_evidence_hit': any(s['id'] in cited_ids and supports(s) for s in answer.sources),
            'citation_ids': not citation_issues(answer.text, answer.sources),
            'route': answer.research.get('plan', {}).get('route') == 'document'}


def score_route(case, plan):
    checks = {'route': plan.route == case['route'], 'clarification': bool(plan.clarification) == case['clarification'],
              'peers': list(plan.peers) == case['peers']}
    if case['followup']:
        checks['company'] = plan.company == 'AAPL'
        checks['cutoff'] = plan.cutoff == ('2025-09-30' if case['id'] == 'route-cutoff-edit' else '2025-10-31')
        checks['metric'] = ('operating cash flow' if case['id'] == 'route-metric-edit' else 'operating margin') in plan.query
    return checks


def evaluate_abstention(case):
    """Explicit failure injections, scored separately from real financial cases."""
    scenario = case['scenario']
    class FailureClient(RecordedSecClient):
        def resolve(self, identifier):
            if scenario in ('current_outage', 'wrong_year'):
                raise SecTransportError('SEC unavailable in benchmark failure injection.', 'network')
            if scenario == 'missing_peer' and identifier == 'MSFT':
                raise ResearchError('Company identity is unresolved in benchmark failure injection.')
            return super().resolve(identifier)

        def get(self, path):
            data = super().get(path)
            if scenario == 'missing_revenue' and path == 'api/xbrl/companyfacts/CIK0000320193.json':
                for tag in ('RevenueFromContractWithCustomerExcludingAssessedTax', 'Revenues', 'SalesRevenueNet'):
                    data['facts']['us-gaap'].pop(tag, None)
            return data
    plan = ResearchPlan('competitors', 'GOOGL' if scenario == 'wrong_year' else 'AAPL', ('MSFT',),
                        date.today().isoformat() if scenario == 'current_outage' else '2025-10-31',
                        2025 if scenario == 'wrong_year' else None, 'Compare operating margin')
    answer = competitor_research(plan, FailureClient())
    if scenario in ('current_outage', 'wrong_year'):
        checks = {'no_invented_sources': not answer.sources, 'not_complete': answer.research['status'] != 'complete'}
    elif scenario == 'missing_revenue':
        focal = answer.sources[0]['financials']
        checks = {'missing_not_zero': 'revenue' not in focal['facts'], 'no_margin': focal['operating_margin_pct'] is None,
                  'no_comparison_claim': "AAPL's reported operating margin is" not in answer.text, 'warning': bool(answer.warnings)}
    else:
        checks = {'single_company_only': len(answer.sources) == 1, 'partial': answer.research['status'] == 'partial',
                  'no_comparison_claim': "AAPL's reported operating margin is" not in answer.text}
    return checks, answer


def summary(results):
    counts = dict(Counter(r['status'] for r in results))
    dimensions = {}
    for row in results:
        for dimension, passed in row.get('checks', {}).items():
            key = row['kind'] + '.' + dimension
            item = dimensions.setdefault(key, {'passed': 0, 'total': 0})
            item['passed'] += int(passed)
            item['total'] += 1
    return {'counts': counts, 'dimensions': dimensions}


def rescore(report):
    """Re-score saved observations after a reviewed label correction, no rerun."""
    from copy import deepcopy
    from finread import Answer
    updated = deepcopy(report)
    dataset = load_dataset()
    cases = {case['id']: case for case in dataset['cases']}
    for result in updated['results']:
        # Only document labels support re-scoring currently. All other outcomes
        # and runtime failures are retained exactly, including skipped cases.
        if result['kind'] != 'document' or result['status'] in ('error', 'skipped'):
            continue
        case = cases[result['id']]
        if result['question'] != case['question']:
            raise ValueError('Re-scoring cannot change the question that was asked.')
        answer = Answer(result['answer'], result['sources'], [], result['question'],
                        {'plan': {'route': 'document'}})
        checks = score_document(case, answer)
        # Preserve the observed route check; do not assume a different route.
        checks['route'] = result['checks']['route']
        result['checks'] = checks
        result['status'] = 'passed' if all(checks.values()) else 'failed'
    updated['rescored_from_dataset_version'] = report['dataset_version']
    updated['dataset_version'] = dataset['version']
    updated['rescored_at'] = datetime.now(timezone.utc).isoformat()
    updated['summary'] = summary(updated['results'])
    return updated


def run(mode='offline', llm=None, retriever_factory=None, on_result=lambda result: None, only=None):
    data = load_dataset()
    results = []
    pages = []
    if mode == 'model':
        from sample_document import sample_pages
        pages = sample_pages()
    recorded = RecordedSecClient()
    for case in data['cases']:
        if only and case['kind'] not in only:
            continue
        started = time.monotonic()
        result = {'id': case['id'], 'kind': case['kind'], 'question': case['question']}
        runnable = case['kind'] == 'financial' or mode == 'model' or (mode == 'offline' and (case.get('followup') or case['kind'] == 'abstention'))
        if not runnable:
            result.update(status='skipped', reason='Requires real local model and/or PDF retrieval.')
        else:
            try:
                if case['kind'] == 'financial':
                    client = SecClient(budget=8, deadline=75, use_cache=False) if mode == 'live' else recorded
                    plan = ResearchPlan('competitors', case['company'], tuple(case['peers']), case['cutoff'], None, case['question'])
                    if mode == 'model':
                        answer = research_answer(case['question'], retriever_factory, llm, pages, sample=True, client=client)
                    else:
                        answer = competitor_research(plan, client)
                    checks = score_financial(case, answer, data['companies'])
                    # An HTTP failure that happens to return the bundled snapshot
                    # fails the live benchmark, even if every number is correct.
                    if mode == 'live':
                        checks['live_network'] = (answer.research.get('data_status') == 'live_sec' and
                                                  any(e['status'] == 'network_ok' for e in client.events))
                    result.update(answer=answer.text, research=answer.research, sources=answer.sources)
                elif case['kind'] == 'abstention':
                    checks, answer = evaluate_abstention(case)
                    result.update(answer=answer.text, research=answer.research, failure_injection=True)
                elif case['kind'] == 'document':
                    answer = research_answer(case['question'], retriever_factory, llm, pages, sample=True, client=recorded)
                    checks = score_document(case, answer)
                    result.update(answer=answer.text, sources=answer.sources)
                else:
                    plan = plan_question(case['question'], llm, pages, previous_comparison() if case['followup'] else (), True)
                    checks = score_route(case, plan)
                    result['plan'] = asdict(plan)
                result.update(checks=checks, status='passed' if all(checks.values()) else 'failed')
            except Exception as exc:
                # Preserve per-case failure without exposing provider response bodies.
                result.update(status='error', error_type=type(exc).__name__)
        result['elapsed_seconds'] = round(time.monotonic() - started, 3)
        results.append(result)
        on_result(result)
    return {'schema_version': 1, 'dataset_version': data['version'], 'mode': mode,
            'data_origin': 'live_SEC_no_cache' if mode == 'live' else 'recorded_SEC_fixture_replay',
            'generated_at': datetime.now(timezone.utc).isoformat(), 'summary': summary(results), 'results': results}

import io
import json
import time
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import sec_data
from sec_data import SecClient, SecTransportError, ResearchError, retry_seconds
from intelligence import ResearchPlan, competitor_research, plan_question, explicit_identifier


class Reliability(unittest.TestCase):
    def setUp(self):
        self.cache = patch.dict(sec_data._cache, {}, clear=True)
        self.cache.start(); self.addCleanup(self.cache.stop)
        self.cooldown = patch('sec_data._blocked_until', 0)
        self.cooldown.start(); self.addCleanup(self.cooldown.stop)

    def response(self, value):
        result = Mock()
        result.__enter__ = Mock(return_value=io.BytesIO(json.dumps(value).encode()))
        result.__exit__ = Mock(return_value=False)
        return result

    def test_retry_after_supports_seconds_and_http_dates(self):
        self.assertEqual(retry_seconds('20'), 20)
        self.assertEqual(retry_seconds('-1'), 0)
        self.assertEqual(retry_seconds('bad'), 1)
        delay = retry_seconds(format_datetime(datetime.now(timezone.utc) + timedelta(seconds=60)))
        self.assertTrue(58 <= delay <= 60)

    def test_long_cooldown_stops_without_retry_and_is_shared_across_clients(self):
        client = SecClient(deadline=5)
        client.opener = Mock()
        client.opener.open.side_effect = HTTPError('', 429, '', {'Retry-After':'120'}, None)
        with self.assertRaises(SecTransportError) as exc:
            client.get('tickers')
        self.assertEqual(exc.exception.code, 'cooldown')
        self.assertEqual(client.opener.open.call_count, 1)
        another = SecClient(deadline=5)
        another.opener = Mock()
        with self.assertRaises(SecTransportError):
            another.get('tickers')
        another.opener.open.assert_not_called()

    def test_transient_503_recovers_within_budget(self):
        client = SecClient(budget=2)
        client.opener = Mock()
        client.opener.open.side_effect = [HTTPError('',503,'',{'Retry-After':'0'},None), self.response({'ok':1})]
        self.assertEqual(client.get('tickers'), {'ok':1})
        self.assertEqual(client.remaining, 0)
        self.assertEqual([e['status'] for e in client.events], ['http_error','network_ok'])

    def test_deadline_rechecked_after_rate_spacing(self):
        client = SecClient()
        client.opener = Mock()
        def expire(_):
            client.deadline = 0
        with patch('sec_data.time.sleep', side_effect=expire), self.assertRaises(SecTransportError):
            client.get('tickers')
        client.opener.open.assert_not_called()

    def test_live_check_bypasses_cached_data(self):
        sec_data._cache['https://www.sec.gov/files/company_tickers.json'] = (time.monotonic(), b'{"cached":true}')
        client = SecClient(use_cache=False)
        client.opener = Mock()
        client.opener.open.return_value = self.response({'fresh':True})
        self.assertEqual(client.get('tickers'), {'fresh':True})
        self.assertEqual(client.events[0]['status'], 'network_ok')

    def test_404_is_data_failure_and_cannot_trigger_snapshot(self):
        client = SecClient()
        client.opener = Mock()
        client.opener.open.side_effect = HTTPError('',404,'',{},None)
        answer = competitor_research(ResearchPlan('competitors','AAPL',('MSFT',),'2025-10-31',None,'Compare margins'),client)
        self.assertFalse(answer.sources)
        self.assertEqual(answer.research['data_status'], 'unavailable')

    def test_wording_cannot_turn_semantic_error_into_fallback(self):
        client=Mock()
        client.resolve.side_effect=ResearchError('denied access is text, not a transport category')
        answer=competitor_research(ResearchPlan('competitors','AAPL',('MSFT',),'2025-10-31',None,'Compare margins'),client)
        self.assertFalse(answer.sources)

    def test_expanded_name_is_allowed_only_when_exact_ticker_was_named(self):
        llm=Mock(invoke=Mock(return_value=json.dumps({'route':'competitors','company':'NVDA',
            'company_mention':'NVIDIA Corporation','peers':[{'ticker':'MSFT','mention':'Microsoft'}]})))
        plan=plan_question('Compare NVDA with MSFT on revenue',llm,[],sample=True)
        self.assertFalse(plan.clarification)
        self.assertEqual(plan.identities,{'NVDA':'NVDA','MSFT':'MSFT'})
        with self.assertRaises(ResearchError):
            plan_question('Compare NVDA with MSFTX on revenue',llm,[],sample=True)

    def test_lowercase_word_is_not_explicit_ticker(self):
        llm=Mock(invoke=Mock(return_value=json.dumps({'route':'competitors','company':'AAPL',
            'peers':[{'ticker':'ON','mention':'ON Semiconductor'}]})))
        with self.assertRaises(ResearchError):
            plan_question('Compare Apple on revenue',llm,[],sample=True)

    def test_numeric_year_string_is_normalized_but_invalid_year_is_rejected(self):
        for raw in ('2025', 2025):
            llm=Mock(invoke=Mock(return_value=json.dumps({'route':'document','fiscal_year':raw})))
            self.assertEqual(plan_question('Revenue in 2025',llm,[],sample=True).fiscal_year,2025)
        for raw in ('2025x', True, '9999'):
            llm=Mock(invoke=Mock(return_value=json.dumps({'route':'document','fiscal_year':raw})))
            with self.assertRaises(ResearchError):
                plan_question('Revenue in 2025',llm,[],sample=True)

    def test_explicit_ticker_pair_keeps_requested_subject_order(self):
        llm=Mock(invoke=Mock(return_value=json.dumps({'route':'competitors','company':'AAPL',
            'peers':[{'ticker':'NVDA','mention':'NVIDIA'}]})))
        plan=plan_question('Compare NVDA with AAPL on operating margin',llm,[],sample=True)
        self.assertEqual(plan.company,'NVDA')
        self.assertEqual(plan.peers,('AAPL',))
        plan=plan_question('Compare NVDA with AAPL.',llm,[],sample=True)
        self.assertEqual(plan.company,'NVDA')
        self.assertTrue(explicit_identifier('MSFT','Use MSFT.'))
        self.assertFalse(explicit_identifier('MSFT','Use MSFT.A'))


if __name__=='__main__':
    unittest.main()

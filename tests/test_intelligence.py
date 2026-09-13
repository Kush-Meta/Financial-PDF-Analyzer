import json
import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from intelligence import ResearchPlan, competitor_research, identity_matches, comparison_relationships, peer_followup, plan_question, research_answer
from peer_snapshot import snapshot
from sec_data import ResearchError, SecClient, _cache, operating_margin, select_annual
from ui import research_notes


class Planner(unittest.TestCase):
    def plan(self, **changes):
        data = {"route": "competitors", "company": "AAPL", "peers": [],
                "as_of": None, "fiscal_year": None, "query": "Is Apple's margin strong?"}
        data.update(changes)
        llm = Mock()
        llm.invoke.return_value = json.dumps(data)
        return plan_question("Compare Apple with Microsoft (MSFT)", llm,
                             [Document(page_content="Apple Inc. AAPL")], sample=True)

    def test_sample_cutoff_is_historical_and_peers_not_invented(self):
        plan = self.plan()
        self.assertEqual(plan.cutoff, "2025-10-31")
        self.assertEqual(plan.peers, ())

    def test_named_peers_require_a_user_mention(self):
        self.assertEqual(self.plan(peers=[{"ticker": "MSFT", "mention": "Microsoft"}]).peers, ("MSFT",))
        with self.assertRaises(ResearchError):
            self.plan(peers=[{"ticker": "TSLA", "mention": "Tesla"}])

    def test_rejects_arbitrary_actions_urls_and_future_dates(self):
        for changes in ({"route": "execute_python"}, {"company": "https://localhost"},
                        {"as_of": "2999-01-01"}, {"fiscal_year": True}, {"peers": [1, 2, 3, 4]}):
            with self.subTest(changes=changes), self.assertRaises(ResearchError):
                self.plan(**changes)

    def test_document_identity_requires_evidence(self):
        result = self.plan(company="TSLA", company_mention="Tesla")
        self.assertIn("identity", result.clarification)

    def test_peer_name_mapping_checked_against_authoritative_company(self):
        record = snapshot("MSFT", "2025-10-31")
        self.assertTrue(identity_matches("Microsoft", record))
        self.assertTrue(identity_matches("MSFT", record))
        self.assertFalse(identity_matches("Apple Inc.", record))

    def test_broken_planner_does_not_fall_through_to_document_answer(self):
        with self.assertRaises(ResearchError):
            plan_question("Compare margins", Mock(invoke=Mock(return_value="not json")), [], sample=True)

    def test_followup_receives_previous_plan_and_user_questions(self):
        llm = Mock(invoke=Mock(return_value=json.dumps({"route": "competitors", "company": "AAPL",
                   "peers": [{"ticker": "MSFT", "mention": "MSFT"}], "query": "Compare cash flow"})))
        plan = plan_question("And cash flow?", llm, [],
                            [{"q": "Compare AAPL and MSFT", "a": "table", "research": {"plan": {"cutoff": "2025-10-31"}}}], True)
        self.assertEqual(plan.peers, ("MSFT",))
        self.assertIn("2025-10-31", llm.invoke.call_args.args[0])

    def test_explicit_followups_preserve_company_and_edit_peers_or_cutoff(self):
        from dataclasses import asdict
        from datetime import date
        previous = ResearchPlan("competitors", "AAPL", ("MSFT",), "2025-10-31", None, "Compare margins", identities={"AAPL": "Apple", "MSFT": "Microsoft"})
        history = [{"q": "Compare Apple and Microsoft", "a": "comparison", "research": {"plan": asdict(previous)}}]
        llm = Mock()
        cash = plan_question("And operating cash flow?", llm, [], history, True)
        self.assertEqual(cash.route, "competitors")
        self.assertEqual(cash.peers, ("MSFT",))
        self.assertIn("operating cash flow", cash.query)
        replacement = plan_question("Use Alphabet instead.", llm, [], history, True)
        self.assertEqual(replacement.company, "AAPL")
        self.assertEqual(replacement.peers, ("GOOGL",))
        self.assertIn("operating margin", replacement.query)
        current = plan_question("Use current data.", llm, [], history, True)
        self.assertEqual(current.cutoff, date.today().isoformat())
        llm.invoke.assert_not_called()

    def test_unknown_or_negated_peer_edits_go_to_planner(self):
        history = [{"research": {"plan": {"route": "competitors", "company": "AAPL", "cutoff": "2025-10-31", "peers": ["MSFT"]}}}]
        self.assertIsNone(peer_followup("What about revenue versus Salesforce?", history))
        self.assertIsNone(peer_followup("Use current data without Microsoft", history))


class AnnualFacts(unittest.TestCase):
    def setUp(self):
        self.profile = {"name": "Test Company", "cik": 1234, "tickers": ["TEST"]}
        self.filing = {"accessionNumber": "0000001234-25-000001", "primaryDocument": "test.htm",
                       "filingDate": "2025-02-01", "reportDate": "2024-12-31", "form": "10-K"}
        self.fact = {"accn": self.filing["accessionNumber"], "form": "10-K", "filed": "2025-02-01",
                     "start": "2024-01-01", "end": "2024-12-31", "val": 100}
        self.data = {"facts": {"us-gaap": {
            "Revenues": {"units": {"USD": [self.fact.copy()]}},
            "OperatingIncomeLoss": {"units": {"USD": [dict(self.fact, val=20)]}},
        }}}

    def select(self):
        return select_annual(self.profile, self.filing, self.data, "2025-02-01")

    def test_exact_accession_and_annual_duration_avoid_later_restatements_and_quarters(self):
        self.data["facts"]["us-gaap"]["Revenues"]["units"]["USD"].extend([
            dict(self.fact, accn="0000001234-26-000002", val=999),
            dict(self.fact, start="2024-10-01", val=50),
            dict(self.fact, filed="2026-02-01", val=200),
        ])
        result = self.select()
        self.assertEqual(result.facts["revenue"]["value"], 100)
        self.assertEqual(operating_margin(result), 20)
        self.assertTrue(any("cash flow" in w for w in result.warnings))

    def test_conflicting_tags_are_missing_not_guessed(self):
        self.data["facts"]["us-gaap"]["SalesRevenueNet"] = {"units": {"USD": [dict(self.fact, val=101)]}}
        result = self.select()
        self.assertNotIn("revenue", result.facts)
        self.assertIsNone(operating_margin(result))

    def test_other_currencies_are_not_silently_usd(self):
        self.data["facts"]["us-gaap"]["Revenues"]["units"] = {"EUR": [self.fact]}
        self.assertNotIn("revenue", self.select().facts)

    def test_inconsistent_starts_rejected(self):
        self.data["facts"]["us-gaap"]["OperatingIncomeLoss"]["units"]["USD"][0]["start"] = "2024-01-02"
        with self.assertRaises(ResearchError):
            self.select()

    def test_nonfinite_and_boolean_values_not_financial_facts(self):
        for val in (float("inf"), float("nan"), True):
            self.data["facts"]["us-gaap"]["Revenues"]["units"]["USD"][0]["val"] = val
            self.assertNotIn("revenue", self.select().facts)

    def test_zero_negative_revenue_and_mismatched_units_do_not_compute_margin(self):
        record = self.select()
        for value in (0, -100):
            record.facts["revenue"]["value"] = value
            self.assertIsNone(operating_margin(record))
        record.facts["revenue"]["value"] = 100
        record.facts["revenue"]["unit"] = "EUR"
        self.assertIsNone(operating_margin(record))

    def test_arbitrary_document_paths_rejected(self):
        self.filing["primaryDocument"] = "../../private"
        with self.assertRaises(ResearchError):
            self.select()

    def test_api_selects_only_filings_available_by_cutoff_and_flags_amendment(self):
        columns = {k: [v] for k, v in self.filing.items()}
        for row in (dict(self.filing, accessionNumber="0000001234-26-000001", filingDate="2026-02-01", reportDate="2025-12-31"),
                    dict(self.filing, accessionNumber="0000001234-25-000002", form="10-K/A")):
            for key, value in row.items():
                columns[key].append(value)
        self.profile["filings"] = {"recent": columns, "files": []}
        client = SecClient()
        client.get = Mock(return_value=dict(self.data, cik=1234))
        record = client.annual(self.profile, "2025-02-01")
        self.assertEqual(record.accession, "0000001234-25-000001")
        self.assertEqual(record.facts["revenue"]["value"], 100)
        self.assertTrue(any("amendment" in w for w in record.warnings))

    def test_api_rejects_financial_facts_from_different_entity(self):
        self.profile["filings"] = {"recent": {k: [v] for k, v in self.filing.items()}, "files": []}
        client = SecClient()
        client.get = Mock(return_value=dict(self.data, cik=9999))
        with self.assertRaisesRegex(ResearchError, "did not match"):
            client.annual(self.profile, "2025-02-01")


class RetrievalBounds(unittest.TestCase):
    def test_403_is_not_retried(self):
        from urllib.error import HTTPError
        client = SecClient()
        client.opener = Mock()
        client.opener.open.side_effect = HTTPError("https://www.sec.gov", 403, "Denied", {}, None)
        with patch.dict(_cache, {}, clear=True), self.assertRaisesRegex(ResearchError, "denied access"):
            client.get("tickers")
        self.assertEqual(client.opener.open.call_count, 1)

    def test_transient_failure_has_only_one_retry(self):
        from urllib.error import HTTPError
        client = SecClient()
        client.opener = Mock()
        client.opener.open.side_effect = HTTPError("https://www.sec.gov", 429, "Slow down", {}, None)
        with patch.dict(_cache, {}, clear=True), patch("sec_data.time.sleep"), self.assertRaisesRegex(ResearchError, "429"):
            client.get("tickers")
        self.assertEqual(client.opener.open.call_count, 2)

    def test_only_allowed_sec_endpoints_and_request_budget(self):
        client = SecClient(budget=0)
        for path in ("https://localhost/secrets", "submissions/../../secrets", "api/xbrl/companyfacts/CIK0000000001.json"):
            with self.assertRaises(ResearchError):
                client.get(path)

    def test_cache_returns_independent_data(self):
        import time
        url = "https://www.sec.gov/files/company_tickers.json"
        with patch.dict(_cache, {url: (time.monotonic(), b'{"example": 1}')}, clear=True):
            client = SecClient(budget=0)
            client.get("tickers")["example"] = 2
            self.assertEqual(client.get("tickers")["example"], 1)

    def test_snapshot_never_claims_current_or_wrong_fiscal_year(self):
        self.assertIsNone(snapshot("AAPL", "2026-09-13"))
        self.assertIsNone(snapshot("GOOGL", "2025-10-31", 2025))
        self.assertEqual(snapshot("GOOGL", "2025-10-31").end, "2024-12-31")


class CompetitorFlow(unittest.TestCase):
    def setUp(self):
        self.plan = ResearchPlan("competitors", "AAPL", (), "2025-10-31", None, "Compare margins")
        self.client = Mock()
        self.client.resolve.side_effect = ResearchError("SEC denied access from this network.")
        self.llm = Mock(invoke=Mock(return_value="Apple's operating margin is below Microsoft's in these annual periods [S1] [S2]."))

    def test_real_snapshot_math_provenance_and_period_warning(self):
        answer = competitor_research(self.plan, self.client)
        self.assertIn("31.97%", answer.text)
        self.assertIn("45.62%", answer.text)
        self.assertIn("32.11%", answer.text)
        self.assertEqual(len(answer.sources), 3)
        self.assertTrue(all(s["page"] is None and s["url"].startswith("https://") for s in answer.sources))
        self.assertIn("Dated snapshot", answer.text)
        self.assertTrue(any("Fiscal periods differ" in w for w in answer.warnings))
        self.client.resolve.assert_called_once()

    def test_comparison_conclusion_is_calculated_from_the_source_values(self):
        answer = competitor_research(self.plan, self.client)
        self.assertIn("AAPL's reported operating margin is below MSFT's [S2]", answer.text)
        self.assertIn("31.97%", answer.text)

    def test_relationships_use_source_values_not_model_arithmetic(self):
        records = [snapshot(t, "2025-10-31") for t in ("AAPL", "MSFT")]
        evidence = comparison_relationships(records, [{"id": "S1"}, {"id": "S2"}])
        self.assertEqual(evidence[1]["observations"]["operating_margin"], "higher than focal")
        self.assertEqual(evidence[1]["observations"]["operating_income"], "lower than focal")
        self.assertEqual(evidence[1]["observations"]["operating_cash_flow"], "higher than focal")

    def test_semantic_retrieval_failure_does_not_fall_back_to_different_evidence(self):
        self.client.resolve.side_effect = ResearchError("SEC company identity did not match the request.")
        answer = competitor_research(self.plan, self.client)
        self.assertFalse(answer.sources)

    def test_focal_name_to_ticker_mismatch_prevents_comparison(self):
        plan = ResearchPlan("competitors", "AAPL", (), "2025-10-31", None, "Compare", identities={"AAPL": "Tesla"})
        answer = competitor_research(plan, self.client)
        self.assertFalse(answer.sources)
        self.assertTrue(any("verify" in w for w in answer.warnings))

    def test_peer_tool_does_not_call_an_answer_model(self):
        factory = Mock()
        with patch("intelligence.plan_question", return_value=self.plan):
            answer = research_answer("Compare", factory, self.llm, [], client=self.client)
        self.llm.invoke.assert_not_called()
        self.assertIn("31.97%", answer.text)

    def test_unsupported_automatic_peers_asks_instead_of_inventing(self):
        plan = ResearchPlan("competitors", "XYZ", (), "2025-10-31", None, "Compare")
        answer = competitor_research(plan, self.client)
        self.assertEqual(answer.research["status"], "needs_clarification")
        self.client.resolve.assert_not_called()

    def test_current_data_failure_cannot_silently_use_snapshot(self):
        plan = ResearchPlan("competitors", "AAPL", (), "2026-09-13", None, "Current comparison")
        answer = competitor_research(plan, self.client)
        self.assertFalse(answer.sources)
        self.assertNotIn("31.97%", answer.text)

    def test_peer_route_does_not_embed_document(self):
        factory = Mock()
        with patch("intelligence.plan_question", return_value=self.plan):
            answer = research_answer("Compare", factory, self.llm, [], sample=True, client=self.client)
        factory.assert_not_called()
        self.assertEqual(len(answer.sources), 3)

    def test_credit_question_cannot_invent_sp_rating(self):
        plan = ResearchPlan("credit", "AAPL", (), "2025-10-31", None, "Rate Apple")
        factory = Mock()
        with patch("intelligence.plan_question", return_value=plan):
            answer = research_answer("Rate Apple", factory, self.llm, [])
        self.assertEqual(answer.research["status"], "unsupported")
        factory.assert_not_called()

    def test_exports_preserve_external_sources_calculations_and_limitations(self):
        answer = competitor_research(self.plan, self.client)
        output = research_notes([{"q": "Compare", "a": answer.text, "sources": answer.sources,
                                  "warnings": answer.warnings, "research": answer.research}])
        self.assertIn("https://www.microsoft.com", output)
        self.assertIn("operating_income / revenue * 100", output)
        self.assertIn("Fiscal periods differ", output)
        self.assertNotIn("page None", output)


if __name__ == "__main__":
    unittest.main()

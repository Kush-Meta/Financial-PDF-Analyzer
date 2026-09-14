"""Public filing regressions and integration checks for deterministic figures."""
from decimal import Decimal
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from benchmarks.evaluate import load_dataset, score_document
from finread import Answer, answer_question, evidence_context, statement_answer
from intelligence import ResearchPlan, research_answer
from numeric_evidence import quantitative_answer
from sample_document import sample_pages
from ui import research_notes


class NumericIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pages = sample_pages()
        cls.cases = [case for case in load_dataset()['cases'] if case['kind'] == 'document']

    def test_public_statement_values_match_all_twelve_existing_development_labels(self):
        for case in self.cases:
            with self.subTest(case=case['id']):
                page = self.pages[case['page'] - 1]
                llm = Mock()
                answer = answer_question(case['question'], Mock(invoke=Mock(return_value=[page])), llm)
                record = answer.research['verification']
                self.assertEqual(record['status'], 'verified')
                expected = Decimal(str(case['expected'])) * 1_000_000
                if case.get('outflow_magnitude'):
                    expected = -expected
                self.assertEqual(Decimal(record['facts'][0]['value']), expected)
                self.assertEqual(record['facts'][0]['year'], case['year'])
                self.assertEqual(record['facts'][0]['currency'], 'USD')
                self.assertIn(record['facts'][0]['row'], page.page_content)
                llm.invoke.assert_not_called()

    def test_original_failed_capex_answer_is_preserved_and_replaced_from_same_evidence(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/capex-model-failure.json').read_text())
        case = next(c for c in self.cases if c['id'] == 'document-capex')
        old = Answer(fixture['answer'], fixture['sources'], [], fixture['question'], {'plan': {'route': 'document'}})
        self.assertFalse(score_document(case, old)['numeric_unit'])
        # Establish currency/entity provenance from the bundled filing, not the
        # failed model text. Each saved excerpt must still occur on that page.
        sources = []
        for saved in fixture['sources']:
            page = self.pages[saved['page'] - 1]
            self.assertIn(saved['text'], page.page_content)
            _, fresh = evidence_context([page])
            sources.append({**fresh[0], 'id': saved['id'], 'text': saved['text']})
        result = quantitative_answer(fixture['question'], sources)
        self.assertEqual(result['verification']['status'], 'verified')
        self.assertIn('$12,715 million ($12.715 billion)', result['text'])
        self.assertNotIn('$12.715 million', result['text'])
        fact = result['verification']['facts'][0]
        self.assertEqual(fact['value'], fixture['expected_base_usd'])
        self.assertEqual(fact['raw_value'], '(12,715)')
        self.assertEqual(fact['page'], 36)
        self.assertEqual(result['verification']['calculations'][0]['formula'], 'abs(reported_cash_flow)')

    def test_margin_and_growth_use_source_cells_and_explicit_formulas(self):
        _, sources = evidence_context([self.pages[31]])
        margin = quantitative_answer('What was operating margin in fiscal 2025?', sources)
        self.assertIn('31.97%', margin['text'])
        self.assertIn('$133,050 million ÷ $416,161 million', margin['text'])
        self.assertEqual(margin['verification']['calculations'][0]['formula'], 'operating_income / revenue * 100')
        growth = quantitative_answer('What was revenue growth from fiscal 2024 to fiscal 2025?', sources)
        self.assertIn('6.43%', growth['text'])
        self.assertIn('($416,161 million − $391,035 million) ÷ $391,035 million', growth['text'])
        self.assertEqual(len(growth['verification']['calculations'][0]['inputs']), 2)

    def test_fy_abbreviation_retains_explicit_year(self):
        _, sources = evidence_context([self.pages[31]])
        answer = quantitative_answer('What was revenue in FY2024?', sources)
        self.assertEqual(answer['verification']['facts'][0]['year'], 2024)

    def test_service_revenue_does_not_use_services_cost_row(self):
        _, sources = evidence_context([self.pages[31]])
        answer = quantitative_answer('What was services revenue in fiscal 2025?', sources)
        self.assertEqual(answer['verification']['facts'][0]['value'], '109158000000')
        self.assertNotIn('26,844', answer['text'])

    def test_planner_cannot_remove_unsupported_scope_before_checking(self):
        plan = ResearchPlan('document', 'AAPL', (), '2025-10-31', None, 'What was revenue in fiscal 2025?')
        llm = Mock(invoke=Mock(return_value='The annual table does not establish quarterly revenue [S1].'))
        retriever = Mock(invoke=Mock(return_value=[self.pages[31]]))
        with patch('intelligence.plan_question', return_value=plan):
            answer = research_answer('What was quarterly revenue in fiscal 2025?', lambda: retriever, llm, self.pages)
        self.assertEqual(answer.research['verification']['status'], 'not_checked')
        self.assertIn('quarterly revenue', llm.invoke.call_args.args[0])

    def test_verification_survives_research_and_both_export_formats(self):
        question = 'What was revenue in fiscal 2025?'
        plan = ResearchPlan('document', 'AAPL', (), '2025-10-31', None, question)
        llm = Mock()
        retriever = Mock(invoke=Mock(return_value=[self.pages[31]]))
        with patch('intelligence.plan_question', return_value=plan):
            answer = research_answer(question, lambda: retriever, llm, self.pages)
        self.assertEqual(answer.research['status'], 'complete')
        self.assertEqual(answer.research['verification']['status'], 'verified')
        entry = {'q': question, 'a': answer.text, 'sources': answer.sources,
                 'warnings': answer.warnings, 'research': answer.research}
        self.assertEqual(json.loads(json.dumps(entry))['research']['verification'], answer.research['verification'])
        markdown = research_notes([entry])
        self.assertIn('416161000000', markdown)
        self.assertIn('deterministic_statement_cells', markdown)
        self.assertIn('Total net sales', markdown)
        llm.invoke.assert_not_called()

    def test_supported_statement_question_does_not_build_or_retrieve_an_index(self):
        question = 'What was revenue in fiscal 2025?'
        factory, llm = Mock(), Mock()
        plan = ResearchPlan('document', 'AAPL', (), '2025-10-31', None, question)
        with patch('intelligence.plan_question', return_value=plan):
            answer = research_answer(question, factory, llm, self.pages)
        self.assertEqual(answer.research['verification']['status'], 'verified')
        self.assertEqual(len(answer.sources), 1)
        factory.assert_not_called()
        llm.invoke.assert_not_called()

    def test_statement_scan_does_not_silently_omit_oversized_candidates(self):
        from langchain_core.documents import Document
        oversize = Document(page_content=self.pages[31].page_content + 'x' * 12000,
                            metadata=self.pages[31].metadata)
        self.assertIsNone(statement_answer('What was revenue in fiscal 2025?', [self.pages[31], oversize]))

    def test_missing_year_is_explicitly_insufficient_and_never_guessed_by_model(self):
        question = 'What was revenue in fiscal 2022?'
        plan = ResearchPlan('document', 'AAPL', (), '2025-10-31', None, question)
        llm = Mock()
        retriever = Mock(invoke=Mock(return_value=[self.pages[31]]))
        with patch('intelligence.plan_question', return_value=plan):
            answer = research_answer(question, lambda: retriever, llm, self.pages)
        self.assertEqual(answer.research['status'], 'insufficient_evidence')
        self.assertEqual(answer.research['verification']['status'], 'unavailable')
        self.assertIn('2022', answer.text)
        self.assertTrue(answer.sources)
        llm.invoke.assert_not_called()

    def test_why_question_stays_model_interpretation(self):
        llm = Mock(invoke=Mock(return_value='The annual totals alone do not explain the change [S1].'))
        answer = answer_question('Why did revenue rise in fiscal 2025?',
                                 Mock(invoke=Mock(return_value=[self.pages[31]])), llm)
        self.assertEqual(answer.research['verification']['status'], 'not_checked')
        llm.invoke.assert_called_once()


if __name__ == '__main__':
    unittest.main()

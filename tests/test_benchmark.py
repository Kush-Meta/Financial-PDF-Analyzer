from copy import deepcopy
import unittest
from unittest.mock import patch
from finread import Answer
from sec_data import SecTransportError
from benchmarks.evaluate import load_dataset, run, rescore, score_document, score_financial, RecordedSecClient
from intelligence import ResearchPlan, competitor_research


class Benchmark(unittest.TestCase):
    def test_live_benchmark_cannot_pass_using_the_correct_snapshot_numbers(self):
        class BlockedClient:
            events = []
            def resolve(self, identifier):
                raise SecTransportError('SEC denied access in test.', 'access_denied')
        with patch('benchmarks.evaluate.SecClient',return_value=BlockedClient()):
            report=run(mode='live',only=['financial'])
        self.assertEqual(len(report['results']),20)
        self.assertTrue(all(r['status']=='failed' for r in report['results']))
        self.assertTrue(all(not r['checks']['live_network'] for r in report['results']))

    def test_captured_financial_cases_and_followups_pass_without_network(self):
        report=run()
        self.assertEqual(report['summary']['counts'], {'passed':27,'skipped':17})
        self.assertNotIn('live', report['data_origin'])

    def test_wrong_number_period_source_and_direction_are_detected(self):
        data=load_dataset();case=data['cases'][0]
        plan=ResearchPlan('competitors',case['company'],tuple(case['peers']),case['cutoff'],None,case['question'])
        answer=competitor_research(plan,RecordedSecClient())
        corrupted=deepcopy(answer)
        corrupted.sources[0]['financials']['facts']['revenue']['value']=1
        corrupted.sources[0]['financials']['end']='2024-09-28'
        corrupted.sources[0]['financials']['url']='https://example.com'
        checks=score_financial(case,corrupted,data['companies'])
        for dimension in ('numeric','period','source_facts'):
            self.assertFalse(checks[dimension])
        changed=Answer(answer.text.replace('above','below'),answer.sources,[],answer.search_query,answer.research)
        self.assertFalse(score_financial(case,changed,data['companies'])['comparison'])

    def test_document_scorer_rejects_wrong_unit_and_wrong_cited_row(self):
        case=next(c for c in load_dataset()['cases'] if c['id']=='document-sales')
        sources=[{'id':'S1','page':32,'text':'Total net sales 416,161 391,035 383,285'},
                 {'id':'S2','page':32,'text':'Other expenses 1,000'}]
        research={'plan':{'route':'document'}}
        correct=Answer('2025 revenue was $416.161 billion [S1].',sources,[],'',research)
        self.assertTrue(all(score_document(case,correct).values()))
        wrong=Answer('2025 revenue was $416.161 million [S2].',sources,[],'',research)
        checks=score_document(case,wrong)
        self.assertFalse(checks['numeric_unit'])
        self.assertFalse(checks['cited_evidence_hit'])

    def test_document_labels_match_the_original_pdf_rows(self):
        from sample_document import sample_pages
        pages=sample_pages()
        for case in load_dataset()['cases']:
            if case['kind']=='document':
                for page in case.get('accepted_pages',[case['page']]):
                    text=pages[page-1].page_content
                    self.assertIn(case['anchor'],text)
                    self.assertIn(f"{case['expected']:,}",text)

    def test_rescoring_retains_errors_and_wrong_routes(self):
        case=next(c for c in load_dataset()['cases'] if c['id']=='document-sales')
        row={'id':case['id'],'question':case['question'],'kind':'document','status':'failed',
             'answer':'2025 revenue was $416161 million [S1].',
             'sources':[{'id':'S1','page':32,'text':'Total net sales 416,161'}], 'checks':{'route':False}}
        error={'id':'document-sales-prior','kind':'document','status':'error','error_type':'RuntimeError'}
        report={'dataset_version':1,'results':[row,error]}
        result=rescore(report)
        self.assertEqual(result['results'][0]['status'],'failed')
        self.assertEqual(result['results'][1],error)
        self.assertEqual(result['rescored_from_dataset_version'],1)


if __name__=='__main__':
    unittest.main()

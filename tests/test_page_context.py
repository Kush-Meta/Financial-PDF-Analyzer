import unittest
from unittest.mock import Mock
from langchain_core.documents import Document
from finread import PageContextRetriever


def doc(text, source='filing.pdf', page=1):
    return Document(page_content=text,metadata={'source':source,'page_number':page})


class PageContext(unittest.TestCase):
    def test_restores_missing_rows_and_deduplicates_expanded_pages(self):
        page=doc('Annual statement in USD millions. Revenue 100. Total liabilities 285.')
        retriever=Mock(invoke=Mock(return_value=[doc('Revenue 100.'),doc('Total liabilities 285.')]))
        result=PageContextRetriever(retriever,[page]).invoke('liabilities')
        self.assertEqual(result,[page])
        self.assertIn('USD millions',result[0].page_content)

    def test_never_expands_a_different_document_with_the_same_page_number(self):
        ranked=doc('Original excerpt','private-a.pdf')
        other=doc('Other customer document','private-b.pdf')
        result=PageContextRetriever(Mock(invoke=Mock(return_value=[ranked])),[other]).invoke('revenue')
        self.assertEqual(result,[ranked])

    def test_oversized_pages_keep_ranked_chunks_without_truncation(self):
        chunk=doc('Relevant ending row')
        page=doc('x'*7000)
        result=PageContextRetriever(Mock(invoke=Mock(return_value=[chunk])),[page]).invoke('question')
        self.assertEqual(result,[chunk])

    def test_total_context_budget_preserves_ranked_evidence(self):
        pages=[doc('x'*100,page=i) for i in (1,2,3)]
        chunks=[doc('row',page=i) for i in (1,2,3)]
        result=PageContextRetriever(Mock(invoke=Mock(return_value=chunks)),pages,max_context_chars=105).invoke('question')
        self.assertEqual([len(p.page_content) for p in result],[100,3])
        self.assertLessEqual(sum(len(p.page_content) for p in result),105)


if __name__=='__main__':
    unittest.main()

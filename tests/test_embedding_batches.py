import unittest
from unittest.mock import Mock, patch
from langchain_core.documents import Document
from services import build_index


class EmbeddingBatches(unittest.TestCase):
    def setUp(self):
        self.pages=[Document(page_content='Revenue of 100 million.', metadata={'source':'public.pdf','page_number':i}) for i in range(40)]

    def test_filing_is_embedded_in_bounded_batches_with_original_metadata(self):
        embedding=Mock()
        embedding.embed_documents.side_effect=lambda batch:[[0.1,0.2] for _ in batch]
        with patch('services.OllamaEmbeddings',return_value=embedding), patch('services.FAISS.from_embeddings') as factory:
            build_index(self.pages,'example','http://localhost')
            self.assertEqual([len(c.args[0]) for c in embedding.embed_documents.call_args_list],[16,16,8])
            self.assertEqual(len(list(factory.call_args.args[0])),40)
            self.assertEqual(factory.call_args.kwargs['metadatas'],[p.metadata for p in self.pages])

    def test_failed_or_incomplete_later_batch_never_publishes_index(self):
        for second in (RuntimeError('model failed'), [[0.1,0.2]]):
            embedding=Mock()
            embedding.embed_documents.side_effect=[[[0.1,0.2]]*16,second]
            with patch('services.OllamaEmbeddings',return_value=embedding), patch('services.FAISS.from_embeddings') as factory:
                with self.assertRaises((RuntimeError,ValueError)):
                    build_index(self.pages,'example','http://localhost')
                factory.assert_not_called()


if __name__=='__main__':
    unittest.main()

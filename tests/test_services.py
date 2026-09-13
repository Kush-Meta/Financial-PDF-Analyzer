from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from services import extract_entities, make_llm, make_retriever


class Integrations(unittest.TestCase):
    def test_extraction_sends_full_selected_pages_and_selected_model(self):
        text = "Introduction. " * 300 + "Revenue was $42 million."
        start = text.index("$42")
        page = Document(page_content=text, metadata={"source": "annual.pdf", "page_number": 7})
        entity = SimpleNamespace(
            extraction_class="revenue", extraction_text="$42 million",
            char_interval=SimpleNamespace(start_pos=start, end_pos=start + 11), attributes={})
        with patch("langextract.extract", return_value=SimpleNamespace(extractions=[entity])) as extract, \
             patch("langextract.providers.ollama.OllamaLanguageModel") as provider:
            rows = extract_entities([page], "mistral", "http://configured:11434")
        self.assertEqual(extract.call_args.kwargs["text_or_documents"], text)
        self.assertFalse(extract.call_args.kwargs["fetch_urls"])
        self.assertEqual(provider.call_args.kwargs["model_id"], "mistral")
        self.assertEqual(provider.call_args.kwargs["model_url"], "http://configured:11434")
        self.assertEqual(rows[0]["page"], 7)
        self.assertTrue(rows[0]["exact_match"])

    def test_missing_or_incorrect_offsets_are_not_marked_exact(self):
        page = Document(page_content="Revenue was $42 million.", metadata={"source": "a.pdf", "page_number": 1})
        for interval in (None, SimpleNamespace(start_pos=0, end_pos=11),
                         SimpleNamespace(start_pos=-11, end_pos=0)):
            entity = SimpleNamespace(extraction_class="revenue", extraction_text="$42 million",
                                     char_interval=interval, attributes={})
            with patch("langextract.extract", return_value=SimpleNamespace(extractions=[entity])), \
                 patch("langextract.providers.ollama.OllamaLanguageModel"):
                self.assertFalse(extract_entities([page], "llama3", "local")[0]["exact_match"])

    def test_blank_pages_are_skipped(self):
        page = Document(page_content="", metadata={"source": "a.pdf", "page_number": 2})
        with patch("langextract.extract") as extract, \
             patch("langextract.providers.ollama.OllamaLanguageModel"):
            self.assertEqual(extract_entities([page], "llama3", "local"), [])
        extract.assert_not_called()

    def test_vector_only_retriever_keeps_requested_limit(self):
        index = Mock()
        retriever = make_retriever(index, 4)
        self.assertIs(retriever, index.as_retriever.return_value)
        index.as_retriever.assert_called_once_with(search_kwargs={"k": 4})

    def test_llm_has_timeout_and_deterministic_temperature(self):
        llm = make_llm("llama3", "http://localhost:11434")
        self.assertEqual(llm.temperature, 0)
        self.assertEqual(llm.client_kwargs["timeout"], 180)


if __name__ == "__main__":
    unittest.main()

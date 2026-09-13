import json
import unittest
from unittest.mock import Mock, patch

from finread import (DocumentError, answer_question, citation_issues, document_key,
                     ensure_index, evidence_context, read_pdf, reset_document)
from helpers import pdf_bytes


class Documents(unittest.TestCase):
    def test_text_pages_have_one_based_provenance(self):
        pages = read_pdf(pdf_bytes(("First page", "Second page")), "annual.pdf")
        self.assertEqual([p.metadata["page_number"] for p in pages], [1, 2])
        self.assertEqual(pages[1].page_content, "Second page")
        self.assertEqual(pages[0].metadata["source"], "annual.pdf")

    def test_rejects_invalid_empty_and_corrupt_files(self):
        for data in (b"", b"hello", b"%PDF-1.7 broken"):
            with self.subTest(data=data), self.assertRaises(DocumentError):
                read_pdf(data, "bad.pdf")

    def test_rejects_scans_with_actionable_message(self):
        with self.assertRaisesRegex(DocumentError, "OCR"):
            read_pdf(pdf_bytes(("",)), "scan.pdf")

    def test_keeps_blank_pages_for_correct_page_numbers(self):
        pages = read_pdf(pdf_bytes(("", "Revenue")), "mixed.pdf")
        self.assertEqual(pages[1].metadata["page_number"], 2)

    def test_rejects_encrypted_documents(self):
        with self.assertRaisesRegex(DocumentError, "encrypted"):
            read_pdf(pdf_bytes(encrypted=True), "encrypted.pdf")

    def test_enforces_size_and_page_limits(self):
        with patch("finread.MAX_UPLOAD_BYTES", 4), self.assertRaises(DocumentError):
            read_pdf(pdf_bytes(), "big.pdf")
        with patch("finread.MAX_PAGES", 1), self.assertRaisesRegex(DocumentError, "at most"):
            read_pdf(pdf_bytes(("a", "b")), "long.pdf")

    def test_content_identity_changes_for_same_filename(self):
        self.assertNotEqual(document_key(b"a", "same.pdf"), document_key(b"b", "same.pdf"))


class Sessions(unittest.TestCase):
    def setUp(self):
        self.state = {}
        reset_document(self.state, "one")
        self.state["pages"] = read_pdf(pdf_bytes(), "one.pdf")
        self.builder = Mock(return_value=object())

    def test_index_reused_across_reruns(self):
        a = ensure_index(self.state, "embed", "local", self.builder)
        b = ensure_index(self.state, "embed", "local", self.builder)
        self.assertIs(a, b)
        self.builder.assert_called_once()

    def test_embedding_or_endpoint_change_rebuilds(self):
        for model, url in (("a", "local"), ("b", "local"), ("b", "other")):
            ensure_index(self.state, model, url, self.builder)
        self.assertEqual(self.builder.call_count, 3)

    def test_failure_cannot_publish_or_reuse_incompatible_index(self):
        original = ensure_index(self.state, "a", "local", self.builder)
        failing = Mock(side_effect=RuntimeError("offline"))
        with self.assertRaises(RuntimeError):
            ensure_index(self.state, "b", "local", failing)
        self.assertIs(self.state["index"], original)
        self.assertEqual(self.state["index_key"][1], "a")
        replacement = Mock(return_value=object())
        ensure_index(self.state, "b", "local", replacement)
        replacement.assert_called_once()

    def test_switch_and_remove_clear_every_document_artifact(self):
        for next_key in ("two", None):
            self.state.update(chat_history=[{"q": "private"}], entities=[1], index=object())
            reset_document(self.state, next_key)
            self.assertEqual(self.state["chat_history"], [])
            for key in ("pages", "entities", "index", "index_key"):
                self.assertIsNone(self.state[key])

    def test_same_document_preserves_history(self):
        self.state["chat_history"].append({"q": "Revenue?"})
        reset_document(self.state, "one")
        self.assertEqual(len(self.state["chat_history"]), 1)

    def test_separate_sessions_do_not_share_indexes(self):
        other = {}
        reset_document(other, "one")
        other["pages"] = self.state["pages"]
        ensure_index(self.state, "a", "local", Mock(return_value="first"))
        self.assertEqual(ensure_index(other, "a", "local", Mock(return_value="second")), "second")
        self.assertEqual(self.state["index"], "first")


class Evidence(unittest.TestCase):
    def setUp(self):
        self.pages = read_pdf(pdf_bytes(), "annual.pdf")
        self.retriever = Mock()
        self.retriever.invoke.return_value = self.pages

    def test_context_and_export_have_identical_full_evidence(self):
        context, sources = evidence_context(self.pages)
        self.assertEqual(json.loads(context), sources)
        self.assertEqual(sources[0]["page"], 1)
        self.assertEqual(sources[0]["text"], self.pages[0].page_content)

    def test_prompt_contains_page_and_citation_contract(self):
        llm = Mock()
        llm.invoke.return_value = "Revenue was $42 million [S1]."
        answer = answer_question("Revenue?", self.retriever, llm)
        self.assertFalse(answer.warnings)
        self.assertIn('"page": 1', llm.invoke.call_args.args[0])
        self.assertIn('"id": "S1"', llm.invoke.call_args.args[0])

    def test_invalid_and_missing_citations_are_flagged(self):
        _, sources = evidence_context(self.pages)
        self.assertTrue(citation_issues("Revenue increased.", sources))
        self.assertIn("[S9]", citation_issues("Revenue increased [S9].", sources)[0])
        self.assertEqual(citation_issues("Revenue [S1].", sources), [])

    def test_followup_query_is_resolved_before_retrieval(self):
        llm = Mock()
        llm.invoke.side_effect = ["What was revenue in 2023?", "Not reported in this excerpt [S1]."]
        answer = answer_question("And in 2023?", self.retriever, llm,
                                 [{"q": "Revenue in 2024?", "a": "$42m [S1]."}])
        self.retriever.invoke.assert_called_once_with("What was revenue in 2023?")
        self.assertEqual(answer.search_query, "What was revenue in 2023?")

    def test_empty_retrieval_does_not_generate_an_answer(self):
        self.retriever.invoke.return_value = []
        llm = Mock()
        answer = answer_question("Revenue?", self.retriever, llm)
        llm.invoke.assert_not_called()
        self.assertEqual(answer.sources, [])

    def test_empty_model_output_is_a_recoverable_failure(self):
        llm = Mock()
        llm.invoke.return_value = " "
        with self.assertRaisesRegex(ValueError, "empty answer"):
            answer_question("Revenue?", self.retriever, llm)


if __name__ == "__main__":
    unittest.main()

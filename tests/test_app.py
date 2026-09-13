from contextlib import ExitStack
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

from finread import Answer
from intelligence import ResearchPlan
from helpers import Upload, pdf_bytes

APP = str(Path(__file__).resolve().parents[1] / "app.py")


class AppBehavior(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.upload = self.stack.enter_context(patch("streamlit.file_uploader", return_value=Upload(pdf_bytes())))
        self.builder = self.stack.enter_context(patch("services.build_index", return_value=Mock()))
        self.stack.enter_context(patch("services.load_reranker", return_value=Mock()))
        self.stack.enter_context(patch("services.make_retriever", return_value=Mock()))
        self.stack.enter_context(patch("services.make_llm", return_value=Mock()))
        self.planner = self.stack.enter_context(patch("intelligence.plan_question", return_value=ResearchPlan(
            "document", "", (), "2025-10-31", None, "Revenue?")))
        self.answer = self.stack.enter_context(patch("finread.answer_question", return_value=Answer(
            "Revenue was $42 million [S1].",
            [{"id": "S1", "file": "filing.pdf", "page": 1, "page_label": "1", "text": "Revenue was $42 million."}],
            [], "Revenue?")))
        self.at = AppTest.from_file(APP, default_timeout=15).run()

    def assert_clean(self):
        self.assertEqual(list(self.at.exception), [])

    def test_upload_does_not_call_models_or_build_index(self):
        self.assert_clean()
        self.builder.assert_not_called()
        self.answer.assert_not_called()

    def test_chat_is_once_per_submit_and_index_survives_settings(self):
        self.at.chat_input[0].set_value("Revenue?").run()
        self.assert_clean()
        self.assertEqual(len(self.at.session_state.chat_history), 1)
        self.at.run()
        self.at.sidebar.selectbox[1].select("mistral").run()
        self.assertEqual(self.answer.call_count, 1)
        self.at.chat_input[0].set_value("Net income?").run()
        self.assert_clean()
        self.assertEqual(self.answer.call_count, 2)
        self.assertEqual(self.builder.call_count, 1)

    def test_document_switch_clears_previous_answers_and_index(self):
        self.at.chat_input[0].set_value("Revenue?").run()
        self.upload.return_value = Upload(pdf_bytes(("Another company's filing",)))
        self.at.run()
        self.assert_clean()
        self.assertEqual(self.at.session_state.chat_history, [])
        self.assertIsNone(self.at.session_state.index)

    def test_document_removal_clears_history(self):
        self.at.chat_input[0].set_value("Revenue?").run()
        self.upload.return_value = None
        self.at.run()
        self.assert_clean()
        self.assertEqual(self.at.session_state.chat_history, [])

    def test_model_failure_preserves_previous_research_without_raw_error(self):
        self.at.chat_input[0].set_value("Revenue?").run()
        self.answer.side_effect = RuntimeError("sensitive model payload")
        self.at.chat_input[0].set_value("Cash flow?").run()
        self.assert_clean()
        self.assertEqual(len(self.at.session_state.chat_history), 1)
        self.assertTrue(self.at.error)
        self.assertNotIn("sensitive", self.at.error[0].value)

    def test_bad_pdf_shows_actionable_error(self):
        self.upload.return_value = Upload(b"invalid")
        self.at.run()
        self.assert_clean()
        self.assertIn("valid PDF header", self.at.error[0].value)

    def test_minimum_retrieval_setting_is_valid(self):
        self.at.sidebar.slider[0].set_value(1).run()
        self.assert_clean()

    def test_entities_survive_rerun_and_show_original_selection(self):
        with patch("services.extract_entities", return_value=[{"entity": "$42 million", "page": 1}]) as extract:
            self.at.session_state.inspector_view = "Figures"
            self.at.run()
            self.at.button(key="extract_entities").click().run()
            self.assert_clean()
            self.at.run()
            self.assert_clean()
            self.assertEqual(extract.call_count, 1)
            self.assertEqual(self.at.session_state.entities["pages"], [1])
            self.assertEqual(len(self.at.dataframe), 1)

    def test_direct_questions_and_followups_without_suggestion_buttons(self):
        self.assertFalse(any((button.key or "").startswith(("start_", "followup_")) for button in self.at.button))
        self.at.chat_input[0].set_value("What changed in cash flow?").run()
        self.assert_clean()
        self.at.run()
        self.assertEqual(self.answer.call_count, 1)
        self.at.chat_input[0].set_value("Explain that in simpler terms.").run()
        self.assert_clean()
        self.assertEqual(len(self.at.session_state.chat_history), 2)
        self.assertEqual(self.builder.call_count, 1)
        self.assertFalse(any((button.key or "").startswith(("start_", "followup_")) for button in self.at.button))

    def test_citation_opens_full_page_without_regenerating(self):
        self.at.chat_input[0].set_value("Revenue?").run()
        self.at.button(key="source_0_S1").click().run()
        self.assert_clean()
        self.assertEqual(self.at.session_state.selected_evidence, (0, "S1"))
        self.at.button(key="read_full_page").click().run()
        self.assert_clean()
        self.assertEqual(self.at.session_state.inspector_view, "Pages")
        self.assertEqual(self.at.number_input(key="reader_page").value, 1)
        self.assertEqual(self.answer.call_count, 1)

    def test_new_conversation_keeps_document_index(self):
        self.at.chat_input[0].set_value("Revenue?").run()
        index = self.at.session_state.index
        self.at.button(key="new_conversation").click().run()
        self.assert_clean()
        self.assertEqual(self.at.session_state.chat_history, [])
        self.assertIs(self.at.session_state.index, index)
        self.assertIsNone(self.at.session_state.selected_evidence)
        self.assertIsNone(self.at.session_state.failed_question)

    def test_failed_question_can_retry_without_retyping(self):
        response = self.answer.return_value
        self.answer.side_effect = RuntimeError("offline")
        self.at.chat_input[0].set_value("Revenue?").run()
        self.assertEqual(self.at.session_state.failed_question, "Revenue?")
        self.answer.side_effect = None
        self.answer.return_value = response
        self.at.button(key="retry_now").click().run()
        self.assert_clean()
        self.assertIsNone(self.at.session_state.failed_question)
        self.assertEqual(len(self.at.session_state.chat_history), 1)
        self.assertEqual(self.builder.call_count, 1)

    def test_document_switch_resets_reader_and_pending_actions(self):
        self.at.session_state.reader_page = 20
        self.at.session_state.pending_question = "Question about the old file"
        self.at.session_state.failed_question = "Old failure"
        self.upload.return_value = Upload(pdf_bytes(("New filing",)), "new.pdf")
        self.at.run()
        self.assert_clean()
        self.assertEqual(self.at.session_state.reader_page, 1)
        self.assertIsNone(self.at.session_state.pending_question)
        self.assertIsNone(self.at.session_state.failed_question)
        self.answer.assert_not_called()

    def test_empty_state_sample_and_close_need_no_model_calls(self):
        self.upload.return_value = None
        self.at.run()
        self.assert_clean()
        self.assertEqual(len(self.at.chat_input), 0)
        self.at.button(key="use_sample").click().run()
        self.assert_clean()
        self.assertEqual(len(self.at.session_state.pages), 80)
        self.assertIn("Apple Inc.", self.at.session_state.pages[0].page_content)
        self.assertIn("2025", self.at.session_state.pages[0].page_content)
        self.assertEqual(self.at.session_state.pages[31].metadata["page_label"], "29")
        self.assertFalse(self.at.chat_input[0].disabled)
        self.at.button(key="close_sample").click().run()
        self.assert_clean()
        self.assertIsNone(self.at.session_state.pages)
        self.assertEqual(len(self.at.chat_input), 0)
        self.builder.assert_not_called()
        self.answer.assert_not_called()

    def test_external_comparison_has_company_sources_and_never_opens_uploaded_page(self):
        from sec_data import ResearchError
        self.planner.return_value = ResearchPlan("competitors", "AAPL", ("MSFT",), "2025-10-31", None, "Compare margins")
        client = Mock()
        client.resolve.side_effect = ResearchError("SEC denied access from this network.")
        with patch("intelligence.SecClient", return_value=client):
            self.at.chat_input[0].set_value("Compare Apple and Microsoft margins").run()
        self.assert_clean()
        self.assertEqual(len(self.at.session_state.chat_history[0]["sources"]), 2)
        self.at.button(key="source_0_S2").click().run()
        self.assert_clean()
        self.assertFalse(any(b.key == "read_full_page" for b in self.at.button))
        self.assertEqual(self.at.get("link_button")[0].proto.url, "https://www.microsoft.com/investor/reports/ar25/index.html")
        self.builder.assert_not_called()
        self.answer.assert_not_called()


if __name__ == "__main__":
    unittest.main()

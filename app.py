"""FinRead: a conversation-first financial research workspace."""
from html import escape
import json
import logging
import os
import time

import streamlit as st

from finread import (DocumentError, document_key, ensure_index,
                     read_pdf, reset_document)
from intelligence import research_answer
from sec_data import ResearchError
from sample_document import SAMPLE_NAME, SAMPLE_SEC_URL, SAMPLE_PDF_URL, sample_bytes, sample_pages
from services import build_index, extract_entities, load_reranker, make_llm, make_retriever
from ui import apply_styles, open_page, queue_question, render_answer, render_source, research_notes

logger = logging.getLogger(__name__)
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
st.set_page_config(page_title="FinRead", page_icon="📑", layout="wide")
apply_styles()
st.session_state.setdefault("demo_active", False)
st.session_state.setdefault("inspector_view", "Sources")
st.session_state.setdefault("pending_question", None)
st.session_state.setdefault("failed_question", None)


@st.cache_resource(show_spinner=False)
def cached_reranker(name):
    return load_reranker(name)


def show_failure(action, exc):
    logger.warning("%s failed (%s)", action, type(exc).__name__)
    st.error(f"{action} failed. Check that Ollama is running and the selected model is installed, then retry.")


def new_conversation():
    if not st.session_state.get("new_conversation"):
        return
    st.session_state.chat_history = []
    st.session_state.selected_evidence = None
    st.session_state.pending_question = None
    st.session_state.failed_question = None


def use_sample():
    st.session_state.demo_active = True


with st.sidebar:
    st.html('<div class="brand"><div class="brand-name">finread</div></div>')
    upload_container = st.expander("Change document") if st.session_state.get("pages") else st.container()
    with upload_container:
        pdf = st.file_uploader("Upload a PDF", type="pdf",
                               help="Text-based PDFs · up to 25 MB · up to 1,000 pages", key="pdf_upload")
        if st.session_state.demo_active:
            if st.button("Close example filing", key="close_sample", type="tertiary"):
                st.session_state.demo_active = False
                st.rerun()

# Uploaded data always takes precedence over the bundled SEC filing.
if pdf is not None:
    st.session_state.demo_active = False
    payload = pdf.getvalue()
    active_key = document_key(payload, pdf.name)
    active_name = pdf.name
elif st.session_state.demo_active:
    payload = sample_bytes()
    active_key = document_key(payload, SAMPLE_NAME)
    active_name = SAMPLE_NAME
else:
    active_key = None
    active_name = ""

reset_document(st.session_state, active_key)
pages = None
document_error = None
if active_key:
    if st.session_state.pages is None:
        try:
            with st.spinner("Getting your document ready…"):
                st.session_state.pages = sample_pages() if st.session_state.demo_active else read_pdf(payload, active_name)
        except DocumentError as exc:
            document_error = str(exc)
    pages = st.session_state.pages

with st.sidebar:
    if pages:
        st.html(f'<div class="source-card"><strong>▤ &nbsp;{escape(active_name)}</strong>'
                f'<small>{len(pages)} pages · {"SEC filing" if st.session_state.demo_active else "Ready to explore"}</small></div>')
        if st.session_state.demo_active:
            st.markdown(f"[SEC record]({SAMPLE_SEC_URL}) · [Original PDF]({SAMPLE_PDF_URL})")
    with st.expander("Settings", icon=":material/tune:"):
        embed_model = st.selectbox("Embedding model", ["mxbai-embed-large", "nomic-embed-text"], key="embed_model")
        llm_model = st.selectbox("Analysis model", ["llama3", "mistral", "gemma2"], key="llm_model")
        enable_reranking = st.toggle("Rerank evidence", value=True, key="enable_reranking",
                                     help="Sort retrieved excerpts by relevance before answering.")
        rerank_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        if enable_reranking:
            rerank_model_name = st.selectbox("Reranker", [
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "cross-encoder/ms-marco-TinyBERT-L-2-v2", "BAAI/bge-reranker-base"], key="rerank_model")
        top_k = st.slider("Search candidates", 1, 30, 15, key="top_k")
        top_n = st.slider("Evidence excerpts", 1, min(top_k, 6), min(top_k, 4), key="top_n") if top_k > 1 else 1
        st.caption("Research lives in this browser session's server memory. Export anything you want to keep before refreshing.")
        st.caption("Text is sent to your configured Ollama server. Reranker weights may download on first use.")
        st.caption("Peer questions look up public company identifiers and financials at the SEC. Uploaded PDF text stays with Ollama. The Apple example can use a dated annual-report snapshot if SEC access fails.")

header, actions = st.columns([4, 1], vertical_alignment="center")
with header:
    title = active_name if pages else "Your research workspace"
    st.html(f'<div class="workspace-header"><div>'
            f'<div class="workspace-title">{escape(title)}</div></div></div>')
with actions:
    if pages and st.session_state.chat_history:
        with st.popover("More", icon=":material/more_horiz:", width="stretch"):
            st.button("New conversation", key="new_conversation", on_click=new_conversation, width="stretch")
            st.download_button("Research JSON", json.dumps({
                "document": active_name, "sha256": active_key.split(":", 1)[0],
                "answers": st.session_state.chat_history,
            }, indent=2), file_name="finread-research.json", mime="application/json", width="stretch")
            transcript = research_notes(st.session_state.chat_history)
            st.download_button("Readable notes", transcript, file_name="finread-research.md",
                               mime="text/markdown", width="stretch")

if document_error:
    st.error(document_error)
    st.info("Choose another PDF in the source panel to give it another go.")
    st.stop()

if not pages:
    with st.container(key="welcome"):
        st.html('<div class="welcome-copy"><h1>Start with a filing.</h1>'
                '<p>Upload your PDF, or open Apple’s annual report.</p></div>')
        left, center, right = st.columns([1, 2, 1])
        with center:
            st.button("Open Apple’s 2025 10-K", type="primary",
                      width="stretch", key="use_sample", on_click=use_sample)
        st.html('<div class="footnote">80 pages · Original SEC filing</div>')
    st.stop()

blank_pages = sum(not p.page_content.strip() for p in pages)
if blank_pages:
    st.warning(f"{blank_pages} page(s) contain no readable text. Scanned content needs OCR before it can be searched.")

has_answers = bool(st.session_state.chat_history)
with st.container(key="research_layout"):
    chat_column, evidence_column = st.columns([1.75, 1], gap="medium")
with chat_column:
    with st.container(key="workspace_chat"):
        conversation = st.container(height=520, border=False, key="conversation") if has_answers else st.container()
        with conversation:
            if not has_answers:
                st.html('<div class="chat-empty"><h1>What would you like to know?</h1>'
                        '<p>Ask about this filing in your own words.</p></div>')
            else:
                for i, entry in enumerate(st.session_state.chat_history):
                    render_answer(entry, i)
            if st.session_state.failed_question:
                st.warning("That question hit a snag. Your research is still here.")
                st.button("Try that question again", icon=":material/refresh:", key="retry_question",
                          on_click=queue_question, args=(st.session_state.failed_question, "retry_question"))
        typed = st.chat_input("Ask about this filing…", max_chars=4000, key="composer")
        st.html('<div class="footnote">Answers include source references.</div>')

    question = typed or st.session_state.get("pending_question")
    st.session_state.pending_question = None
    if question and question.strip():
        question = question.strip()
        st.session_state.pending_question = None
        try:
            started = time.monotonic()
            with st.status("Reading the filing…", expanded=True) as status:
                def document_retriever():
                    index = ensure_index(st.session_state, embed_model, OLLAMA_BASE, build_index)
                    encoder = cached_reranker(rerank_model_name) if enable_reranking else None
                    return make_retriever(index, top_k if encoder else top_n, encoder, top_n)
                result = research_answer(question, document_retriever, make_llm(llm_model, OLLAMA_BASE),
                                         pages, st.session_state.chat_history,
                                         sample=st.session_state.demo_active, progress=st.write)
                label = "One detail needed" if result.research.get("status") == "needs_clarification" else "Research complete"
                status.update(label=label, state="complete", expanded=False)
            st.session_state.chat_history.append({
                "q": question, "a": result.text, "sources": result.sources,
                "warnings": result.warnings, "search_query": result.search_query,
                "research": result.research,
                "model": llm_model, "seconds": time.monotonic() - started,
            })
            st.session_state.failed_question = None
            st.session_state.selected_evidence = (len(st.session_state.chat_history) - 1, result.sources[0]["id"]) if result.sources else None
            st.session_state.inspector_view = "Sources"
            st.rerun()
        except Exception as exc:
            st.session_state.failed_question = question
            if isinstance(exc, ResearchError):
                st.error(str(exc))
            else:
                show_failure("Research", exc)
            st.button("Retry question", icon=":material/refresh:", key="retry_now",
                      on_click=queue_question, args=(question, "retry_now"))

with evidence_column:
    with st.expander("Evidence", expanded=has_answers):
        view = st.radio("Evidence view", ["Sources", "Pages", "Figures"], horizontal=True,
                        key="inspector_view", label_visibility="collapsed")
        if view == "Sources" or view is None:
            selected = st.session_state.get("selected_evidence")
            if selected and selected[0] < len(st.session_state.chat_history):
                entry = st.session_state.chat_history[selected[0]]
                source = next((s for s in entry["sources"] if s["id"] == selected[1]), None)
                if source:
                    with st.container(height=410, border=False, key="source_scroll"):
                        render_source(source)
                    if source.get("url"):
                        st.link_button("Open original report", source["url"], icon=":material/open_in_new:")
                    elif source.get("page"):
                        st.button(f"Read full page {source['page']}", icon=":material/open_in_new:",
                                  key="read_full_page", on_click=open_page, args=(source["page"],))
                    st.caption("Source IDs identify excerpts, not a guarantee that every claim is correct.")
            else:
                st.caption("Supporting passages appear here after you ask a question.")
        elif view == "Pages":
            page_number = st.number_input("PDF page", min_value=1, max_value=len(pages), key="reader_page")
            page = pages[page_number - 1]
            label = page.metadata.get("page_label")
            st.caption(f"PDF page {page_number} of {len(pages)}" + (f" · report page {label}" if label else ""))
            with st.container(height=410, border=False):
                st.text(page.page_content or "No readable text on this page. OCR may be needed.")
            st.caption("Extracted text can flatten tables. Check the original PDF for layout.")
        elif view == "Figures":
            st.caption("Extract figures, companies, and periods from up to five pages.")
            selected_pages = st.multiselect("Pages to extract", range(1, len(pages) + 1),
                                            default=[1], max_selections=5, key="entity_pages",
                                            format_func=lambda n: f"Page {n}")
            if st.button("Find financial entities", icon=":material/auto_awesome:", type="primary",
                         disabled=not selected_pages, key="extract_entities", width="stretch"):
                try:
                    with st.spinner("Extracting entities…"):
                        rows = extract_entities([pages[n - 1] for n in selected_pages], llm_model, OLLAMA_BASE)
                    st.session_state.entities = {"pages": selected_pages, "model": llm_model, "rows": rows}
                except Exception as exc:
                    show_failure("Entity extraction", exc)
            if st.session_state.entities is not None:
                extraction = st.session_state.entities
                st.caption(f"Results from pages {extraction['pages']} · {extraction['model']}")
                if extraction["rows"]:
                    st.dataframe(extraction["rows"], width="stretch", hide_index=True)
                    st.download_button("Download entities", json.dumps(extraction, indent=2),
                                       file_name="finread-entities.json", mime="application/json", width="stretch")
                    st.caption("exact_match checks the text against source offsets, not its financial interpretation.")
                else:
                    st.info("No entities on those pages. Try the financial statements.")

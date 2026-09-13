"""Presentation helpers for FinRead's conversation workspace."""
from html import escape
from pathlib import Path

import streamlit as st

def apply_styles():
    st.html(Path(__file__).parent / "assets" / "finread.css")


def queue_question(question, trigger_key):
    # A disappearing widget can emit a false state during a rerun. Only a real
    # press should submit research.
    if st.session_state.get(trigger_key):
        st.session_state.pending_question = question


def select_evidence(answer_index, source_id):
    st.session_state.selected_evidence = (answer_index, source_id)
    st.session_state.inspector_view = "Sources"


def open_page(page_number):
    st.session_state.reader_page = page_number
    st.session_state.inspector_view = "Pages"


def render_answer(entry, index):
    with st.chat_message("user", avatar=":material/person:"):
        st.write(entry["q"])
    with st.chat_message("assistant", avatar=":material/auto_awesome:"):
        st.markdown(entry["a"].replace("![", r"\!["))
        for warning in entry["warnings"]:
            st.warning(warning)
        if entry["sources"]:
            with st.container(horizontal=True, key=f"citations_{index}"):
                for source in entry["sources"]:
                    st.button(f"{source['id']} · p. {source['page']}",
                              key=f"source_{index}_{source['id']}",
                              help=f"Read the source in {source['file']}",
                              on_click=select_evidence, args=(index, source["id"]))
        st.caption(f"{len(entry['sources'])} source excerpts · {entry['seconds']:.1f}s")


def render_source(source):
    st.html(f'<span class="evidence-badge">{escape(source["id"])} · PAGE {source["page"]}</span>')
    st.write(source["file"])
    if source["page_label"] and source["page_label"] != str(source["page"]):
        st.caption(f'Printed page label: {source["page_label"]}')
    st.text(source["text"])

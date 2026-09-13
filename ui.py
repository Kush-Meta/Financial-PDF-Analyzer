"""Presentation helpers for FinRead's conversation workspace."""
from html import escape
from pathlib import Path
import json

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
                    label = source.get("financials", {}).get("ticker") or f"p. {source['page']}"
                    st.button(f"{source['id']} · {label}",
                              key=f"source_{index}_{source['id']}",
                              help=f"Read the source in {source['file']}",
                              on_click=select_evidence, args=(index, source["id"]))
        st.caption(f"{len(entry['sources'])} sources · {entry['seconds']:.1f}s")
        if entry.get("research", {}).get("trace"):
            with st.expander("Research details"):
                st.json(entry["research"], expanded=False)


def render_source(source):
    location = "ANNUAL FINANCIALS" if source.get("kind") == "financials" else f'PAGE {source["page"]}'
    st.html(f'<span class="evidence-badge">{escape(source["id"])} · {escape(location)}</span>')
    st.write(source["file"])
    if source.get("financials"):
        facts = source["financials"]
        st.caption(f"{facts['start']} to {facts['end']} · {facts['currency']} · filed {facts['filed']}")
        st.caption("Bundled annual-report snapshot" if facts["origin"] != "live_sec" else "Retrieved from SEC")
        rows = ["| Reported figure | USD |", "| --- | ---: |"]
        rows.extend(f"| {metric.replace('_', ' ').capitalize()} | {value['value']:,.0f} |"
                    for metric, value in facts["facts"].items())
        st.markdown("\n".join(rows))
        if facts.get("operating_margin_pct") is not None:
            st.caption(f"Operating margin: {facts['operating_margin_pct']:.2f}% · operating income ÷ revenue × 100")
        with st.expander("Source data"):
            st.json(facts, expanded=False)
        return
    if source["page_label"] and source["page_label"] != str(source["page"]):
        st.caption(f'Printed page label: {source["page_label"]}')
    st.text(source["text"])


def research_notes(history):
    entries = []
    for entry in history:
        sources = []
        for source in entry["sources"]:
            location = source.get("url") or f"page {source['page']}"
            sources.append(f"[{source['id']}] {source['file']} · {location}\n\n{source['text']}")
        notes = "## " + entry["q"] + "\n\n" + entry["a"]
        if entry.get("warnings"):
            notes += "\n\nResearch limitations:\n" + "\n".join("- " + w for w in entry["warnings"])
        notes += "\n\n" + "\n\n".join(sources)
        if entry.get("research"):
            notes += "\n\nResearch record:\n\n```json\n" + json.dumps(entry["research"], indent=2) + "\n```"
        entries.append(notes)
    return "\n\n---\n\n".join(entries)

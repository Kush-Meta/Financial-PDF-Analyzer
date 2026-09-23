"""Presentation helpers for FinRead's conversation workspace."""
from html import escape
from pathlib import Path
import csv
from io import StringIO
import json
import re

import streamlit as st

# Suggested prompts mirror patterns users click first in FinSight / FinDocIntel /
# financial-rag demos: checked figures and formulas that work without Ollama.
SAMPLE_STARTERS = (
    "What was revenue in fiscal 2025?",
    "What was free cash flow in fiscal 2025?",
    "What was operating margin in fiscal 2025?",
    "What was current ratio in fiscal 2025?",
    "What was return on equity in fiscal 2025?",
    "What was net income growth from fiscal 2024 to fiscal 2025?",
)
GENERIC_STARTERS = (
    "What was revenue in fiscal 2024?",
    "What was free cash flow in fiscal 2024?",
    "What was current ratio in fiscal 2024?",
    "What was operating margin in fiscal 2024?",
    "What was return on equity in fiscal 2024?",
)


def apply_styles():
    st.html(Path(__file__).parent / "assets" / "finread.css")


def starter_prompts(sample=False):
    return SAMPLE_STARTERS if sample else GENERIC_STARTERS


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


def checked_figures_csv(history):
    """Flatten verified/unavailable figure records for spreadsheet export."""
    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow([
        "question", "status", "metric", "year", "value", "currency", "scale",
        "raw_value", "row", "source_id", "page", "formula", "result",
    ])
    for entry in history:
        verification = (entry.get("research") or {}).get("verification") or {}
        status = verification.get("status", "")
        if status not in ("verified", "unavailable"):
            continue
        calcs = verification.get("calculations") or []
        formula = "; ".join(c.get("formula", "") for c in calcs if c.get("formula"))
        result = "; ".join(str(c.get("result", "")) for c in calcs if c.get("result") is not None)
        facts = verification.get("facts") or []
        if not facts:
            writer.writerow([entry["q"], status, "", "", "", "", "", "", "", "", "", formula, result])
            continue
        for fact in facts:
            writer.writerow([
                entry["q"], status, fact.get("metric", ""), fact.get("year", ""),
                fact.get("value", ""), fact.get("currency", ""), fact.get("scale", ""),
                fact.get("raw_value", ""), fact.get("row", ""), fact.get("source_id", ""),
                fact.get("page", ""), formula, result,
            ])
    return buffer.getvalue()


def render_answer(entry, index):
    with st.chat_message("user", avatar=":material/person:"):
        st.write(entry["q"])
    with st.chat_message("assistant", avatar=":material/auto_awesome:"):
        # Financial dollar pairs are currency, not inline LaTeX delimiters.
        display = re.sub(r"(?<!\\)\$", r"\\$", entry["a"].replace("![", r"\!["))
        st.markdown(display)
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
        count = len(entry["sources"])
        st.caption(f"{count} {'source' if count == 1 else 'sources'} · {entry['seconds']:.1f}s")
        verification = entry.get("research", {}).get("verification")
        if verification:
            captions = {"verified": "Figures checked against statement rows",
                        "unavailable": "Not enough evidence to check these figures",
                        "not_checked": "Model interpretation · figures not numerically checked"}
            st.caption(captions.get(verification["status"], "Figures not numerically checked"))
            if verification["status"] != "not_checked":
                with st.expander("Figure checks"):
                    st.caption(verification["scope"])
                    for fact in verification.get("facts", []):
                        st.write(f"{fact['metric'].replace('_', ' ').capitalize()} · {fact['year']} · {fact['source_id']}")
                        st.text(fact["row"])
                        st.caption(f"{fact['currency']} {fact['scale']} · {fact['raw_value']} as reported")
                    st.json(verification, expanded=False)
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

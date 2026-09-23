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

"""Opt-in local-model integration test using synthetic, non-sensitive evidence."""
import json
import os
from pathlib import Path
import sys
import time

# Do not download models during a smoke test.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document
from finread import answer_question
from services import build_index, extract_entities, load_reranker, make_llm, make_retriever

pages = [
    Document(page_content="Example Corp reported revenue of $42 million in fiscal 2024. "
             "Revenue in fiscal 2023 was $35 million. All figures are in US dollars.",
             metadata={"source": "synthetic.pdf", "page_number": 1, "page_label": "1"}),
    Document(page_content="Example Corp operating cash flow was $9 million in fiscal 2024. "
             "Net income was $6 million. Management identified supplier concentration as a risk.",
             metadata={"source": "synthetic.pdf", "page_number": 2, "page_label": "2"}),
]
started = time.monotonic()
base = "http://localhost:11434"
print("Building real FAISS index with local embeddings…", flush=True)
index = build_index(pages, "mxbai-embed-large", base)
print("Loading cached reranker…", flush=True)
encoder = load_reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
retriever = make_retriever(index, 2, encoder, 2)
print("Asking local llama3…", flush=True)
result = answer_question("What was revenue in fiscal 2024?", retriever, make_llm("llama3", base))
assert "42" in result.text, result.text
assert not result.warnings, result.warnings
assert any(s["page"] == 1 for s in result.sources)
print(json.dumps({"answer": result.text, "source_pages": [s["page"] for s in result.sources]}), flush=True)
print("Extracting financial entities from full synthetic page…", flush=True)
rows = extract_entities(pages[:1], "llama3", base)
assert any("42" in row["entity"] and row["exact_match"] for row in rows), rows
print(json.dumps({"entities": rows, "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)

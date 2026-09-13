"""Opt-in local-model routing and public-data integration smoke test."""
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intelligence import plan_question, research_answer
from sample_document import sample_pages
from services import make_llm


def main():
    import os
    llm = make_llm(os.environ.get("FINREAD_MODEL", "llama3"), os.environ.get("OLLAMA_BASE", "http://localhost:11434"))
    pages = sample_pages()
    cases = [
        ("What was revenue?", "document"),
        ("Is this company's operating margin strong?", "competitors"),
        ("Compare Apple with Microsoft on operating margin.", "competitors"),
        ("Compare quarterly revenue with Microsoft.", "competitors"),
        ("Give Apple an S&P credit rating.", "credit"),
    ]
    for question, expected in cases:
        plan = plan_question(question, llm, pages, sample=True)
        print(json.dumps({"question": question, "plan": asdict(plan)}), flush=True)
        assert plan.route == expected, (question, plan)
        if "quarterly" in question:
            assert plan.clarification, "Quarterly request must not silently become annual"
    def no_embeddings():
        raise AssertionError("Peer research must not build a PDF index")
    answer = research_answer("Compare Apple with Microsoft on operating margin.", no_embeddings, llm, pages, sample=True)
    print(json.dumps({"status": answer.research["status"], "sources": len(answer.sources), "warnings": answer.warnings}), flush=True)
    assert len(answer.sources) == 2
    assert "31.97%" in answer.text and "45.62%" in answer.text
    assert all(s["url"].startswith("https://") for s in answer.sources)
    print("Research smoke checks passed. Inspect source origins: a snapshot result does not verify live SEC connectivity.")


if __name__ == "__main__":
    main()

# Open-source projects in FinRead’s space

Surveyed September 23, 2026. FinRead already covers local PDF Q&A with citations,
deterministic statement-cell answers, peer SEC comparisons, and evidence exports.
This note records comparable open-source work and which user-facing patterns we
adopted.

## Projects reviewed

| Project | What users get | Closest FinRead overlap |
| --- | --- | --- |
| [sec-filing-agent](https://github.com/Zhonghui-li/sec-filing-agent) | XBRL tools for exact figures, fixed ratios, YoY growth, citation guardrails, CSV export, audit trail | Numerical rigor; FinRead already abstains and checks cells |
| [FinSight](https://github.com/samadarsh/fin-sight) | Local PDF RAG, page citations, multi-company compare, Streamlit chat | Same core loop as FinRead’s document path |
| [financial-rag-chatbot](https://github.com/Kantamaniprakash/financial-rag-chatbot) | Multi-PDF upload, conversational memory, source chips | Citations and chat memory |
| [FinDocIntel](https://github.com/singhsudhir/fin-doc-intel) | Two-doc compare tab, page citations, ingest list | Side-by-side doc compare (not yet in FinRead) |
| [FinQA / table agents](https://finqasite.github.io/) | SQL/calculator tools over statement tables | Formula + evidence recording |
| [EdgarTools](https://github.com/dgunning/edgartools) | SEC company/filing/compare MCP tools | Peer research boundaries already reflected in FinRead |

## Features people actually use (and our choices)

1. **One-click starter questions** — every Streamlit financial RAG demo leads with
   suggested prompts. FinRead previously showed an empty “What would you like to
   know?” state. **Adopted:** starter prompts for checked figures and formulas.
2. **YoY growth beyond revenue** — `sec-filing-agent`’s `get_growth` is a primary
   tool. FinRead only checked revenue growth. **Adopted:** growth for any supported
   statement metric (e.g. operating income, net income).
3. **Standard ratios with explicit formulas** — ratio tools are central in
   finance-grade agents. **Adopted:** return on equity
   (`net_income / shareholders_equity × 100`) as a checked cross-statement formula.
4. **CSV export of metrics** — `sec-filing-agent` exposes `/export`. FinRead had
   JSON/Markdown research exports only. **Adopted:** CSV of checked figures from
   the conversation.
5. **Deferred for later** — multi-PDF workspaces, XBRL-live tools for arbitrary
   tickers, side-by-side PDF compare tabs, and full ratio packs (current ratio,
   leverage). Those need broader ingestion or accounting contracts than this slice.

## Design constraints we keep

- Numbers stay deterministic; the model does not invent arithmetic.
- Unsupported scopes still abstain (adjusted/organic/segment/quarterly).
- Starter prompts prefer questions FinRead can answer without Ollama when a
  recognized statement is present.

# 📄 PDF Chat Analyzer v2

A Streamlit app that lets you chat with financial PDFs (10-K filings, etc.) using a fully local AI pipeline with **upgraded embeddings**, **cross-encoder reranking**, and **LangExtract entity extraction**.

## What's New in v2

| Feature | v1 | v2 |
|---------|----|----|
| **Embedding model** | `llama2` (not an embedding model!) | `mxbai-embed-large` — SOTA for its size, outperforms OpenAI text-embedding-3-large |
| **LLM** | Mixed `llama2`/`llama3` | Configurable: `llama3`, `mistral`, `gemma2` |
| **Retrieval** | Basic vector similarity | **Two-stage retrieval with cross-encoder reranking** |
| **Entity extraction** | None | **Google's LangExtract** — structured financial entity extraction with source grounding |
| **Prompt** | Default LangChain | Custom financial analyst prompt |
| **Source transparency** | None | Expandable source chunks shown with every answer |
| **Architecture** | Single page | Tabbed UI with educational content |

## Prerequisites

1. **Ollama** running on your host machine ([install Ollama](https://ollama.com))
2. Pull the required models:

```bash
# Embedding model (REQUIRED)
ollama pull mxbai-embed-large

# LLM for Q&A and extraction (REQUIRED)
ollama pull llama3

# Optional alternative models
ollama pull nomic-embed-text
ollama pull mistral
ollama pull gemma2
```

## Quick Start

```bash
# Clone and run
docker compose up --build

# Open in browser
open http://localhost:8501
```

Or without Docker:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How Reranking Works

This is the key architectural upgrade. The app implements **two-stage retrieval**:

```
PDF → chunk → embed (mxbai-embed-large) → FAISS index
                                               │
Query → embed → cosine similarity (top 15) ────┘
                                               │
              cross-encoder rerank (top 4) ────→ Llama 3 → Answer
```

### Stage 1: Vector Search (Fast, Approximate)
- Embeds query and chunks independently (bi-encoder)
- Finds top-k candidates by cosine similarity
- **Fast** (milliseconds) but **lossy** — compresses meaning into a single vector

### Stage 2: Cross-Encoder Reranking (Slow, Precise)
- Takes each (query, chunk) pair and processes them **jointly** through a transformer
- The model attends across both query and document simultaneously
- Produces much more accurate relevance scores
- **Slower** (~200-500ms) but **dramatically more accurate** (20-35% improvement)

### Why not skip Stage 1?
Cross-encoders must evaluate each document individually against the query. For 1000 chunks, that's 1000 forward passes. Vector search narrows it to ~15 candidates first, making the cross-encoder step feasible.

### Reranker models available:
- `cross-encoder/ms-marco-MiniLM-L-6-v2` — best speed/accuracy balance
- `cross-encoder/ms-marco-TinyBERT-L-2-v2` — fastest, slightly less accurate
- `BAAI/bge-reranker-base` — strong multilingual support

## LangExtract Entity Extraction

[LangExtract](https://github.com/google/langextract) (by Google) extracts structured financial entities from your PDF:

- **Companies** — names, tickers
- **Revenue/Income figures** — with fiscal periods
- **Executives** — names and roles
- **Risk factors** — key risks mentioned
- **Fiscal periods** — year-end dates

It uses few-shot prompting with Ollama as the local LLM backend, so no API keys are needed.

## Configuration

All settings are adjustable in the sidebar:

| Setting | Default | Description |
|---------|---------|-------------|
| Embedding model | `mxbai-embed-large` | Local embedding via Ollama |
| LLM | `llama3` | For Q&A and extraction |
| Reranking | Enabled | Cross-encoder second-stage retrieval |
| Reranker model | `ms-marco-MiniLM-L-6-v2` | Lightweight cross-encoder |
| Initial retrieval k | 15 | Chunks fetched by vector search |
| Rerank top-n | 4 | Chunks kept after reranking |
| Chunk size | 1500 | Characters per chunk |
| Chunk overlap | 200 | Overlap between chunks |

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container build
├── docker-compose.yml     # Easy deployment
└── README.md              # This file
```
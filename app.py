import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.prompts import PromptTemplate
import langextract as lx
import textwrap
import time
import json
import os

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(page_title="📄 PDF Chat Analyzer", layout="wide")
st.title("💬 Chat with your Financial PDF")
st.caption(
    "**mxbai-embed-large** embeddings · **Cross-Encoder reranking** · "
    "**LangExtract** entity extraction · **Llama 3** analysis"
)

OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")

# ─────────────────────────────────────────────
# Sidebar – Configuration
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("Embedding Model")
    embed_model = st.selectbox(
        "Choose embedding model (via Ollama)",
        ["mxbai-embed-large", "nomic-embed-text"],
        index=0,
        help="mxbai-embed-large: higher accuracy. nomic-embed-text: faster.",
    )

    st.subheader("LLM for Q&A")
    llm_model = st.selectbox(
        "Choose LLM (via Ollama)",
        ["llama3", "mistral", "gemma2"],
        index=0,
    )

    st.subheader("Reranking")
    enable_reranking = st.toggle("Enable Cross-Encoder Reranking", value=True)
    if enable_reranking:
        rerank_model_name = st.selectbox(
            "Reranker model",
            [
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "cross-encoder/ms-marco-TinyBERT-L-2-v2",
                "BAAI/bge-reranker-base",
            ],
            index=0,
        )
        top_k_retrieve = st.slider("Initial retrieval (k)", 5, 30, 15)
        top_n_rerank = st.slider("Keep after reranking (n)", 1, 10, 4)
    else:
        top_k_retrieve = st.slider("Retrieval (k)", 1, 15, 4)

    st.subheader("LangExtract")
    enable_extraction = st.toggle("Enable Financial Entity Extraction", value=True)

    st.markdown("---")
    st.markdown(
        "**How reranking works:**\n\n"
        "1️⃣ Vector search retrieves a broad set of candidate chunks "
        "(fast but lossy).\n\n"
        "2️⃣ A cross-encoder reads each query+chunk pair jointly, "
        "producing much more accurate relevance scores.\n\n"
        "Result: fewer but *better* chunks → better answers."
    )


# ─────────────────────────────────────────────
# LangExtract – Financial entity extraction
# ─────────────────────────────────────────────
def run_langextract(text: str) -> list:
    """Use LangExtract to pull structured financial entities via Ollama."""

    prompt = textwrap.dedent("""\
        Extract financial entities from documents in the order they appear.
        Focus on: company names, revenue figures, net income,
        earnings per share, fiscal year periods, and key executives.
        Use exact text for extractions. Do not paraphrase.""")

    example_text = (
        "Apple Inc. reported total net revenue of $383.3 billion "
        "for the fiscal year ended December 31, 2023. "
        "Net income was $97.0 billion."
    )

    examples = [
        lx.data.ExampleData(
            text=example_text,
            extractions=[
                lx.data.Extraction(
                    extraction_class="company",
                    extraction_text="Apple Inc.",
                ),
                lx.data.Extraction(
                    extraction_class="revenue",
                    extraction_text="$383.3 billion",
                ),
                lx.data.Extraction(
                    extraction_class="fiscal_period",
                    extraction_text="fiscal year ended December 31, 2023",
                ),
                lx.data.Extraction(
                    extraction_class="net_income",
                    extraction_text="$97.0 billion",
                ),
            ],
        )
    ]

    try:
        # Only send first ~3000 chars to stay within Ollama context limits
        input_text = text[:3000]
        if not input_text.strip():
            return []

        result = lx.extract(
            text_or_documents=input_text,
            prompt_description=prompt,
            examples=examples,
            language_model_type=lx.inference.OllamaLanguageModel,
            model_id="llama3",
            model_url=OLLAMA_BASE,
            fence_output=False,
            use_schema_constraints=False,
        )
        return result.extractions if result and result.extractions else []
    except Exception as e:
        st.warning(f"⚠️ LangExtract error: {e}")
        return []


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
pdf = st.file_uploader("Upload a 10-K or other financial PDF", type="pdf")

if pdf:
    with open("uploaded.pdf", "wb") as f:
        f.write(pdf.read())

    # ── Tabs ────────────────────────────────
    tab_chat, tab_entities, tab_learn = st.tabs([
        "🧠 Q&A Chat", "🏷️ Entity Extraction", "📖 How Reranking Works"
    ])

    # ── TAB 1: Chat ─────────────────────────
    with tab_chat:

        # Embed the document
        with st.spinner("📚 Embedding your document…"):
            start = time.time()

            loader = PyPDFLoader("uploaded.pdf")
            docs = loader.load()

            # mxbai-embed-large context is ~512 tokens ≈ ~1500 chars
            # Keep chunks well within that limit
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150,
            )
            chunks = splitter.split_documents(docs)

            embeddings = OllamaEmbeddings(
                model=embed_model,
                base_url=OLLAMA_BASE,
            )
            vectorstore = FAISS.from_documents(chunks, embeddings)

            elapsed = round(time.time() - start, 2)
            st.success(
                f"✅ Embedded **{len(chunks)}** chunks in **{elapsed}s** "
                f"using **{embed_model}**"
            )

        # Build retriever (with optional reranking)
        if enable_reranking:
            base_retriever = vectorstore.as_retriever(
                search_kwargs={"k": top_k_retrieve}
            )
            with st.spinner("Loading cross-encoder reranker…"):
                cross_encoder = HuggingFaceCrossEncoder(
                    model_name=rerank_model_name
                )
                compressor = CrossEncoderReranker(
                    model=cross_encoder, top_n=top_n_rerank
                )
                retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=base_retriever,
                )
            st.info(
                f"🔀 Reranking: top {top_k_retrieve} → "
                f"best {top_n_rerank} via `{rerank_model_name}`"
            )
        else:
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": top_k_retrieve}
            )

        # LLM + QA chain
        llm = Ollama(model=llm_model, base_url=OLLAMA_BASE)

        qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=textwrap.dedent("""\
                You are an expert financial analyst. Use the following context
                from a financial document to answer the question. Be precise
                with numbers and cite specific sections when possible. If you
                cannot find the answer in the context, say so clearly.

                Context:
                {context}

                Question: {question}

                Answer:"""),
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": qa_prompt},
            return_source_documents=True,
        )

        # Chat interface
        st.subheader("🧠 Ask a question about your document")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        query = st.text_input("🔍 Your question:")

        if query:
            with st.spinner("Thinking…"):
                result = qa_chain.invoke({"query": query})
                answer = result["result"]
                sources = result.get("source_documents", [])

            st.session_state.chat_history.append({
                "q": query,
                "a": answer,
                "sources": sources,
            })

        for entry in reversed(st.session_state.chat_history):
            st.markdown(f"**You:** {entry['q']}")
            st.markdown(f"**Bot:** {entry['a']}")
            if entry.get("sources"):
                with st.expander("📄 Source chunks used"):
                    for i, doc in enumerate(entry["sources"]):
                        pg = doc.metadata.get("page", "?")
                        st.markdown(f"**Chunk {i+1}** (page {pg}):")
                        st.text(doc.page_content[:400])
            st.markdown("---")

    # ── TAB 2: LangExtract ──────────────────
    with tab_entities:
        st.subheader("🏷️ Financial Entity Extraction (LangExtract)")
        st.markdown(
            "Uses [Google's LangExtract](https://github.com/google/langextract) "
            "to identify structured financial entities with source grounding."
        )

        if not enable_extraction:
            st.info("Enable entity extraction in the sidebar.")
        else:
            if st.button("🚀 Run Entity Extraction"):
                with st.spinner("Running LangExtract via Ollama…"):
                    loader = PyPDFLoader("uploaded.pdf")
                    docs = loader.load()
                    combined = "\n".join([d.page_content for d in docs[:5]])
                    extractions = run_langextract(combined)

                if extractions:
                    st.success(f"✅ Extracted **{len(extractions)}** entities")
                    rows = []
                    for ext in extractions:
                        row = {
                            "Class": getattr(ext, "extraction_class", ""),
                            "Entity": getattr(ext, "extraction_text", ""),
                        }
                        attrs = getattr(ext, "attributes", None)
                        if attrs:
                            row["Attributes"] = json.dumps(attrs)
                        rows.append(row)
                    import pandas as pd
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    st.download_button(
                        "⬇️ Download Entities JSON",
                        data=json.dumps(rows, indent=2),
                        file_name="extracted_entities.json",
                        mime="application/json",
                    )
                else:
                    st.info(
                        "No entities extracted. This can happen if the first "
                        "few pages don't contain financial data (e.g. cover "
                        "pages or table of contents). Try a different document."
                    )

    # ── TAB 3: How Reranking Works ──────────
    with tab_learn:
        st.subheader("📖 Understanding Two-Stage Retrieval with Reranking")
        st.markdown("""
**Why does basic vector search sometimes fail?**

When you embed a chunk into a vector, you compress all its meaning into a
single point in high-dimensional space. This is fast but *lossy* — subtle
nuances get lost. A chunk that's *topically* similar might rank high even
if it doesn't actually *answer* your question.

---

**The Two-Stage Solution**

| Stage | Method | Speed | Accuracy |
|-------|--------|-------|----------|
| 1 — Retrieval | Bi-encoder (vector similarity) | ⚡ Very fast | 🟡 Good |
| 2 — Reranking | Cross-encoder (joint attention) | 🐢 Slower | 🟢 Excellent |

**Stage 1** casts a wide net — fetching ~15 candidate chunks via cosine
similarity.

**Stage 2** feeds each (query, chunk) pair *together* through a cross-encoder
transformer. Because the model attends across both simultaneously, it catches
nuanced relevance that bi-encoders miss.

---

**Why not just use a cross-encoder for everything?**

Cross-encoders process every query-document pair individually. For 10,000
chunks that's 10,000 forward passes per query — far too slow. The two-stage
approach gives you speed from vector search + precision from the cross-encoder.

---

**Pipeline in this app:**

```
PDF → Chunks → mxbai-embed-large → FAISS vector store
                                         │
Query → embed → cosine search (top 15) ──┘
                                         │
          cross-encoder rerank (top 4) ──→ Llama 3 → Answer
```

**Key metrics:**
- Reranking improves accuracy ~20-35%
- Adds ~200-500ms latency per query (worth it for quality)
        """)
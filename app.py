import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
import pdfplumber
import pandas as pd
import time

st.set_page_config(page_title="📄 PDF Chat Analyzer", layout="wide")
st.title("💬 Chat with your Financial PDF!")

pdf = st.file_uploader("Upload a 10-K or other financial PDF", type="pdf")

# Function to extract and clean tables from PDF
def extract_clean_tables(pdf_path):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            try:
                text = page.extract_text() or ""
                raw_tables = page.extract_tables()
                for table in raw_tables:
                    if not table or not table[0]:
                        continue
                    df = pd.DataFrame(table[1:], columns=table[0])
                    df.dropna(how='all', inplace=True)
                    df.dropna(axis=1, how='all', inplace=True)
                    df.columns = [col.strip().replace('\n', ' ') if col else '' for col in df.columns]
                    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
                    tables.append((i + 1, text[:200], df))  # page number, heading, table
            except Exception as e:
                print(f"⚠️ Failed to process page {i+1}: {e}")
    return tables

if pdf:
    with open("uploaded.pdf", "wb") as f:
        f.write(pdf.read())

    tables = extract_clean_tables("uploaded.pdf")
    st.subheader("📊 Extracted Financial Tables")
    st.write(f"🔍 Extracted {len(tables)} tables from this PDF.")

    search_term = st.text_input("🔎 Search for a keyword in all tables (optional)")

    for page_num, title, df in tables:
        match = True
        if search_term:
            match = df.astype(str).apply(lambda x: search_term.lower() in x.str.lower().to_string(), axis=1).any()

        if match:
            st.markdown(f"### 📄 Table from Page {page_num}")
            st.markdown(f"**Title Preview:** {title.strip()}")
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download this Table as CSV",
                data=csv,
                file_name=f"table_page_{page_num}.csv",
                mime="text/csv"
            )

    with st.spinner("📚 Reading and embedding your document... please wait..."):
        start = time.time()

        loader = PyPDFLoader("uploaded.pdf")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        chunks = splitter.split_documents(docs)

        embeddings = OllamaEmbeddings(
            model="llama2",
            base_url="http://host.docker.internal:11434"
        )
        vectorstore = FAISS.from_documents(chunks, embeddings)

        end = time.time()
        st.success(f"✅ Document processed in {round(end - start, 2)} seconds")

        retriever = vectorstore.as_retriever()
        llm = Ollama(
            model="llama3",
            base_url="http://host.docker.internal:11434"
        )
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        st.subheader("🧠 Ask your question")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        query = st.text_input("🔍 Ask a question about the document:")

        if query:
            response = qa_chain.run(query)
            st.session_state.chat_history.append((query, response))

        for i, (q, a) in enumerate(st.session_state.chat_history[::-1]):
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Bot:** {a}")
            st.markdown("---")
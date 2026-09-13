"""Local model integrations. No uploaded document is saved to disk."""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from finread import CHUNK_OVERLAP, CHUNK_SIZE


def build_index(pages, model, base_url):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents([p for p in pages if p.page_content.strip()])
    embeddings = OllamaEmbeddings(model=model, base_url=base_url, client_kwargs={"timeout": 120})
    return FAISS.from_documents(chunks, embeddings)


def load_reranker(model_name):
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    return HuggingFaceCrossEncoder(model_name=model_name)


def make_retriever(index, k, cross_encoder=None, top_n=4):
    retriever = index.as_retriever(search_kwargs={"k": k})
    if cross_encoder is not None:
        from langchain.retrievers import ContextualCompressionRetriever
        from langchain.retrievers.document_compressors import CrossEncoderReranker
        return ContextualCompressionRetriever(
            base_retriever=retriever,
            base_compressor=CrossEncoderReranker(model=cross_encoder, top_n=min(k, top_n)),
        )
    return retriever


def make_llm(model, base_url):
    return OllamaLLM(model=model, base_url=base_url, temperature=0,
                     num_ctx=8192, num_predict=1024, client_kwargs={"timeout": 180})


def extract_entities(pages, model_name, base_url):
    import langextract as lx
    from langextract.providers.ollama import OllamaLanguageModel

    example = lx.data.ExampleData(
        text="Example Corp reported revenue of $42 million in fiscal 2024.",
        extractions=[
            lx.data.Extraction(extraction_class="company", extraction_text="Example Corp"),
            lx.data.Extraction(extraction_class="revenue", extraction_text="$42 million"),
            lx.data.Extraction(extraction_class="fiscal_period", extraction_text="fiscal 2024"),
        ],
    )
    model = OllamaLanguageModel(model_id=model_name, model_url=base_url, timeout=180)
    rows = []
    for page in pages:
        if not page.page_content.strip():
            continue
        result = lx.extract(
            text_or_documents=page.page_content,
            prompt_description=("Extract company names, revenue, net income, earnings per share, "
                                "fiscal periods, and executives in order. Use exact source text. "
                                "Include units and periods in attributes when explicitly present."),
            examples=[example], model=model, fence_output=False,
            use_schema_constraints=False, max_char_buffer=1500, max_workers=1,
            fetch_urls=False, show_progress=False,
        )
        for entity in result.extractions or []:
            interval = entity.char_interval
            start = getattr(interval, "start_pos", None)
            end = getattr(interval, "end_pos", None)
            exact = (isinstance(start, int) and isinstance(end, int)
                     and 0 <= start < end <= len(page.page_content)
                     and page.page_content[start:end] == entity.extraction_text)
            rows.append({"class": entity.extraction_class, "entity": entity.extraction_text,
                         "page": page.metadata["page_number"], "file": page.metadata["source"],
                         "exact_match": exact, "start": start, "end": end,
                         "attributes": entity.attributes or {}})
    return rows

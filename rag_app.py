import os
import sys
import shutil
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


# ======================================================================
# 1. CONFIG
# ======================================================================

load_dotenv()  # reads .env file and loads GROQ_API_KEY into environment

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Create a .env file (see .env.example) "
        "and add your free Groq API key from https://console.groq.com/keys"
    )

# LLM settings — free models on Groq. Swap freely.
# Options: "llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"
GROQ_MODEL_NAME = "openai/gpt-oss-120b"
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 1024

# Embedding settings — free, local, runs on CPU, no API key needed
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Text splitting settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Vector store settings
VECTOR_DB_DIR = "vector_store_db"   # where the FAISS index is saved on disk

# Retriever settings
RETRIEVER_TOP_K = 4   # number of chunks fetched per question

# Upload settings (used by the Streamlit UI)
UPLOAD_DIR = "uploaded_docs"


# ======================================================================
# 2. DOCUMENT LOADER — Step 1: load raw PDF/TXT into Documents
# ======================================================================

def load_document(file_path: str) -> List[Document]:
    """Load a single file (.pdf or .txt) into LangChain Document objects."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}. Only .pdf and .txt are supported.")

    documents = loader.load()
    for doc in documents:
        doc.metadata["source_file"] = os.path.basename(file_path)
    return documents


def load_multiple_documents(file_paths: List[str]) -> List[Document]:
    """Load several files at once and combine them into one list of Documents."""
    all_docs: List[Document] = []
    for path in file_paths:
        try:
            docs = load_document(path)
            all_docs.extend(docs)
            print(f"[loader] Loaded {len(docs)} page(s) from {os.path.basename(path)}")
        except Exception as e:
            print(f"[loader] Failed to load {path}: {e}")
    return all_docs


# ======================================================================
# 3. TEXT SPLITTER — Step 2: chunk documents for embedding
# ======================================================================

def split_documents(documents: List[Document]) -> List[Document]:
    """Split Documents into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    print(f"[splitter] Split {len(documents)} document(s) into {len(chunks)} chunks.")
    return chunks


# ======================================================================
# 4. VECTOR STORE — Step 3: embeddings + FAISS index
# ======================================================================

def get_embedding_model() -> HuggingFaceEmbeddings:
    """Free, local sentence-transformers embedding model (CPU, no API key)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(chunks: List[Document], persist_dir: str = VECTOR_DB_DIR) -> FAISS:
    """Embed chunks and build a fresh FAISS index, saved to disk."""
    embedding_model = get_embedding_model()
    vector_db = FAISS.from_documents(documents=chunks, embedding=embedding_model)
    os.makedirs(persist_dir, exist_ok=True)
    vector_db.save_local(persist_dir)
    print(f"[vector_store] Stored {len(chunks)} chunks in FAISS index at '{persist_dir}'.")
    return vector_db


def load_vector_store(persist_dir: str = VECTOR_DB_DIR) -> FAISS:
    """Load an already-built FAISS index from disk."""
    embedding_model = get_embedding_model()
    return FAISS.load_local(persist_dir, embedding_model, allow_dangerous_deserialization=True)


# ======================================================================
# 5. LLM — free Groq chat model
# ======================================================================

def get_llm() -> ChatGroq:
    """Configured Groq chat model instance (free tier)."""
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )


# ======================================================================
# 6. RAG PIPELINE — Step 4-5: retriever + prompt + chain
# ======================================================================

RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions using ONLY the
context provided below, which was retrieved from the user's uploaded document(s).

Rules:
- Answer strictly using the given context. Do not use outside knowledge.
- If the answer is not present in the context, say clearly:
  "I couldn't find this information in the document."
- Be concise and accurate. Quote specific figures/terms from the context where relevant.
- If helpful, mention which part of the document (page/source) the answer came from.

Context:
{context}

Question:
{question}

Answer:"""


def format_docs(docs: List[Document]) -> str:
    """Combine retrieved chunks into a single context string for the prompt."""
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "N/A")
        formatted.append(f"[Source: {source}, Page: {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


class RAGPipeline:
    """Wraps the full pipeline: ingest documents once, then ask unlimited questions."""

    def __init__(self):
        self.vector_db = None
        self.retriever = None
        self.chain = None
        self.llm = get_llm()

    def ingest(self, file_paths: List[str]):
        """Steps 1-3: load -> split -> embed & store. Call once per new document set."""
        print("[pipeline] Loading documents...")
        raw_docs = load_multiple_documents(file_paths)
        if not raw_docs:
            raise ValueError("No documents were loaded. Check file paths/types.")

        print("[pipeline] Splitting into chunks...")
        chunks = split_documents(raw_docs)

        print("[pipeline] Building vector store (embedding)...")
        self.vector_db = build_vector_store(chunks)

        self._build_retrieval_chain()
        print("[pipeline] Ingestion complete. Ready for questions.")

    def load_existing(self):
        """Skip ingestion, load a previously-persisted FAISS index."""
        self.vector_db = load_vector_store()
        self._build_retrieval_chain()

    def _build_retrieval_chain(self):
        """Step 4 (retriever) + Step 5 (prompt -> llm -> parser) wired as a chain."""
        self.retriever = self.vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVER_TOP_K},
        )
        prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question: str) -> str:
        """Ask a question against the ingested document(s)."""
        if self.chain is None:
            raise RuntimeError("No documents ingested yet. Call ingest() first.")
        return self.chain.invoke(question)

    def get_sources(self, question: str) -> List[Document]:
        """Return the raw chunks retrieved for a question (for citation/debugging)."""
        if self.retriever is None:
            raise RuntimeError("No documents ingested yet. Call ingest() first.")
        return self.retriever.invoke(question)


# ======================================================================
# 7. STREAMLIT UI — only runs when launched via `streamlit run rag_app.py`
# ======================================================================

def run_streamlit_app():
    import streamlit as st

    st.set_page_config(page_title="Chat with your Documents (RAG + Groq)", page_icon="📄")
    st.title("📄 Chat with your PDF / TXT — Free RAG (LangChain + Groq)")

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "ingested" not in st.session_state:
        st.session_state.ingested = False

    with st.sidebar:
        st.header("Upload documents")
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT file(s)", type=["pdf", "txt"], accept_multiple_files=True
        )

        if st.button("Process documents", type="primary", disabled=not uploaded_files):
            saved_paths = []
            for uploaded_file in uploaded_files:
                save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_paths.append(save_path)

            with st.spinner("Reading, chunking, and embedding your documents..."):
                pipeline = RAGPipeline()
                pipeline.ingest(saved_paths)
                st.session_state.pipeline = pipeline
                st.session_state.ingested = True
                st.session_state.chat_history = []

            st.success(f"Processed {len(saved_paths)} file(s). Ask away!")

        st.divider()
        st.caption(
            "Embeddings run locally and free (sentence-transformers). "
            "Answers are generated by a free Groq-hosted LLM."
        )

    st.header("Ask questions")

    if not st.session_state.ingested:
        st.info("Upload and process a document from the sidebar to get started.")
        return

    for role, text in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(text)

    user_question = st.chat_input("Ask something about your document...")

    if user_question:
        st.session_state.chat_history.append(("user", user_question))
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.pipeline.ask(user_question)
                st.markdown(answer)

                with st.expander("View retrieved source chunks"):
                    sources = st.session_state.pipeline.get_sources(user_question)
                    for i, doc in enumerate(sources, 1):
                        src = doc.metadata.get("source_file", "unknown")
                        page = doc.metadata.get("page", "N/A")
                        st.markdown(f"**Chunk {i} — {src}, page {page}**")
                        st.text(doc.page_content[:500])

        st.session_state.chat_history.append(("assistant", answer))


# ======================================================================
# ENTRY POINT
# ======================================================================

def run_cli():
    """Terminal mode: python rag_app.py file1.pdf [file2.txt] ..."""
    if len(sys.argv) < 2:
        print("Usage: python rag_app.py <file1.pdf> [file2.txt] ...")
        sys.exit(1)

    pipeline = RAGPipeline()
    pipeline.ingest(sys.argv[1:])

    print("\nDocument ready. Type your questions ('exit' to quit).\n")
    while True:
        q = input("You: ")
        if q.strip().lower() in {"exit", "quit"}:
            break
        answer = pipeline.ask(q)
        print(f"\nAI: {answer}\n")


# Streamlit re-executes this whole script on every interaction and sets this
# env var internally — but the simplest reliable check is whether the script
# is being run under the `streamlit run` command, which imports streamlit
# and calls this module directly rather than via __main__ with argv.
if __name__ == "__main__":
    try:
        import streamlit.runtime.scriptrunner as _sr
        in_streamlit = _sr.get_script_run_ctx() is not None
    except Exception:
        in_streamlit = False

    if in_streamlit:
        run_streamlit_app()
    else:
        run_cli()
else:
    # When Streamlit imports this file as a module (`streamlit run rag_app.py`
    # actually runs it as __main__ too, so this branch rarely triggers — kept
    # as a safety net for edge cases).
    pass

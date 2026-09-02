# RAG Based Document Q&A System

Upload any PDF or TXT file and ask questions about it. Fully free stack:
- **LLM**: Groq (free tier, e.g. `openai/gpt-oss-120b`)
- **Embeddings**: local `sentence-transformers/all-MiniLM-L6-v2` (no API cost)
- **Vector store**: FAISS (local, in-memory index saved to disk)

##  Live Deployments & UI Links
✨ **Click on the badges below to interact with the project and view the user interface:**

[![Streamlit App](streamlit.jpg)](https://rag-based-document-q-a-system-cnaukv9dnt5mzbal3uafwc.streamlit.app/)


## Pipeline

```
PDF/TXT --> document_loader.py --> text_splitter.py --> vector_store.py
                                                              |
                                                         (embeddings)
                                                              |
question --------------------------------------------> rag_pipeline.py
                                                       (retriever -> prompt -> llm)
                                                              |
                                                           answer
```

## File structure

```
rag_pdf_qa/
├── config.py           # all settings: models, chunk size, paths
├── document_loader.py  # Step 1: load PDF/TXT into Documents
├── text_splitter.py    # Step 2: chunk documents
├── vector_store.py     # Step 3: embeddings + FAISS vector index
├── llm.py               # Groq LLM setup
├── rag_pipeline.py     # Step 4-5: retriever + prompt + chain (core logic)
├── app.py               # Streamlit UI
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Get a free Groq API key**
   Sign up at https://console.groq.com/keys (free, no credit card needed).

3. **Configure your key**
   ```bash
   cp .env.example .env
   # then open .env and paste your key:
   # GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
   ```

## Run

**Option A — Web UI (recommended)**
```bash
streamlit run app.py
```
Upload a PDF/TXT in the sidebar, click "Process documents", then chat.

**Option B — Command line**
```bash
python rag_pipeline.py path/to/your_file.pdf
```
Then type questions directly in the terminal.

## How it works (step by step)

1. **`document_loader.py`** — `PyPDFLoader` / `TextLoader` reads the file and returns LangChain `Document` objects (one per PDF page).
2. **`text_splitter.py`** — `RecursiveCharacterTextSplitter` breaks documents into ~1000-character overlapping chunks so they fit the embedding model and give the LLM focused context.
3. **`vector_store.py`** — each chunk is embedded with a free local HuggingFace model and stored in a FAISS index, which is saved to disk (`vector_store_db/`) so it can be reloaded without re-embedding.
4. **`rag_pipeline.py`** — builds a retriever (`vector_db.as_retriever`) that does similarity search to fetch the top-k relevant chunks for a question, then feeds those chunks + the question into a prompt template, which is sent to the Groq LLM. LangChain Expression Language (`|`) chains these steps together.
5. **`app.py`** — Streamlit front-end: upload files, trigger ingestion, chat, and inspect which chunks were used to answer (for transparency/debugging).

## Customization tips

- Change `GROQ_MODEL_NAME` in `config.py` to try other free Groq models (`llama-3.1-8b-instant` is faster/cheaper on tokens; `llama-3.3-70b-versatile` is more capable).
- Increase `RETRIEVER_TOP_K` in `config.py` if answers seem to be missing context.
- Adjust `CHUNK_SIZE` / `CHUNK_OVERLAP` for very long or very short documents.
- Swap FAISS for Chroma if you'd rather have a database-style store with metadata filtering built in.

## Pushing this project to GitHub

1. **Create a new repo** on GitHub (github.com → New repository). Don't initialize it with a README since you already have one — just create it empty.

2. **Initialize git locally** (inside the `rag_pdf_qa` folder):
   ```bash
   cd rag_pdf_qa
   git init
   git add .
   git commit -m "Initial commit: RAG PDF/TXT QA pipeline with LangChain + Groq"
   ```

3. **Connect it to your GitHub repo and push:**
   ```bash
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```
   (Use the HTTPS or SSH URL GitHub shows you right after creating the repo.)

4. **Important — never commit your API key.** This repo already includes a `.gitignore` that excludes `.env`, so your real `GROQ_API_KEY` stays local. Only `.env.example` (the template) gets pushed. Double-check with:
   ```bash
   git status
   ```
   `.env` should NOT appear in the list of tracked/staged files.

5. **Anyone who clones your repo** (including you, on a new machine) just runs:
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   pip install -r requirements.txt
   cp .env.example .env   # then paste in a Groq API key
   streamlit run app.py
   ```

6. **Optional polish for the repo:**
   - Add a screenshot of the Streamlit app to the README (`![demo](screenshot.png)`).
   - Add a short GIF/demo video link if you want it to stand out for placements or a portfolio.
   - Add topics/tags on GitHub like `rag`, `langchain`, `groq`, `llm`, `genai` so it's discoverable.

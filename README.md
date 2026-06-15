# Contract Clause Q&A Agent

A RAG-based agent that lets users upload a contract PDF and ask natural language questions about its clauses. Built with LangGraph (ReAct), FAISS, sentence-transformers, FastAPI, and Streamlit.

## Architecture

```
User (Streamlit) → FastAPI → LangGraph ReAct Agent
                                  ├── Tool: semantic_search  → FAISS vector store
                                  └── Tool: clause_lookup    → clause-type matcher
                                  └── LLM: claude-sonnet-4-6 (Anthropic)
```

## Folder Structure

```
contract-qa-agent/
├── app/
│   ├── core/
│   │   └── config.py          # Env vars, shared settings
│   ├── ingestion/
│   │   ├── pdf_parser.py      # PDF → raw text (pypdf)
│   │   ├── chunker.py         # Text → overlapping chunks
│   │   └── embedder.py        # Chunks → embeddings → FAISS index
│   ├── vectorstore/
│   │   └── store.py           # FAISS load/save/search wrapper
│   ├── agent/
│   │   ├── tools.py           # semantic_search + clause_lookup tool defs
│   │   └── graph.py           # LangGraph ReAct graph
│   ├── api/
│   │   ├── routes.py          # POST /upload, POST /query
│   │   └── schemas.py         # Pydantic request/response models
│   └── main.py                # FastAPI app entrypoint
├── frontend/
│   └── app.py                 # Streamlit UI
├── data/                      # Runtime storage for FAISS index + chunks
├── requirements.txt
├── .env.example
└── README.md
```

## Modules (build order)

1. **`app/core/config.py`** — load `ANTHROPIC_API_KEY` and constants (chunk size, overlap, model name, index path)
2. **`app/ingestion/pdf_parser.py`** — `parse_pdf(file_bytes) -> str`
3. **`app/ingestion/chunker.py`** — `chunk_text(text) -> list[dict]` with `text`, `page`, `chunk_id`
4. **`app/ingestion/embedder.py`** — `build_index(chunks) -> None` (saves FAISS index + metadata to `data/`)
5. **`app/vectorstore/store.py`** — `similarity_search(query, k) -> list[dict]`
6. **`app/agent/tools.py`** — `semantic_search` tool, `clause_lookup` tool (searches by clause type keyword)
7. **`app/agent/graph.py`** — LangGraph ReAct graph binding tools to `claude-sonnet-4-6`
8. **`app/api/schemas.py`** — `UploadResponse`, `QueryRequest`, `QueryResponse`
9. **`app/api/routes.py`** — `/upload` ingests PDF, `/query` runs agent
10. **`app/main.py`** — mounts router, runs with uvicorn
11. **`frontend/app.py`** — Streamlit file uploader + streaming chat

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env          # add your ANTHROPIC_API_KEY
```

## Running

```bash
# Terminal 1 — backend
uvicorn app.main:app --reload --port 8000

# Terminal 2 — frontend
streamlit run frontend/app.py
```

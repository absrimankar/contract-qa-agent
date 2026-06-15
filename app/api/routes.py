from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.agent.graph import create_agent, run_agent
from app.api.schemas import QueryRequest, QueryResponse, UploadResponse
from app.ingestion.chunker import chunk_pages
from app.ingestion.embedder import build_index
from app.ingestion.pdf_parser import parse_pdf
from app.vectorstore.store import VectorStore

router = APIRouter()


def _get_or_load_agent(request: Request):
    """Return cached agent or load from disk; raises 400 if no index exists."""
    if request.app.state.agent is not None:
        return request.app.state.agent

    store = VectorStore()
    if not store.load():
        raise HTTPException(
            status_code=400,
            detail="No contract indexed yet. Please upload a PDF via POST /upload first.",
        )
    agent = create_agent(store)
    request.app.state.vector_store = store
    request.app.state.agent = agent
    return agent


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/upload", response_model=UploadResponse)
async def upload_contract(request: Request, file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    pages = parse_pdf(file_bytes)
    if not pages:
        raise HTTPException(
            status_code=422,
            detail="No extractable text found in the PDF. Scanned PDFs are not supported.",
        )

    chunks = chunk_pages(pages)
    build_index(chunks)

    # Rebuild in-memory state so queries use the new index immediately
    store = VectorStore()
    store.load()
    request.app.state.vector_store = store
    request.app.state.agent = create_agent(store)

    return UploadResponse(
        message="Contract uploaded and indexed successfully.",
        page_count=len(pages),
        chunk_count=len(chunks),
    )


@router.post("/query", response_model=QueryResponse)
async def query_contract(request: Request, body: QueryRequest):
    agent = _get_or_load_agent(request)
    answer = run_agent(agent, body.question)
    return QueryResponse(question=body.question, answer=answer)

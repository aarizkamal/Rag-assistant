"""
FastAPI backend for the RAG assistant.

Endpoints:
  POST /ingest         - ingest all files in data/sample_docs (or an uploaded file)
  POST /upload          - upload a single file and ingest it immediately
  POST /query           - ask a question, get a grounded answer + sources
  GET  /health          - basic health/status check
  GET  /stats            - number of vectors currently stored
"""
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.ingest import ingest_directory, ingest_file
from app.rag_chain import answer_query
from app.vectorstore import vector_store

app = FastAPI(title="RAG Knowledge Assistant", version="1.0.0")

# Allow the local frontend (or any origin during dev) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    top_k: int | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    return {"vector_count": vector_store.count()}


@app.post("/ingest")
def ingest_default_directory():
    """Ingest every file currently in data/sample_docs."""
    summary = ingest_directory("./data/sample_docs")
    if not summary:
        raise HTTPException(status_code=400, detail="No supported files found in data/sample_docs")
    return {"ingested": summary, "total_vectors": vector_store.count()}


@app.post("/upload")
async def upload_and_ingest(file: UploadFile = File(...)):
    """Upload a single document and ingest it immediately."""
    allowed = {".txt", ".md", ".pdf"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    dest_dir = Path("./data/uploads")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / file.filename

    contents = await file.read()
    dest_path.write_bytes(contents)

    n_chunks = ingest_file(dest_path)
    return {"filename": file.filename, "chunks_added": n_chunks, "total_vectors": vector_store.count()}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    result = answer_query(request.query, top_k=request.top_k)
    return QueryResponse(answer=result["answer"], sources=result["sources"])


# Serve the simple chat frontend at /
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

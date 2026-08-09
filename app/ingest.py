"""
Ingestion pipeline: load raw documents (.txt / .pdf / .md), split them into
overlapping chunks, and push them into the vector store with source metadata
so answers can be traced back to their origin document.
"""
import hashlib
import os
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import settings
from app.vectorstore import vector_store

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_file_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _read_pdf(path)
    if ext in (".txt", ".md"):
        return _read_txt(path)
    raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def _chunk_id(source: str, index: int, text: str) -> str:
    """Deterministic ID so re-ingesting the same file doesn't create duplicates."""
    digest = hashlib.sha1(f"{source}-{index}-{text[:50]}".encode()).hexdigest()[:12]
    return f"{Path(source).stem}-{index}-{digest}"


def ingest_file(path: Path) -> int:
    """Ingest a single file. Returns number of chunks added."""
    text = load_file_text(path)
    if not text.strip():
        return 0

    chunks = chunk_text(text)
    ids = [_chunk_id(str(path), i, c) for i, c in enumerate(chunks)]
    metadatas = [
        {"source": path.name, "chunk_index": i, "path": str(path)}
        for i in range(len(chunks))
    ]

    vector_store.add_documents(ids=ids, texts=chunks, metadatas=metadatas)
    return len(chunks)


def ingest_directory(directory: str) -> dict:
    """Ingest every supported file in a directory. Returns a summary dict."""
    dir_path = Path(directory)
    results = {}
    for file_path in sorted(dir_path.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            n_chunks = ingest_file(file_path)
            results[file_path.name] = n_chunks
    return results


if __name__ == "__main__":
    # Quick manual run: python -m app.ingest
    summary = ingest_directory("./data/sample_docs")
    print("Ingestion complete:")
    for fname, n in summary.items():
        print(f"  {fname}: {n} chunks")
    print(f"Total vectors in store: {vector_store.count()}")

"""
Thin wrapper around ChromaDB that handles embedding + persistence.
Embeddings run locally via ChromaDB's built-in ONNX MiniLM model, so the
knowledge base can be built and queried with zero external API calls
(only the final answer-generation step calls an LLM). This avoids a heavy
torch/transformers dependency while still giving good-quality embeddings.
"""
import chromadb
from chromadb.utils import embedding_functions

from app.config import settings


class VectorStore:
    def __init__(self):
        self._client = chromadb.PersistentClient(path=settings.CHROMA_DIR)

        # Lightweight ONNX-based embedding model bundled with ChromaDB.
        # No torch/transformers dependency required.
        self._embedder = embedding_functions.DefaultEmbeddingFunction()

        self._collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            embedding_function=self._embedder,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, ids: list[str], texts: list[str], metadatas: list[dict]):
        """Embed and persist a batch of chunks."""
        self._collection.add(ids=ids, documents=texts, metadatas=metadatas)

    def query(self, query_text: str, top_k: int | None = None) -> dict:
        """Return the top_k most relevant chunks for a query."""
        k = top_k or settings.TOP_K
        return self._collection.query(query_texts=[query_text], n_results=k)

    def count(self) -> int:
        return self._collection.count()

    def reset(self):
        """Delete all vectors in the collection (useful for re-ingesting)."""
        self._client.delete_collection(settings.CHROMA_COLLECTION)
        self._collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            embedding_function=self._embedder,
            metadata={"hnsw:space": "cosine"},
        )


# Singleton instance used across the app
vector_store = VectorStore()

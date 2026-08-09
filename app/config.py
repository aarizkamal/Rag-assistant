"""
Central configuration for the RAG assistant.
All values are read from environment variables (see .env.example),
with sane defaults so the app can run out of the box for local testing.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # LLM provider selection
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "anthropic").lower()

    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Embeddings (local, no API key required)
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Vector store
    CHROMA_DIR: str = os.getenv("CHROMA_DIR", "./data/chroma_db")
    CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "knowledge_base")

    # Chunking strategy
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "120"))

    # Retrieval
    TOP_K: int = int(os.getenv("TOP_K", "4"))


settings = Settings()

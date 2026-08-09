"""
The core RAG logic:
1. Retrieve top-k relevant chunks from the vector store for a user query.
2. Build a grounded prompt that forces the LLM to answer only from context.
3. Call the configured LLM provider (Anthropic or OpenAI) for the final answer.
4. Return the answer along with the sources used, for traceability/citation.

This is intentionally provider-agnostic: swapping LLM_PROVIDER in .env
switches the backend without touching the retrieval logic.
"""
from app.config import settings
from app.vectorstore import vector_store

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY \
the provided context. If the answer is not contained in the context, say clearly \
that you don't have enough information in the knowledge base to answer, rather \
than guessing. Keep answers concise and cite which source(s) you used by filename."""


def _build_prompt(query: str, contexts: list[dict]) -> str:
    context_block = "\n\n".join(
        f"[Source: {c['source']} | chunk {c['chunk_index']}]\n{c['text']}"
        for c in contexts
    )
    return (
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the context above. Cite sources by filename in brackets."
    )


def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    """Query the vector store and return normalized chunk dicts."""
    results = vector_store.query(query, top_k=top_k)
    contexts = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0] if results.get("distances") else [None] * len(docs)

    for text, meta, dist in zip(docs, metas, distances):
        contexts.append(
            {
                "text": text,
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", -1),
                "distance": dist,
            }
        )
    return contexts


def _call_anthropic(prompt: str) -> str:
    import anthropic

    if not settings.ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not set in your environment/.env")

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=settings.ANTHROPIC_MODEL,
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def _call_openai(prompt: str) -> str:
    from openai import OpenAI

    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set in your environment/.env")

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        max_tokens=800,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def generate_answer(prompt: str) -> str:
    if settings.LLM_PROVIDER == "anthropic":
        return _call_anthropic(prompt)
    if settings.LLM_PROVIDER == "openai":
        return _call_openai(prompt)
    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.LLM_PROVIDER}")


def answer_query(query: str, top_k: int | None = None) -> dict:
    """Full RAG flow: retrieve -> prompt -> generate -> return with sources."""
    contexts = retrieve(query, top_k=top_k)

    if not contexts:
        return {
            "answer": "The knowledge base is empty or no relevant documents were found. "
                      "Please ingest some documents first.",
            "sources": [],
        }

    prompt = _build_prompt(query, contexts)
    answer = generate_answer(prompt)

    sources = sorted({c["source"] for c in contexts})
    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": contexts,
    }

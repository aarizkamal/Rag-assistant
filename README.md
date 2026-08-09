# RAG Knowledge Assistant

A retrieval-augmented generation (RAG) system: upload documents, they get
chunked and embedded into a vector store, and you can then ask questions
that get answered strictly from your document content — with cited sources.

Built to match the **Cognizant Ace Frontier Engineer** JD, which explicitly
calls for: *"Design and build RAG pipelines, vector stores, and knowledge
bases"* and *"Deploy and operate LLMs, agents, and AI-native applications in
production environments."*

## Architecture

```
Document (.pdf/.txt/.md)
        │
        ▼
   Chunking (LangChain RecursiveCharacterTextSplitter)
        │
        ▼
   Embedding (local sentence-transformers model — no API key needed)
        │
        ▼
   ChromaDB (persistent vector store)
        │
        ▼
User Query ──► Retrieve top-k chunks ──► Build grounded prompt ──► LLM (Anthropic/OpenAI) ──► Answer + Sources
```

## Project structure

```
rag-assistant/
├── app/
│   ├── config.py       # env-based settings
│   ├── vectorstore.py  # ChromaDB wrapper + embeddings
│   ├── ingest.py        # load, chunk, and store documents
│   ├── rag_chain.py     # retrieval + prompt + LLM call
│   └── main.py          # FastAPI app (ingest/upload/query endpoints)
├── frontend/
│   └── index.html       # simple chat UI, no build step needed
├── data/
│   ├── sample_docs/     # sample .txt to test with immediately
│   └── chroma_db/        # persisted vector store (created at runtime)
├── requirements.txt
└── .env.example
```

## Setup

1. **Create a virtual environment and install dependencies**
   ```bash
   cd rag-assistant
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure your LLM provider**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and set `LLM_PROVIDER` to `anthropic` or `openai`, and paste
   the matching API key. The embedding model runs locally and needs no key.

3. **Run the server**
   ```bash
   uvicorn app.main:app --reload
   ```
   Open **http://localhost:8000** — the chat UI loads automatically.

4. **Ingest the sample docs** (or upload your own via the UI)
   ```bash
   curl -X POST http://localhost:8000/ingest
   ```

5. **Ask a question** — either through the UI, or:
   ```bash
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the Cognizant Ace Team program?"}'
   ```

## API reference

| Method | Endpoint    | Description                                      |
|--------|-------------|---------------------------------------------------|
| GET    | `/health`   | Health check                                        |
| GET    | `/stats`    | Number of chunks currently in the vector store       |
| POST   | `/ingest`   | Ingest every file in `data/sample_docs`               |
| POST   | `/upload`   | Upload + ingest a single file (multipart form)        |
| POST   | `/query`    | `{"query": "...", "top_k": 4}` → answer + sources      |

## Design choices worth mentioning in an interview

- **Local embeddings, swappable LLM**: embedding (retrieval) and generation
  (answering) are decoupled. Retrieval never costs an API call; only the
  final answer step does. This mirrors real cost/latency trade-off decisions
  the JD asks about ("Optimize model, reasoning, latency, cost... trade-offs").
- **Grounded prompting**: the system prompt explicitly instructs the model to
  say "I don't know" rather than hallucinate when context is insufficient —
  a basic but real guardrail.
- **Source citations**: every answer returns which source documents were
  used, giving traceability/auditability (relevant to "Responsible AI" and
  "explainable, auditable AI solutions" in the JD).
- **Deterministic chunk IDs**: re-ingesting the same file doesn't create
  duplicate vectors, which matters for production reliability.

## Natural extensions (good talking points / next steps)

- Add a `/feedback` endpoint and log query→answer→thumbs-up/down for basic
  AI quality monitoring.
- Add response latency + token-cost logging (ties to "AI Evaluation & Ops").
- Swap in a second agent (e.g., a "critic" agent that checks the answer
  against retrieved context before returning it) to demonstrate multi-agent
  coordination — a natural pairing with this project.
- Add hybrid search (keyword + vector) for better recall on exact terms.

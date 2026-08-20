# doc-qa-pgvector

Production-minded document Q&A service built on FastAPI, PostgreSQL 16, and pgvector.

## Architecture

```text
client
  |-- POST /documents --> parser --> 500-token chunks --> embedding provider
  |                                                        |
  |                                                        v
  |                                            PostgreSQL documents/chunks
  |                                              | vector HNSW + tsvector GIN
  |
  `-- POST /query -------> query embedding --> vector search + full text search
                                                   |
                                              RRF top-k contexts
                                                   |
                                           LLM provider + citations
                                                   |
                                              query_log audit row
```

`EmbeddingProvider` and `LLMProvider` are small provider boundaries. Production defaults are
OpenAI `text-embedding-3-small` and Anthropic Claude. Tests use deterministic fake providers and
never call a network API.

## Why HNSW + cosine

HNSW gives low-latency approximate nearest-neighbor search with good recall and supports online
inserts. Cosine distance matches normalized semantic embeddings and ignores vector magnitude.
The HNSW index uses `vector_cosine_ops`; vectors are stored as `vector(1536)`.

## Why reciprocal rank fusion

Vector search catches paraphrases. PostgreSQL full-text search catches exact names, identifiers,
and rare terms. Their raw scores are not comparable, so RRF combines result ranks instead of
pretending both score scales mean the same thing. `rrf_k=60` limits sensitivity to any single ranker.

## Cost and latency trade-offs

- Batch document embeddings in one provider request; larger batches reduce request overhead but
  increase request size and retry cost.
- Increase `top_k` for recall and richer citations; lower it for cheaper, faster LLM calls.
- Raise `SIMILARITY_THRESHOLD` to reduce unsupported answers; lower it to improve recall at the cost
  of more “I don't know” answers being replaced by LLM responses.
- HNSW uses more memory and build time than an exact scan, trading those costs for query latency.

## Quickstart

Requirements: Docker, Python 3.12.

```bash
cp .env.example .env
docker compose up -d db
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest -q
uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs`. Upload text or PDF with `POST /documents`, then query with:

```bash
curl -F "file=@README.md" http://localhost:8000/documents
curl -H "content-type: application/json" -d '{"question":"Why use HNSW?"}' http://localhost:8000/query
```

For real providers, put API keys in local environment variables only:
`OPENAI_API_KEY` and `ANTHROPIC_API_KEY`. `.env.example` intentionally contains no values.

## Quality and CI

```bash
ruff check .
mypy app
pytest -q
```

GitHub Actions runs all three checks against a real `pgvector/pgvector:pg16` service container.


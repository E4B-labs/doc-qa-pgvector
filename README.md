# doc-qa-pgvector

[![CI](https://github.com/E4B-labs/doc-qa-pgvector/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/E4B-labs/doc-qa-pgvector/actions/workflows/ci.yml)

Production-minded document Q&A service built on FastAPI, PostgreSQL 16, and pgvector.

## Contents

- [What this demonstrates](#what-this-demonstrates)
- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [REST API](#rest-api)
- [Configuration](#configuration)
- [Why these choices](#why-these-choices)
- [Cost and latency trade-offs](#cost-and-latency-trade-offs)
- [Checks](#checks)

## What this demonstrates

- FastAPI endpoints for text/PDF ingestion, hybrid retrieval, health checks, and cited answers.
- PostgreSQL `vector(1536)` storage with cosine HNSW search and `tsvector` full-text search.
- Reciprocal Rank Fusion (RRF) for combining semantic and lexical result ranks.
- `EmbeddingProvider` and `LLMProvider` boundaries with OpenAI and Anthropic production defaults.
- Deterministic `FakeEmbeddingProvider` and `FakeLLMProvider` tests with no network calls.
- Upload-size limits, Pydantic validation, LLM timeouts, similarity thresholds, and query audit logs.
- Docker Compose, Python 3.12, Ruff, mypy, pytest-asyncio, and GitHub Actions.

## Architecture

```mermaid
graph TD
    C[Client] --> D[POST /documents]
    D --> P[Parser<br/>text or PDF]
    P --> CH[500-token chunks<br/>50-token overlap]
    CH --> EP[EmbeddingProvider<br/>OpenAI text-embedding-3-small]
    EP --> DB[(PostgreSQL 16<br/>documents + chunks<br/>vector HNSW + tsvector GIN)]

    C --> Q[POST /query]
    Q --> QE[Query embedding]
    QE --> VS[Vector search<br/>pgvector cosine]
    Q --> FS[Full-text search<br/>tsvector]
    VS --> RRF[RRF top-k contexts]
    FS --> RRF
    RRF --> LLM[LLMProvider<br/>Anthropic + citations]
    LLM --> AUDIT[(query_log<br/>audit row)]
```

`EmbeddingProvider` and `LLMProvider` are small provider boundaries. Production defaults are
OpenAI `text-embedding-3-small` and Anthropic Claude. Tests use deterministic fake providers and
never call a network API.

## Quickstart

Requirements: Python 3.12, Docker Desktop or Docker with Compose.

```bash
cp .env.example .env
docker compose up -d db
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest -q
uvicorn app.main:app --reload
```

Docker runs API and PostgreSQL together:

```bash
docker compose up --build
```

Open `http://localhost:8000/docs`. The API accepts text or PDF documents, then answers questions
with source chunk ids, source fragments, hybrid scores, and similarity scores.

## REST API

Upload a document:

```bash
curl -F "file=@README.md" http://localhost:8000/documents
```

Ask a question:

```bash
curl -H "content-type: application/json" \
  -d '{"question":"Why use HNSW?"}' \
  http://localhost:8000/query
```

`GET /health` returns `{"status":"ok"}` when the database connection is healthy.

## Configuration

API keys are read from environment variables only. `.env.example` contains no secret values.

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:15432/docqa` | PostgreSQL connection string |
| `EMBEDDING_PROVIDER` | `openai` | `openai` for production; any other value selects fake provider |
| `OPENAI_API_KEY` | empty | OpenAI API key; required for real embeddings |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `LLM_PROVIDER` | `anthropic` | `anthropic` for production; any other value selects fake provider |
| `ANTHROPIC_API_KEY` | empty | Anthropic API key; required for real answers |
| `ANTHROPIC_MODEL` | `claude-3-5-sonnet-latest` | Anthropic model |
| `MAX_UPLOAD_BYTES` | `5242880` | Maximum upload size, 5 MiB |
| `LLM_TIMEOUT_SECONDS` | `20` | LLM request timeout |
| `SIMILARITY_THRESHOLD` | `0.35` | Below this threshold the service says it does not know |

The Compose database is exposed on host port `15432` to avoid collisions with other local
PostgreSQL containers; the API uses service-internal port `5432`.

## Why these choices

### Why HNSW + cosine

HNSW gives low-latency approximate nearest-neighbor search with good recall and supports online
inserts. Cosine distance matches normalized semantic embeddings and ignores vector magnitude.
The HNSW index uses `vector_cosine_ops`; vectors are stored as `vector(1536)`.

### Why reciprocal rank fusion

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

## Checks

```bash
ruff check app tests
mypy app
pytest -q
```

The test suite contains unit tests for chunking, providers, validation, and RRF plus integration
tests against a real PostgreSQL/pgvector database. SQL similarity search is not mocked.

GitHub Actions runs Ruff, mypy, and all tests against a real `pgvector/pgvector:pg16` service
container.

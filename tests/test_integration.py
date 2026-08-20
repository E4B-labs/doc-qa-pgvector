import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.providers import FakeEmbeddingProvider, FakeLLMProvider
from app.service import DocumentService


class SlowLLM(FakeLLMProvider):
    async def generate(self, question, contexts):
        import asyncio

        await asyncio.sleep(0.05)
        return await super().generate(question, contexts)


@pytest.fixture
def service(database, settings) -> DocumentService:
    return DocumentService(database, FakeEmbeddingProvider(), FakeLLMProvider(), settings)


@pytest.fixture
async def client(database, settings) -> AsyncClient:
    app = create_app(
        settings=settings,
        database=database,
        embedding_provider=FakeEmbeddingProvider(),
        llm_provider=FakeLLMProvider(),
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as test_client:
        yield test_client


@pytest.mark.asyncio
async def test_pgvector_extension_and_hnsw_index(database) -> None:
    async with database.acquire() as connection:
        assert (
            await connection.fetchval("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            == "vector"
        )
        assert await connection.fetchval(
            "SELECT indexdef FROM pg_indexes WHERE indexname = 'chunks_embedding_hnsw_idx'"
        )


@pytest.mark.asyncio
async def test_schema_uses_1536_dimensions(database) -> None:
    async with database.acquire() as connection:
        row = await connection.fetchrow(
            "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
            "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
        )
    assert row[0] == "vector(1536)"


@pytest.mark.asyncio
async def test_ingest_persists_document_and_chunks(service, database) -> None:
    document_id, count = await service.ingest("guide.txt", "text/plain", b"Postgres guide")
    assert count == 1
    async with database.acquire() as connection:
        assert (
            await connection.fetchval("SELECT count(*) FROM documents WHERE id = $1", document_id)
            == 1
        )
        assert (
            await connection.fetchval(
                "SELECT count(*) FROM chunks WHERE document_id = $1", document_id
            )
            == 1
        )


@pytest.mark.asyncio
async def test_vector_search_runs_against_real_pgvector(service) -> None:
    await service.ingest("guide.txt", "text/plain", b"Postgres vector search")
    vector = (await service.embedding_provider.embed(["Postgres vector search"]))[0]
    results = await service.search("Postgres vector search", vector, 3)
    assert results
    assert results[0].similarity > 0.99


@pytest.mark.asyncio
async def test_lexical_search_contributes_to_hybrid_results(service) -> None:
    await service.ingest("guide.txt", "text/plain", b"unusual lexicalneedle database phrase")
    vector = (await service.embedding_provider.embed(["lexicalneedle"]))[0]
    results = await service.search("lexicalneedle", vector, 3)
    assert results
    assert "lexicalneedle" in results[0].fragment


@pytest.mark.asyncio
async def test_health_endpoint_reads_real_database(client) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_document_endpoint_accepts_text_upload(client) -> None:
    response = await client.post(
        "/documents", files={"file": ("guide.txt", b"FastAPI and PostgreSQL", "text/plain")}
    )
    assert response.status_code == 201
    assert response.json()["chunks"] == 1


@pytest.mark.asyncio
async def test_document_endpoint_enforces_upload_limit(client, settings) -> None:
    settings.max_upload_bytes = 2
    response = await client.post(
        "/documents", files={"file": ("guide.txt", b"too large", "text/plain")}
    )
    assert response.status_code == 413


@pytest.mark.asyncio
async def test_query_endpoint_returns_answer_sources_and_scores(client) -> None:
    await client.post(
        "/documents", files={"file": ("guide.txt", b"FastAPI uses async endpoints", "text/plain")}
    )
    response = await client.post(
        "/query", json={"question": "FastAPI uses async endpoints", "top_k": 3}
    )
    body = response.json()
    assert response.status_code == 200
    assert "Sources:" in body["answer"]
    assert body["sources"][0]["chunk_id"]
    assert "similarity" in body["sources"][0]


@pytest.mark.asyncio
async def test_query_without_documents_says_unknown_and_audits(client, database) -> None:
    response = await client.post("/query", json={"question": "missing information"})
    assert response.status_code == 200
    assert response.json()["answer"] == "I don't know based on the indexed documents."
    async with database.acquire() as connection:
        assert await connection.fetchval("SELECT count(*) FROM query_log") == 1


@pytest.mark.asyncio
async def test_query_timeout_returns_504(database, settings) -> None:
    settings.llm_timeout_seconds = 0.001
    app = create_app(
        settings=settings,
        database=database,
        embedding_provider=FakeEmbeddingProvider(),
        llm_provider=SlowLLM(),
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as test_client:
        await test_client.post(
            "/documents", files={"file": ("guide.txt", b"timeout source", "text/plain")}
        )
        response = await test_client.post("/query", json={"question": "timeout source"})
    assert response.status_code == 504


@pytest.mark.asyncio
async def test_query_validation_rejects_empty_question(client) -> None:
    response = await client.post("/query", json={"question": ""})
    assert response.status_code == 422

from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.chunking import chunk_text
from app.config import Settings
from app.db import vector_literal
from app.main import QueryRequest, default_providers
from app.providers import (
    AnthropicLLMProvider,
    FakeEmbeddingProvider,
    FakeLLMProvider,
    OpenAIEmbeddingProvider,
)
from app.service import (
    DocumentService,
    EmptyDocumentError,
    SearchResult,
    UnsupportedDocumentError,
    UploadTooLargeError,
    extract_document_text,
    reciprocal_rank_fusion,
)


def test_chunking_empty_text() -> None:
    assert chunk_text("\n\t") == []


def test_chunking_has_overlap() -> None:
    chunks = chunk_text(" ".join(f"w{i}" for i in range(120)), size=50, overlap=10)
    assert chunks[0].text.split()[-10:] == chunks[1].text.split()[:10]


def test_chunking_rejects_bad_overlap() -> None:
    with pytest.raises(ValueError):
        chunk_text("text", size=5, overlap=5)


def test_chunking_token_limit() -> None:
    assert all(chunk.token_count <= 500 for chunk in chunk_text("x " * 1001))


def test_vector_literal_is_pgvector_syntax() -> None:
    assert vector_literal([1.0, -0.5, 0.25]) == "[1,-0.5,0.25]"


@pytest.mark.asyncio
async def test_fake_embeddings_are_deterministic() -> None:
    provider = FakeEmbeddingProvider()
    first = await provider.embed(["same"])
    second = await provider.embed(["same"])
    assert first == second
    assert len(first[0]) == 1536


@pytest.mark.asyncio
async def test_fake_embeddings_batch_preserves_order() -> None:
    vectors = await FakeEmbeddingProvider().embed(["one", "two"])
    assert vectors[0] != vectors[1]


@pytest.mark.asyncio
async def test_fake_llm_returns_citations() -> None:
    chunk_id = uuid4()
    answer = await FakeLLMProvider().generate("question", [{"chunk_id": str(chunk_id)}])
    assert str(chunk_id) in answer


def test_default_provider_selection() -> None:
    embedding, llm = default_providers(Settings(embedding_provider="fake", llm_provider="fake"))
    assert isinstance(embedding, FakeEmbeddingProvider)
    assert isinstance(llm, FakeLLMProvider)


@pytest.mark.asyncio
async def test_openai_provider_requires_key() -> None:
    with pytest.raises(RuntimeError):
        await OpenAIEmbeddingProvider(None).embed(["question"])


@pytest.mark.asyncio
async def test_anthropic_provider_requires_key() -> None:
    with pytest.raises(RuntimeError):
        await AnthropicLLMProvider(None).generate("question", [])


def test_query_request_validates_top_k() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(question="question", top_k=21)


def test_query_request_rejects_blank_question() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(question=" ")


def test_text_extraction() -> None:
    assert extract_document_text(b"hello", "note.txt", "text/plain") == "hello"


def test_unsupported_document_rejected() -> None:
    with pytest.raises(UnsupportedDocumentError):
        extract_document_text(b"data", "image.png", "image/png")


def test_rrf_prefers_document_found_by_both_rankers() -> None:
    both = SearchResult(uuid4(), uuid4(), "both", 0.8)
    vector_only = SearchResult(uuid4(), uuid4(), "vector", 0.9)
    result = reciprocal_rank_fusion([both, vector_only], [both])
    assert result[0].fragment == "both"


def test_service_upload_limit_fails_before_database_access(settings: Settings) -> None:
    settings.max_upload_bytes = 2
    service = DocumentService(None, FakeEmbeddingProvider(), FakeLLMProvider(), settings)  # type: ignore[arg-type]
    with pytest.raises(UploadTooLargeError):
        import asyncio

        asyncio.run(service.ingest("x.txt", "text/plain", b"123"))


def test_service_rejects_empty_document(settings: Settings) -> None:
    service = DocumentService(None, FakeEmbeddingProvider(), FakeLLMProvider(), settings)  # type: ignore[arg-type]
    with pytest.raises(EmptyDocumentError):
        import asyncio

        asyncio.run(service.ingest("x.txt", "text/plain", b"  "))


from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any
from uuid import UUID

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field, field_validator

from app.config import Settings
from app.db import Database
from app.providers import (
    AnthropicLLMProvider,
    EmbeddingProvider,
    FakeEmbeddingProvider,
    FakeLLMProvider,
    LLMProvider,
    OpenAIEmbeddingProvider,
)
from app.service import (
    DocumentService,
    EmptyDocumentError,
    UnsupportedDocumentError,
    UploadTooLargeError,
)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def question_must_contain_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must contain non-whitespace text")
        return value


class SourceResponse(BaseModel):
    chunk_id: UUID
    document_id: UUID
    fragment: str
    score: float
    similarity: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    top_similarity: float | None


class DocumentResponse(BaseModel):
    document_id: UUID
    chunks: int


def default_providers(settings: Settings) -> tuple[EmbeddingProvider, LLMProvider]:
    if settings.embedding_provider == "openai":
        embedding: EmbeddingProvider = OpenAIEmbeddingProvider(
            settings.openai_api_key, settings.openai_embedding_model
        )
    else:
        embedding = FakeEmbeddingProvider()
    if settings.llm_provider == "anthropic":
        llm: LLMProvider = AnthropicLLMProvider(
            settings.anthropic_api_key, settings.anthropic_model
        )
    else:
        llm = FakeLLMProvider()
    return embedding, llm


def create_app(
    settings: Settings | None = None,
    database: Database | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    llm_provider: LLMProvider | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings()
    resolved_database = database or Database(resolved_settings.database_url)
    default_embedding, default_llm = default_providers(resolved_settings)
    service = DocumentService(
        resolved_database,
        embedding_provider or default_embedding,
        llm_provider or default_llm,
        resolved_settings,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await resolved_database.connect()
        try:
            yield
        finally:
            await resolved_database.close()

    app = FastAPI(title="doc-qa-pgvector", version="0.1.0", lifespan=lifespan)
    app.state.database = resolved_database
    app.state.service = service

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        if not await request.app.state.database.health():
            raise HTTPException(status_code=503, detail="database unavailable")
        return {"status": "ok"}

    @app.post("/documents", response_model=DocumentResponse, status_code=201)
    async def upload_document(
        request: Request,
        file: Annotated[UploadFile, File(...)],
        title: Annotated[str | None, Form(max_length=255)] = None,
    ) -> DocumentResponse:
        service: DocumentService = request.app.state.service
        filename = title or file.filename or "document.txt"
        content_type = file.content_type or "application/octet-stream"
        data = await file.read(service.settings.max_upload_bytes + 1)
        try:
            document_id, chunks = await service.ingest(filename, content_type, data)
        except UploadTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except (UnsupportedDocumentError, EmptyDocumentError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return DocumentResponse(document_id=document_id, chunks=chunks)

    @app.post("/query", response_model=QueryResponse)
    async def query(request: Request, payload: QueryRequest) -> QueryResponse:
        service: DocumentService = request.app.state.service
        try:
            answer, results, top_similarity = await service.query(payload.question, payload.top_k)
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail="LLM request timed out") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return QueryResponse(
            answer=answer,
            sources=[SourceResponse(**result.as_context()) for result in results],
            top_similarity=top_similarity,
        )

    return app


app = create_app()

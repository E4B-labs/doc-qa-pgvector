import asyncio
import io
import time
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

from pypdf import PdfReader

from app.chunking import chunk_text
from app.config import Settings
from app.db import Database, vector_literal
from app.providers import EmbeddingProvider, LLMProvider


class UploadTooLargeError(ValueError):
    pass


class UnsupportedDocumentError(ValueError):
    pass


class EmptyDocumentError(ValueError):
    pass


@dataclass(slots=True)
class SearchResult:
    chunk_id: UUID
    document_id: UUID
    fragment: str
    similarity: float
    rrf_score: float = 0.0

    def as_context(self) -> dict[str, Any]:
        return {
            "chunk_id": str(self.chunk_id),
            "document_id": str(self.document_id),
            "fragment": self.fragment,
            "score": self.rrf_score,
            "similarity": self.similarity,
        }


def reciprocal_rank_fusion(
    vector_rows: list[SearchResult], lexical_rows: list[SearchResult], k: int = 60
) -> list[SearchResult]:
    merged: dict[UUID, SearchResult] = {}
    for rank, row in enumerate(vector_rows, start=1):
        merged[row.chunk_id] = row
        row.rrf_score += 1 / (k + rank)
    for rank, row in enumerate(lexical_rows, start=1):
        existing = merged.get(row.chunk_id)
        if existing is None:
            merged[row.chunk_id] = row
            existing = row
        else:
            existing.similarity = max(existing.similarity, row.similarity)
        existing.rrf_score += 1 / (k + rank)
    return sorted(merged.values(), key=lambda row: (row.rrf_score, row.similarity), reverse=True)


def extract_document_text(data: bytes, filename: str, content_type: str) -> str:
    is_pdf = content_type == "application/pdf" or filename.lower().endswith(".pdf")
    if is_pdf:
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    if content_type.startswith("text/") or filename.lower().endswith(
        (".txt", ".md", ".csv", ".json")
    ):
        return data.decode("utf-8", errors="replace").strip()
    raise UnsupportedDocumentError("only text and PDF documents are supported")


class DocumentService:
    def __init__(
        self,
        database: Database,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
        settings: Settings,
    ) -> None:
        self.database = database
        self.embedding_provider = embedding_provider
        self.llm_provider = llm_provider
        self.settings = settings

    async def ingest(self, filename: str, content_type: str, data: bytes) -> tuple[UUID, int]:
        if len(data) > self.settings.max_upload_bytes:
            raise UploadTooLargeError(f"upload exceeds {self.settings.max_upload_bytes} bytes")
        text = extract_document_text(data, filename, content_type)
        chunks = chunk_text(text)
        if not chunks:
            raise EmptyDocumentError("document contains no text")
        vectors = await self.embedding_provider.embed([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks) or any(
            len(vector) != self.settings.embedding_dimensions for vector in vectors
        ):
            raise ValueError("embedding provider returned invalid dimensions")

        document_id = uuid4()
        async with self.database.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    """
                    INSERT INTO documents (id, filename, content_type, byte_size)
                    VALUES ($1, $2, $3, $4)
                    """,
                    document_id,
                    filename,
                    content_type,
                    len(data),
                )
                await connection.executemany(
                    """
                    INSERT INTO chunks
                        (id, document_id, chunk_index, content, token_count, embedding)
                    VALUES ($1, $2, $3, $4, $5, $6::vector)
                    """,
                    [
                        (
                            uuid4(),
                            document_id,
                            index,
                            chunk.text,
                            chunk.token_count,
                            vector_literal(vector),
                        )
                        for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
                    ],
                )
        return document_id, len(chunks)

    async def _vector_search(
        self, connection: Any, vector: list[float], limit: int
    ) -> list[SearchResult]:
        rows = await connection.fetch(
            """
            SELECT id, document_id, content, 1 - (embedding <=> $1::vector) AS similarity
            FROM chunks
            ORDER BY embedding <=> $1::vector
            LIMIT $2
            """,
            vector_literal(vector),
            limit,
        )
        return [
            SearchResult(row["id"], row["document_id"], row["content"], float(row["similarity"]))
            for row in rows
        ]

    async def _lexical_search(
        self, connection: Any, question: str, limit: int
    ) -> list[SearchResult]:
        rows = await connection.fetch(
            """
            SELECT id, document_id, content,
                   ts_rank_cd(search_vector, plainto_tsquery('simple', $1)) AS lexical_score
            FROM chunks
            WHERE search_vector @@ plainto_tsquery('simple', $1)
            ORDER BY lexical_score DESC
            LIMIT $2
            """,
            question,
            limit,
        )
        return [
            SearchResult(row["id"], row["document_id"], row["content"], 0.0) for row in rows
        ]

    async def search(self, question: str, vector: list[float], top_k: int) -> list[SearchResult]:
        limit = max(top_k * 3, 20)
        async with self.database.acquire() as connection:
            vector_rows = await self._vector_search(connection, vector, limit)
            lexical_rows = await self._lexical_search(connection, question, limit)
        return reciprocal_rank_fusion(vector_rows, lexical_rows, self.settings.rrf_k)[:top_k]

    async def query(
        self, question: str, top_k: int
    ) -> tuple[str, list[SearchResult], float | None]:
        started = time.perf_counter()
        vector = (await self.embedding_provider.embed([question]))[0]
        results = await self.search(question, vector, top_k)
        top_similarity = results[0].similarity if results else None
        if top_similarity is None or top_similarity < self.settings.similarity_threshold:
            answer = "I don't know based on the indexed documents."
        else:
            answer = await asyncio.wait_for(
                self.llm_provider.generate(question, [result.as_context() for result in results]),
                timeout=self.settings.llm_timeout_seconds,
            )
        await self._audit(question, answer, top_similarity, time.perf_counter() - started)
        return answer, results, top_similarity

    async def _audit(
        self, question: str, answer: str, top_similarity: float | None, duration: float
    ) -> None:
        del duration
        async with self.database.acquire() as connection:
            await connection.execute(
                "INSERT INTO query_log (question, answer, top_similarity) VALUES ($1, $2, $3)",
                question,
                answer,
                top_similarity,
            )

import hashlib
import math
from abc import ABC, abstractmethod
from typing import Any, cast

import httpx


class EmbeddingProvider(ABC):
    dimensions = 1536

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, question: str, contexts: list[dict[str, Any]]) -> str:
        raise NotImplementedError


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str | None, model: str = "text-embedding-3-small") -> None:
        self.api_key = api_key
        self.model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI embeddings")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"input": texts, "model": self.model},
            )
            response.raise_for_status()
        data = sorted(response.json()["data"], key=lambda item: item["index"])
        vectors = [item["embedding"] for item in data]
        if any(len(vector) != self.dimensions for vector in vectors):
            raise RuntimeError(f"embedding dimension must be {self.dimensions}")
        return vectors


class AnthropicLLMProvider(LLMProvider):
    def __init__(self, api_key: str | None, model: str = "claude-3-5-sonnet-latest") -> None:
        self.api_key = api_key
        self.model = model

    async def generate(self, question: str, contexts: list[dict[str, Any]]) -> str:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for Anthropic answers")
        source_text = "\n\n".join(
            f"[{context['chunk_id']}] {context['fragment']}" for context in contexts
        )
        prompt = (
            "Answer only from supplied sources. If sources do not support answer, say you do "
            "not know. Cite source chunk ids in brackets.\n\n"
            f"Question: {question}\n\nSources:\n{source_text}"
        )
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": 600,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            response.raise_for_status()
        return cast(str, response.json()["content"][0]["text"])


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic local provider for tests and offline development."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            values = [
                (hashlib.sha256(f"{text}:{index}".encode()).digest()[0] / 127.5) - 1
                for index in range(self.dimensions)
            ]
            norm = math.sqrt(sum(value * value for value in values))
            vectors.append([value / norm for value in values])
        return vectors


class FakeLLMProvider(LLMProvider):
    async def generate(self, question: str, contexts: list[dict[str, Any]]) -> str:
        citations = ", ".join(f"[{context['chunk_id']}]" for context in contexts)
        return f"Answer for: {question} Sources: {citations}"

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast

import asyncpg


def vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(format(value, ".9g") for value in vector) + "]"


class Database:
    def __init__(self, url: str) -> None:
        self.url = url
        self.pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        if self.pool is not None:
            return
        self.pool = await asyncpg.create_pool(self.url, min_size=1, max_size=10)
        migration = Path(__file__).resolve().parents[1] / "migrations" / "001_init.sql"
        async with self.pool.acquire() as connection:
            await connection.execute(migration.read_text(encoding="utf-8"))

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            self.pool = None

    async def health(self) -> bool:
        if self.pool is None:
            return False
        async with self.pool.acquire() as connection:
            return cast(int, await connection.fetchval("SELECT 1")) == 1

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[asyncpg.Connection]:
        if self.pool is None:
            raise RuntimeError("database is not connected")
        async with self.pool.acquire() as connection:
            yield connection

    async def clear(self) -> None:
        async with self.acquire() as connection:
            await connection.execute("TRUNCATE query_log, chunks, documents CASCADE")

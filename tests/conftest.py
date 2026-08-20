import os

import pytest
import pytest_asyncio

from app.config import Settings
from app.db import Database


@pytest_asyncio.fixture
async def database() -> Database:
    url = os.environ.get(
        "TEST_DATABASE_URL",
        os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:15432/docqa"),
    )
    database = Database(url)
    await database.connect()
    await database.clear()
    try:
        yield database
    finally:
        await database.clear()
        await database.close()


@pytest.fixture
def settings() -> Settings:
    return Settings(
        embedding_provider="fake",
        llm_provider="fake",
        similarity_threshold=0.35,
        llm_timeout_seconds=1,
    )

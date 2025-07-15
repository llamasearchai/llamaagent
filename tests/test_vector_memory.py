import os

import pytest

pytestmark = pytest.mark.asyncio


@pytest.fixture()
async def memory():
    os.environ.setdefault(
        "DATABASE_URL", "postgresql://llama:llama@localhost:5432/llamaagent"
    )
    from src.llamaagent.storage.database import DatabaseManager, DatabaseConfig

    try:
        # Create database instance
        config = DatabaseConfig()
        db = DatabaseManager(config)
        await db.initialize()

        # Skip test if database is not available
        if db.pool is None:
            pytest.skip("Database not available")

        from src.llamaagent.storage.vector_memory import VectorMemory as PostgresVectorMemory

        mem = PostgresVectorMemory(agent_id="test")
        yield mem
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Database init failed – {exc}")


aSYNC_PROMPTS = [
    "Hello world",
    "Second entry",
]


async def test_add_and_search(memory):
    for p in aSYNC_PROMPTS:
        await memory.add(p)

    results = await memory.search("Hello", limit=1)
    assert results

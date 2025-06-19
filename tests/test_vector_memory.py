import os
import pytest
import asyncio

pytestmark = pytest.mark.asyncio


@pytest.fixture()
async def memory():
    os.environ.setdefault("DATABASE_URL", "postgresql://llama:llama@localhost:5432/llamaagent")
    from llamaagent.storage.database import Database
    try:
        await Database.initialise()
        # Attempt trivial query to verify connectivity
        try:
            await Database.execute("SELECT 1")
        except Exception as exc:
            pytest.skip(f"Database not available – {exc}")

        from llamaagent.storage.vector_memory import PostgresVectorMemory

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
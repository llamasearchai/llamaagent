from __future__ import annotations

"""Postgres-backed vector memory using `pgvector` extension.

Only the *minimal* subset required by :class:`llamaagent.agents.base.Agent`
(`add` and `search`) is implemented.  The table schema is created lazily on the
first insert, meaning the package can be imported on systems where the
extension is not installed – the first insert will raise a descriptive error
rather than at import-time.
"""

import asyncio
import os
from typing import List

from ..llm import LLMProvider, create_provider
from .database import Database

TABLE_SQL = """
CREATE TABLE IF NOT EXISTS agent_memory (
    id SERIAL PRIMARY KEY,
    agent_id UUID NOT NULL,
    text TEXT NOT NULL,
    embedding vector(1536) NOT NULL,
    created TIMESTAMP WITH TIME ZONE DEFAULT now()
);
"""

INSERT_SQL = """
INSERT INTO agent_memory (agent_id, text, embedding)
VALUES ($1, $2, $3)
"""

SEARCH_SQL = """
SELECT text, (embedding <=> $2) AS distance
FROM agent_memory
WHERE agent_id = $1
ORDER BY embedding <=> $2
LIMIT $3
"""


class PostgresVectorMemory:
    """Vector memory implementation backed by Postgres/pgvector."""

    def __init__(self, agent_id: str, provider: LLMProvider | None = None) -> None:
        self.agent_id = agent_id
        self.llm = provider or create_provider(os.getenv("LLAMAAGENT_LLM_PROVIDER", "mock"))
        self._schema_ready = False
        self._schema_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    async def add(self, text: str) -> None:  # noqa: D401
        await self._ensure_schema()
        vector = await self._embed(text)
        await Database.execute(INSERT_SQL, self.agent_id, text, vector)

    async def search(self, query: str, limit: int = 5) -> List[str]:  # noqa: D401
        await self._ensure_schema()
        q_emb = await self._embed(query)
        rows = await Database.fetch(SEARCH_SQL, self.agent_id, q_emb, limit)
        return [r["text"] for r in rows]

    # ------------------------------------------------------------------
    async def _ensure_schema(self) -> None:  # noqa: D401
        if self._schema_ready:
            return
        async with self._schema_lock:
            if not self._schema_ready:
                await Database.execute(TABLE_SQL)
                self._schema_ready = True

    async def _embed(self, text: str) -> List[float]:  # noqa: D401
        """Generate embedding using the underlying LLM provider (sync wrapper)."""

        response = await self.llm.embed(text)  # type: ignore[attr-defined]
        vector_raw = response.embedding if hasattr(response, "embedding") else response
        # Ensure JSON-serialisable list of floats (asyncpg maps to PostgreSQL vector)
        return list(map(float, vector_raw))

from __future__ import annotations

"""Asynchronous Postgres connection management for LlamaAgent.

This wrapper is intentionally minimal and avoids heavyweight ORMs.  It uses
``asyncpg`` directly for maximum performance and minimal overhead.  The class
exposes a single global connection pool instance which can be queried from any
part of the codebase.

Usage::

    from llamaagent.storage.database import Database
    rows = await Database.fetch("SELECT 1")

If the ``DATABASE_URL`` environment variable is *not* defined the module falls
back to an in-memory fake implementation so that the wider codebase can
transparently import the class without requiring Postgres in local dev or CI.
"""

import asyncio
import os
from typing import Any, Optional

try:
    import asyncpg  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – optional dependency
    asyncpg = None  # type: ignore

__all__ = ["Database"]


class _NullPool:  # pylint: disable=too-few-public-methods
    """Fallback pool that raises on query – used when DB is unavailable."""

    async def execute(self, *_: Any, **__: Any) -> None:  # noqa: D401, ANN001
        raise RuntimeError("Database support is not enabled (asyncpg missing or DATABASE_URL unset)")

    fetch = execute  # type: ignore[assignment]
    fetchrow = execute  # type: ignore[assignment]
    fetchval = execute  # type: ignore[assignment]


class Database:  # pylint: disable=too-few-public-methods
    """Global async connection pool wrapper (singleton-like).

    The first call to :pymeth:`initialise` creates the pool, subsequent calls
    are no-ops.  Convenience thin-wrappers around common ``asyncpg`` methods
    are provided.  For advanced usage, call :pyattr:`pool` directly.
    """

    _pool: Optional["asyncpg.Pool"] = None  # type: ignore[type-arg]
    _lock: asyncio.Lock = asyncio.Lock()

    @classmethod
    async def initialise(cls) -> None:  # noqa: D401
        """Initialise the global connection pool (idempotent)."""

        if cls._pool is not None or asyncpg is None:
            return

        async with cls._lock:
            if cls._pool is None:  # pragma: no branch – re-check inside lock
                dsn = os.getenv("DATABASE_URL")
                if not dsn:
                    # Leave pool as None so that accessing ``pool`` property
                    # yields _NullPool.
                    return
                cls._pool = await asyncpg.create_pool(dsn, min_size=1, max_size=10)

    # ---------------------------------------------------------------------
    # Delegated convenience wrappers
    # ---------------------------------------------------------------------
    @classmethod
    def _get_pool(cls):  # noqa: D401, ANN001
        """Return active asyncpg pool or fallback dummy."""

        return cls._pool or _NullPool()

    @classmethod
    async def execute(cls, query: str, *args: Any) -> Any:  # noqa: D401, ANN401
        await cls.initialise()
        return await cls._get_pool().execute(query, *args)

    @classmethod
    async def fetch(cls, query: str, *args: Any) -> Any:  # noqa: ANN401
        await cls.initialise()
        return await cls._get_pool().fetch(query, *args)

    @classmethod
    async def fetchrow(cls, query: str, *args: Any) -> Any:  # noqa: ANN401
        await cls.initialise()
        return await cls._get_pool().fetchrow(query, *args)

    @classmethod
    async def fetchval(cls, query: str, *args: Any) -> Any:  # noqa: ANN401
        await cls.initialise()
        return await cls._get_pool().fetchval(query, *args)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property  # type: ignore[override] – classproperty pattern
    def pool(self) -> "asyncpg.Pool | _NullPool":  # type: ignore[name-defined]
        """Return active pool or dummy instance."""

        return self._pool or _NullPool()

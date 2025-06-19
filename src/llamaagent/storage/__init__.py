from __future__ import annotations

"""Database and vector-store support for LlamaAgent.

This sub-package provides optional but production-grade persistence layers for
agent traces, short-term memories, and vector search.  The default in-memory
implementation (`SimpleMemory`) remains available for offline or ephemeral
workloads – applications can opt into the Postgres-backed variant simply by
setting the ``DATABASE_URL`` environment variable (any DSN supported by
``asyncpg``).  Tests are skipped automatically if no database is reachable so
that CI pipelines remain self-contained.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from .database import Database  # noqa: F401
from .vector_memory import PostgresVectorMemory  # noqa: F401

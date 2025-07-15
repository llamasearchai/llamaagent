"""Storage module for LlamaAgent."""

from __future__ import annotations

# Import database components
try:
    from .database import Database, DatabaseConfig
except ImportError:
    Database = None  # type: ignore
    DatabaseConfig = None  # type: ignore

# Import vector memory components
try:
    from .vector_memory import PostgresVectorMemory, VectorMemory
except ImportError:
    VectorMemory = None  # type: ignore
    PostgresVectorMemory = None  # type: ignore

# Public API
__all__ = [
    "Database",
    "DatabaseConfig",
    "VectorMemory",
    "PostgresVectorMemory",
]

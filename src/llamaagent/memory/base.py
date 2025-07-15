"""Base memory classes for LlamaAgent."""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional


class BaseMemory(abc.ABC):
    """Abstract base class for memory back-ends used by LlamaAgent."""

    @abc.abstractmethod
    async def add(
        self, content: str, tags: Optional[List[str]] = None, **metadata: Any
    ) -> str:
        """Add content to memory.

        Args:
            content: The content to store
            tags: Optional tags for categorization
            **metadata: Additional metadata

        Returns:
            The ID of the stored memory
        """
        ...

    @abc.abstractmethod
    async def search(
        self, query: str, limit: int = 5, tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search memories by content.

        Args:
            query: The search query
            limit: Maximum number of results
            tags: Optional tags to filter by

        Returns:
            List of matching memory entries
        """
        ...

    @abc.abstractmethod
    async def get_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent memories.

        Args:
            limit: Maximum number of results

        Returns:
            List of recent memory entries
        """
        ...

    @abc.abstractmethod
    async def clear(self) -> None:
        """Clear all memories."""
        ...

    @abc.abstractmethod
    def count(self) -> int:
        """Count total memories.

        Returns:
            Number of stored memories
        """
        ...


class SimpleMemory(BaseMemory):
    """Simple in-memory storage implementation for development and testing."""

    def __init__(self) -> None:
        self._memories: List[Dict[str, Any]] = []
        self._id_counter = 0

    async def add(
        self, content: str, tags: Optional[List[str]] = None, **metadata: Any
    ) -> str:
        """Add content to memory."""
        self._id_counter += 1
        memory_id = str(self._id_counter)

        memory_entry = {
            "id": memory_id,
            "content": content,
            "tags": tags or [],
            **metadata,
        }
        self._memories.append(memory_entry)

        return memory_id

    async def search(
        self, query: str, limit: int = 5, tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search memories by content."""
        results: List[Dict[str, Any]] = []
        query_lower = query.lower()

        for memory in self._memories:
            # Check if query matches content
            if query_lower in memory["content"].lower():
                # Check if tags match (if provided)
                if not tags or any(tag in memory.get("tags", []) for tag in tags):
                    results.append(memory)
                    if len(results) >= limit:
                        break

        return results

    async def get_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent memories."""
        # Return the last 'limit' memories (most recent)
        return list(self._memories[-limit:]) if self._memories else []

    async def clear(self) -> None:
        """Clear all memories."""
        self._memories.clear()
        self._id_counter = 0

    def count(self) -> int:
        """Count total memories."""
        return len(self._memories)

    @property
    def messages(self) -> List[Dict[str, Any]]:
        """Backward compatibility property for tests."""
        return self._memories

"""Memory system for LlamaAgent."""

import hashlib
import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MemoryEntry:
    content: str
    timestamp: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    metadata: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Generate a unique ID for the entry."""
        return hashlib.md5(f"{self.content}{self.timestamp}".encode()).hexdigest()


class SimpleMemory:
    """Simple in-memory storage for agent memories."""

    def __init__(self, max_entries: int = 1000):
        self._entries: List[MemoryEntry] = []
        self.max_entries = max_entries

    async def add(self, content: str, tags: Optional[List[str]] = None, **metadata) -> str:
        """Add a memory entry."""
        entry = MemoryEntry(content=content, tags=tags or [], metadata=metadata)

        self._entries.append(entry)

        # Maintain max size
        if len(self._entries) > self.max_entries:
            self._entries.pop(0)

        return entry.id

    async def search(self, query: str, limit: int = 5, tags: Optional[List[str]] = None) -> List[MemoryEntry]:
        """Search for relevant memories."""
        results = []

        # Simple keyword-based search
        query_lower = query.lower()

        for entry in self._entries:
            score = 0.0

            # Content matching
            if query_lower in entry.content.lower():
                score += 1.0

            # Keyword overlap
            query_words = set(query_lower.split())
            content_words = set(entry.content.lower().split())
            overlap = len(query_words & content_words)
            if overlap > 0:
                score += overlap / len(query_words)

            # Tag matching
            if tags:
                tag_overlap = len(set(tags) & set(entry.tags))
                if tag_overlap > 0:
                    score += tag_overlap / len(tags)

            if score > 0:
                results.append((score, entry))

        # Sort by score and return top results
        results.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in results[:limit]]

    async def get_recent(self, limit: int = 10) -> List[MemoryEntry]:
        """Get the most recent memories."""
        return sorted(self._entries, key=lambda x: x.timestamp, reverse=True)[:limit]

    async def clear(self) -> None:
        """Clear all memories."""
        self._entries.clear()

    def count(self) -> int:
        """Get the number of stored memories."""
        return len(self._entries)

    def get_stats(self) -> dict:
        """Get memory statistics."""
        if not self._entries:
            return {"count": 0, "oldest": None, "newest": None}

        timestamps = [entry.timestamp for entry in self._entries]
        return {
            "count": len(self._entries),
            "oldest": min(timestamps),
            "newest": max(timestamps),
            "avg_content_length": sum(len(entry.content) for entry in self._entries) / len(self._entries),
        }


__all__ = [
    "MemoryEntry",
    "SimpleMemory",
]

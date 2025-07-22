"""
Memory pool implementation.
"""

from typing import Any, Optional


class MemoryPool:
    """Memory pool for caching."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.pool = {}

    def get(self, key: str) -> Optional[Any]:
        """Get item from pool."""
        return self.pool.get(key)

    def set(self, key: str, value: Any) -> None:
        """Set item in pool."""
        if len(self.pool) >= self.max_size:
            # Remove oldest item
            oldest_key = next(iter(self.pool))
            del self.pool[oldest_key]
        self.pool[key] = value

    def clear(self) -> None:
        """Clear the pool."""
        self.pool.clear()

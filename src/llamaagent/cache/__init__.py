"""Cache module for LlamaAgent."""

from typing import Any, Dict, Optional


class ResultCache:
    """Lightweight in-memory result cache used in tests."""

    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    def get(self, key: str) -> Optional[Any]:
        return self._store.get(key)

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value


class CacheManager:
    """Basic cache manager."""

    def __init__(self):
        self.cache = {}

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return self.cache.get(key)

    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        self.cache[key] = value


from .advanced_cache import AdvancedCache, CacheStrategy  # re-export for tests

__all__ = ['CacheManager', 'ResultCache', 'AdvancedCache', 'CacheStrategy']

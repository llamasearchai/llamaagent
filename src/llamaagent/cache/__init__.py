"""Cache module for LlamaAgent."""

from typing import Any, Dict, Optional


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

__all__ = ['CacheManager']

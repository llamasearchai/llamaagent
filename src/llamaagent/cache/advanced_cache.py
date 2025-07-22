"""
Advanced cache implementation with multiple strategies and features.
"""

import asyncio
import logging
import pickle
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import numpy as np
except ImportError:
    np = None


class CacheStrategy(Enum):
    """Cache eviction strategies."""

    LRU = "lru"
    LFU = "lfu"
    ARC = "arc"
    HYBRID = "hybrid"
    PREDICTIVE = "predictive"
    SEMANTIC = "semantic"


@dataclass
class CacheEntry:
    """Enhanced cache entry with metadata."""

    key: str
    value: Any
    creation_time: float = field(default_factory=time.time)
    last_access_time: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[float] = None
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.creation_time > self.ttl

    def access(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_access_time = time.time()


@dataclass
class CacheStats:
    """Cache performance statistics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    semantic_dedup_count: int = 0
    total_access_time: float = 0.0
    memory_usage_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def avg_access_time(self) -> float:
        """Calculate average access time."""
        total = self.hits + self.misses
        return self.total_access_time / total if total > 0 else 0.0


class AdvancedCache:
    """
    Advanced caching system with multiple strategies and features.
    """

    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: float = 100.0,
        strategy: CacheStrategy = CacheStrategy.LRU,
        similarity_threshold: float = 0.8,
        persistence_path: Optional[str] = None,
    ):
        """Initialize advanced cache."""
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.strategy = strategy
        self.similarity_threshold = similarity_threshold
        self.persistence_path = persistence_path

        # Core cache storage
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = asyncio.Lock()

        # Strategy-specific structures
        self.frequency_map: Dict[str, int] = defaultdict(int)

        # ARC algorithm structures
        self.arc_t1: OrderedDict[str, CacheEntry] = OrderedDict()
        self.arc_b1: OrderedDict[str, None] = OrderedDict()
        self.arc_t2: OrderedDict[str, CacheEntry] = OrderedDict()
        self.arc_b2: OrderedDict[str, None] = OrderedDict()
        self.arc_p = 0  # ARC adaptation parameter

        # Statistics and patterns
        self.stats = CacheStats()
        self.access_patterns: List[Tuple[str, float]] = []
        self.embeddings_index: Dict[str, List[float]] = {}

        # Load persisted cache if available
        if self.persistence_path and Path(self.persistence_path).exists():
            self._load_from_disk()

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self.lock:
            start_time = time.time()

            # Check if key exists and not expired
            if key in self.cache:
                entry = self.cache[key]
                if not entry.is_expired():
                    entry.access()
                    self._record_hit(key)
                    self.stats.total_access_time += time.time() - start_time
                    return entry.value
                else:
                    # Remove expired entry
                    del self.cache[key]

            # Try semantic matching if enabled
            if self.strategy == CacheStrategy.SEMANTIC:
                semantic_key = await self._find_semantic_match(key)
                if semantic_key and semantic_key in self.cache:
                    entry = self.cache[semantic_key]
                    if not entry.is_expired():
                        entry.access()
                        self._record_hit(semantic_key)
                        self.stats.semantic_dedup_count += 1
                        self.stats.total_access_time += time.time() - start_time
                        return entry.value

            # Cache miss
            self.stats.misses += 1
            self.stats.total_access_time += time.time() - start_time
            return None

    async def put(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Put value in cache with optional TTL and metadata."""
        async with self.lock:
            # Calculate size
            size_bytes = self._estimate_size(value)

            # Check if we need to evict
            while self._needs_eviction(size_bytes):
                await self._evict_by_strategy()

            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl,
                size_bytes=size_bytes,
                metadata=metadata or {},
            )

            # Store entry
            self.cache[key] = entry
            self.stats.memory_usage_bytes += size_bytes

            # Update strategy-specific structures
            self._update_strategy_structures(key)

            # Store embedding for semantic matching
            if self.strategy == CacheStrategy.SEMANTIC:
                self.embeddings_index[key] = self._compute_embedding(key)

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        async with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                self.stats.memory_usage_bytes -= entry.size_bytes
                del self.cache[key]

                # Clean up strategy structures
                if key in self.frequency_map:
                    del self.frequency_map[key]
                if key in self.embeddings_index:
                    del self.embeddings_index[key]

                return True
            return False

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self.lock:
            self.cache.clear()
            self.frequency_map.clear()
            self.embeddings_index.clear()
            self.arc_t1.clear()
            self.arc_b1.clear()
            self.arc_t2.clear()
            self.arc_b2.clear()
            self.stats = CacheStats()

    async def size(self) -> int:
        """Get number of entries in cache."""
        return len(self.cache)

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            "hit_rate": self.stats.hit_rate,
            "avg_access_time_ms": self.stats.avg_access_time * 1000,
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "evictions": self.stats.evictions,
            "semantic_dedup_count": self.stats.semantic_dedup_count,
            "memory_usage_mb": self.stats.memory_usage_bytes / (1024 * 1024),
            "cache_size": len(self.cache),
            "current_strategy": self.strategy.value,
        }

    async def warm_cache(
        self, warm_data: List[Tuple[str, Any, Optional[Dict]]]
    ) -> None:
        """Warm cache with precomputed values."""
        for key, value, metadata in warm_data:
            await self.put(key, value, metadata=metadata)

    def persist_to_disk(self) -> None:
        """Persist cache to disk."""
        if not self.persistence_path:
            return

        cache_data = {
            "entries": {
                k: {
                    "value": v.value,
                    "creation_time": v.creation_time,
                    "access_count": v.access_count,
                    "ttl": v.ttl,
                    "metadata": v.metadata,
                }
                for k, v in self.cache.items()
            },
            "stats": {
                "hits": self.stats.hits,
                "misses": self.stats.misses,
                "evictions": self.stats.evictions,
            },
            "strategy": self.strategy.value,
        }

        with open(self.persistence_path, "wb") as f:
            pickle.dump(cache_data, f)

    def _load_from_disk(self) -> None:
        """Load cache from disk."""
        try:
            with open(self.persistence_path, "rb") as f:
                cache_data = pickle.load(f)

            # Restore entries
            for key, entry_data in cache_data.get("entries", {}).items():
                entry = CacheEntry(
                    key=key,
                    value=entry_data["value"],
                    creation_time=entry_data["creation_time"],
                    access_count=entry_data["access_count"],
                    ttl=entry_data.get("ttl"),
                    metadata=entry_data.get("metadata", {}),
                )
                self.cache[key] = entry

            # Restore stats
            stats = cache_data.get("stats", {})
            self.stats.hits = stats.get("hits", 0)
            self.stats.misses = stats.get("misses", 0)
            self.stats.evictions = stats.get("evictions", 0)
        except Exception as e:
            logger.error(f"Error: {e}")  # Ignore load errors

    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value."""
        try:
            return len(pickle.dumps(value))
        except Exception:
            return len(str(value).encode())

    def _needs_eviction(self, new_size: int) -> bool:
        """Check if eviction is needed."""
        return (
            len(self.cache) >= self.max_size
            or self.stats.memory_usage_bytes + new_size > self.max_memory_bytes
        )

    async def _evict_by_strategy(self) -> None:
        """Evict entry based on current strategy."""
        if not self.cache:
            return

        if self.strategy == CacheStrategy.LRU:
            key, entry = self.cache.popitem(last=False)
        elif self.strategy == CacheStrategy.LFU:
            # Find key with lowest frequency
            min_key = min(
                self.frequency_map.keys(), key=lambda k: self.frequency_map[k]
            )
            entry = self.cache.pop(min_key)
            del self.frequency_map[min_key]
        else:
            # Default to LRU
            key, entry = self.cache.popitem(last=False)

        self.stats.memory_usage_bytes -= entry.size_bytes
        self.stats.evictions += 1

    def _record_hit(self, key: str) -> None:
        """Record cache hit and update patterns."""
        self.stats.hits += 1
        self.access_patterns.append(key, time.time())

        # Keep pattern history manageable
        if len(self.access_patterns) > 10000:
            self.access_patterns = self.access_patterns[-5000:]

    def _update_strategy_structures(self, key: str) -> None:
        """Update strategy-specific data structures."""
        if self.strategy == CacheStrategy.LRU:
            self.cache.move_to_end(key)
        elif self.strategy == CacheStrategy.LFU:
            self.frequency_map[key] += 1

    async def _find_semantic_match(self, key: str) -> Optional[str]:
        """Find semantically similar cached key."""
        if not self.embeddings_index:
            return None

        query_embedding = self._compute_embedding(key)
        best_match = None
        best_similarity = 0.0

        for cached_key, cached_embedding in self.embeddings_index.items():
            similarity = self._cosine_similarity(query_embedding, cached_embedding)
            if similarity > self.similarity_threshold and similarity > best_similarity:
                best_match = cached_key
                best_similarity = similarity

        return best_match

    def _compute_embedding(self, text: str) -> List[float]:
        """Compute simple embedding for text."""
        # Simple character-based embedding
        chars = [ord(c) for c in text[:50]]  # Take first 50 chars
        while len(chars) < 50:
            chars.append(0)  # Pad with zeros
        return [float(c) / 128.0 for c in chars]  # Normalize to [0, 2]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

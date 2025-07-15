"""
Advanced Cache Manager for LlamaAgent

Provides hierarchical caching with L1 (memory), L2 (disk), and L3 (Redis) support.
Includes intelligent eviction policies, compression, and performance monitoring.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import redis.asyncio as redis

    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False
    redis = None

from . import CacheBackend
from . import CacheManager as BaseCacheManager

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Cache statistics tracking"""

    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    compression_ratio: float = 1.0
    hot_keys: List[str] = field(default_factory=list)
    cold_keys: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        total_requests = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "deletes": self.deletes,
            "evictions": self.evictions,
            "hit_rate": self.hits / total_requests if total_requests > 0 else 0,
            "total_requests": total_requests,
            "compression_ratio": self.compression_ratio,
            "hot_keys_count": len(self.hot_keys),
            "cold_keys_count": len(self.cold_keys),
        }


@dataclass
class CacheConfig:
    """Configuration for cache system"""

    # Backend settings
    primary_backend: CacheBackend = CacheBackend.MEMORY
    secondary_backend: Optional[CacheBackend] = None

    # Size limits
    memory_cache_size: int = 10000
    memory_cache_mb: int = 1024  # 1GB
    disk_cache_mb: int = 10240  # 10GB

    # TTL settings
    default_ttl: float = 3600  # 1 hour
    max_ttl: float = 86400  # 24 hours

    # Redis settings
    redis_url: Optional[str] = None
    redis_prefix: str = "llamaagent:"
    redis_pool_size: int = 10

    # Performance settings
    enable_compression: bool = True
    compression_threshold: int = 1024  # Compress if larger than 1KB
    enable_statistics: bool = True
    enable_warming: bool = True
    warming_interval: float = 300  # 5 minutes

    # Eviction settings
    eviction_policy: str = "lru"  # lru, lfu, arc
    eviction_batch_size: int = 100
    cold_threshold: int = 10  # accesses below this are considered cold


class HierarchicalCache:
    """Multi-level cache with L1 (memory), L2 (disk), and L3 (Redis) support"""

    def __init__(self, config: CacheConfig) -> None:
        self.config = config
        self.stats = CacheStats()
        self.access_counts: Dict[str, int] = defaultdict(int)
        self.last_access: Dict[str, float] = {}
        self._running = False
        self._tasks: List[asyncio.Task[Any]] = []

        # Initialize cache levels
        self.l1_cache = BaseCacheManager(backend=CacheBackend.MEMORY)

        self.l2_cache = None
        if config.secondary_backend:
            self.l2_cache = BaseCacheManager(backend=config.secondary_backend)

        # Initialize Redis if available
        self.redis_client = None
        if config.redis_url and _REDIS_AVAILABLE and redis is not None:
            try:
                self.redis_client = redis.from_url(
                    config.redis_url,
                    decode_responses=True,
                    max_connections=config.redis_pool_size,
                )
                logger.info("Redis cache initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis: {e}")
                self.config.redis_url = None

        # Start background tasks
        if config.enable_warming or config.enable_statistics:
            self._start_background_tasks()

    def _start_background_tasks(self) -> None:
        """Start background tasks for cache maintenance"""
        self._running = True

        if self.config.enable_warming:
            task = asyncio.create_task(self._warming_task())
            self._tasks.append(task)

        if self.config.enable_statistics:
            task = asyncio.create_task(self._stats_task())
            self._tasks.append(task)

        # Eviction task
        task = asyncio.create_task(self._eviction_task())
        self._tasks.append(task)

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with L1 -> L2 -> L3 fallback"""
        time.time()

        # Update access tracking
        self.access_counts[key] += 1
        self.last_access[key] = time.time()

        # Try L1 cache first
        value = await self.l1_cache.get(key)
        if value is not None:
            self.stats.hits += 1
            return self._maybe_decompress(value)

        # Try L2 cache
        if self.l2_cache:
            value = await self.l2_cache.get(key)
            if value is not None:
                self.stats.hits += 1
                # Promote to L1
                await self.l1_cache.set(key, value, ttl=self.config.default_ttl)
                return self._maybe_decompress(value)

        # Try L3 (Redis)
        if self.redis_client:
            value = await self._get_from_redis(key)
            if value is not None:
                self.stats.hits += 1
                # Promote to L1 and L2
                await self.l1_cache.set(key, value, ttl=self.config.default_ttl)
                if self.l2_cache:
                    await self.l2_cache.set(key, value, ttl=self.config.default_ttl)
                return self._maybe_decompress(value)

        # Cache miss
        self.stats.misses += 1
        return None

    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache across all levels"""
        if ttl is None:
            ttl = self.config.default_ttl

        ttl = min(ttl, self.config.max_ttl)

        # Compress if needed
        compressed_value = self._maybe_compress(value)

        # Set in all cache levels
        await self.l1_cache.set(key, compressed_value, ttl=ttl)

        if self.l2_cache:
            await self.l2_cache.set(key, compressed_value, ttl=ttl)

        if self.redis_client:
            await self._set_in_redis(key, compressed_value, ttl)

        # Update stats
        self.stats.sets += 1
        self.access_counts[key] += 1
        self.last_access[key] = time.time()

    async def delete(self, key: str) -> bool:
        """Delete key from all cache levels"""
        results = []

        # Delete from all levels
        results.append(await self.l1_cache.delete(key))

        if self.l2_cache:
            results.append(await self.l2_cache.delete(key))

        if self.redis_client:
            results.append(await self._delete_from_redis(key))

        # Clean up tracking
        self.access_counts.pop(key, None)
        self.last_access.pop(key, None)

        if any(results):
            self.stats.deletes += 1
            return True

        return False

    async def clear(self) -> None:
        """Clear all cache levels"""
        await self.l1_cache.clear()

        if self.l2_cache:
            await self.l2_cache.clear()

        if self.redis_client:
            await self._clear_redis()

        self.access_counts.clear()
        self.last_access.clear()

    async def exists(self, key: str) -> bool:
        """Check if key exists in any cache level"""
        if await self.l1_cache.exists(key):
            return True

        if self.l2_cache and await self.l2_cache.exists(key):
            return True

        if self.redis_client:
            return await self._exists_in_redis(key)

        return False

    async def warm_cache(self, keys: List[str]) -> None:
        """Warm cache with frequently accessed keys"""
        if not keys:
            return

        for key in keys:
            if not await self.l1_cache.exists(key):
                # Try to promote from L2 or L3
                value = None

                if self.l2_cache:
                    value = await self.l2_cache.get(key)

                if value is None and self.redis_client:
                    value = await self._get_from_redis(key)

                if value is not None:
                    await self.l1_cache.set(key, value, ttl=self.config.default_ttl)

    async def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze cache access patterns"""
        total_accesses = sum(self.access_counts.values())

        # Find hot keys (top 10%)
        sorted_keys = sorted(
            self.access_counts.items(), key=lambda x: x[1], reverse=True
        )

        hot_threshold = max(1, total_accesses * 0.1)
        hot_keys = [key for key, count in sorted_keys if count >= hot_threshold]

        # Find cold keys (bottom 10%)
        cold_keys = [
            key for key, count in sorted_keys if count <= self.config.cold_threshold
        ]

        # Update stats
        self.stats.hot_keys = hot_keys[:50]  # Keep top 50
        self.stats.cold_keys = cold_keys[:100]  # Keep bottom 100

        return {
            "total_accesses": total_accesses,
            "hot_keys": hot_keys[:10],  # Top 10
            "cold_keys": cold_keys[:10],  # Bottom 10
            "hot_keys_count": len(hot_keys),
            "cold_keys_count": len(cold_keys),
            "compression_ratio": self.stats.compression_ratio,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        base_stats = self.stats.to_dict()

        # Add level-specific stats
        base_stats["l1_stats"] = self.l1_cache.get_stats()

        if self.l2_cache:
            base_stats["l2_stats"] = self.l2_cache.get_stats()

        if self.redis_client:
            base_stats["l3_enabled"] = True

        return base_stats

    async def shutdown(self) -> None:
        """Shutdown cache and cleanup resources"""
        self._running = False

        # Cancel background tasks
        for task in self._tasks:
            task.cancel()

        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()

    # Redis operations
    async def _get_from_redis(self, key: str) -> Optional[Any]:
        """Get value from Redis"""
        try:
            redis_key = f"{self.config.redis_prefix}{key}"
            value = await self.redis_client.get(redis_key)
            return pickle.loads(value.encode()) if value else None
        except Exception as e:
            logger.warning(f"Redis get error: {e}")
            return None

    async def _set_in_redis(self, key: str, value: Any, ttl: float) -> None:
        """Set value in Redis"""
        try:
            redis_key = f"{self.config.redis_prefix}{key}"
            serialized = pickle.dumps(value).decode()
            await self.redis_client.setex(redis_key, int(ttl), serialized)
        except Exception as e:
            logger.warning(f"Redis set error: {e}")

    async def _delete_from_redis(self, key: str) -> bool:
        """Delete key from Redis"""
        try:
            redis_key = f"{self.config.redis_prefix}{key}"
            result = await self.redis_client.delete(redis_key)
            return result > 0
        except Exception as e:
            logger.warning(f"Redis delete error: {e}")
            return False

    async def _exists_in_redis(self, key: str) -> bool:
        """Check if key exists in Redis"""
        try:
            redis_key = f"{self.config.redis_prefix}{key}"
            return await self.redis_client.exists(redis_key)
        except Exception as e:
            logger.warning(f"Redis exists error: {e}")
            return False

    async def _clear_redis(self) -> None:
        """Clear Redis cache"""
        try:
            pattern = f"{self.config.redis_prefix}*"
            cursor = 0

            while True:
                cursor, keys = await self.redis_client.scan(
                    cursor, match=pattern, count=100
                )

                if keys:
                    await self.redis_client.delete(*keys)

                if cursor == 0:
                    break
        except Exception as e:
            logger.warning(f"Redis clear error: {e}")

    # Utility methods
    def _maybe_compress(self, value: Any) -> Any:
        """Compress value if it's large enough"""
        if not self.config.enable_compression:
            return value

        try:
            serialized = pickle.dumps(value)

            if len(serialized) > self.config.compression_threshold:
                compressed = gzip.compress(serialized)

                if len(compressed) < len(serialized):
                    self.stats.compression_ratio = len(compressed) / len(serialized)
                    return {"_compressed": True, "data": compressed.hex()}
        except Exception as e:
            logger.error(f"Error: {e}")

        return value

    def _maybe_decompress(self, value: Any) -> Any:
        """Decompress value if it was compressed"""
        if isinstance(value, dict) and value.get("_compressed"):
            try:
                compressed = bytes.fromhex(value["data"])
                serialized = gzip.decompress(compressed)
                return pickle.loads(serialized)
            except Exception as e:
                logger.error(f"Error: {e}")

        return value

    # Background tasks
    async def _warming_task(self) -> None:
        """Periodically warm cache with hot keys"""
        while self._running:
            try:
                await asyncio.sleep(self.config.warming_interval)

                # Get hot keys that aren't in L1
                hot_keys = [
                    key
                    for key, count in self.access_counts.items()
                    if count in self.stats.hot_keys
                    and not await self.l1_cache.exists(key)
                ]

                if hot_keys:
                    await self.warm_cache(hot_keys[:50])  # Warm top 50

            except Exception as e:
                logger.error(f"Error in cache warming: {e}")

    async def _eviction_task(self) -> None:
        """Periodically evict cold entries"""
        while self._running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes

                # Analyze patterns
                await self.analyze_patterns()

                # Evict cold keys from L1
                cold_keys = self.stats.cold_keys
                evicted = 0

                for key in cold_keys[: self.config.eviction_batch_size]:
                    if await self.l1_cache.delete(key):
                        evicted += 1

                if evicted > 0:
                    self.stats.evictions += evicted
                    logger.info(f"Evicted {evicted} cold keys from L1 cache")

            except Exception as e:
                logger.error(f"Error in cache eviction: {e}")

    async def _stats_task(self) -> None:
        """Periodically log cache statistics"""
        while self._running:
            try:
                await asyncio.sleep(600)  # Every 10 minutes

                stats = self.get_stats()
                logger.info(f"Cache stats: {json.dumps(stats, indent=2)}")

            except Exception as e:
                logger.error(f"Error in cache stats: {e}")


# Global cache manager
_cache_manager: Optional[HierarchicalCache] = None


def get_cache_manager(config: Optional[CacheConfig] = None) -> HierarchicalCache:
    """Get global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = HierarchicalCache(config or CacheConfig())
    return _cache_manager


# Convenience functions
async def get_cached(key: str) -> Optional[Any]:
    """Get value from global cache"""
    return await get_cache_manager().get(key)


async def set_cached(key: str, value: Any, ttl: Optional[float] = None) -> None:
    """Set value in global cache"""
    await get_cache_manager().set(key, value, ttl)


async def delete_cached(key: str) -> bool:
    """Delete key from global cache"""
    return await get_cache_manager().delete(key)


async def clear_cache() -> None:
    """Clear global cache"""
    await get_cache_manager().clear()

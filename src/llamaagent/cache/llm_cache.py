"""
Specialized LLM Response Caching

Provides semantic caching and response optimization for LLM calls.
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    from sentence_transformers import SentenceTransformer

    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    SentenceTransformer = None

from ..llm.messages import LLMMessage, LLMResponse
from . import CacheBackend

logger = logging.getLogger(__name__)


@dataclass
class CachedResponse:
    """Cached LLM response with metadata"""

    response: LLMResponse
    prompt_hash: str
    prompt_text: str
    embedding: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)
    hit_count: int = 0
    cost_saved: float = 0.0
    tokens_saved: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "response": {
                "content": self.response.content,
                "model": self.response.model,
                "tokens_used": self.response.tokens_used,
                "cost": getattr(self.response, 'cost', None),
                "usage": getattr(self.response, 'usage', None),
            },
            "prompt_hash": self.prompt_hash,
            "prompt_text": self.prompt_text,
            "embedding": (
                self.embedding.tolist() if self.embedding is not None else None
            ),
            "timestamp": self.timestamp,
            "hit_count": self.hit_count,
            "cost_saved": self.cost_saved,
            "tokens_saved": self.tokens_saved,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CachedResponse':
        """Create from dictionary"""
        response_data = data["response"]
        response = LLMResponse(
            content=response_data["content"],
            model=response_data.get("model", "unknown"),
            tokens_used=response_data.get("tokens_used", 0),
        )

        # Set additional attributes
        if response_data.get("cost"):
            response.cost = response_data["cost"]
        if response_data.get("usage"):
            response.usage = response_data["usage"]

        embedding = None
        if data.get("embedding") and NUMPY_AVAILABLE:
            embedding = np.array(data["embedding"])

        return cls(
            response=response,
            prompt_hash=data["prompt_hash"],
            prompt_text=data["prompt_text"],
            embedding=embedding,
            timestamp=data["timestamp"],
            hit_count=data.get("hit_count", 0),
            cost_saved=data.get("cost_saved", 0.0),
            tokens_saved=data.get("tokens_saved", 0),
        )


class LLMCache:
    """Advanced LLM response caching system"""

    def __init__(
        self,
        ttl: int = 300,  # 5 minutes default
        max_entries: int = 10000,
        cache_backend: CacheBackend = CacheBackend.MEMORY,
        enable_semantic_cache: bool = True,
        similarity_threshold: float = 0.85,
        **cache_config,
    ):
        """Initialize LLM cache."""
        # Import here to avoid circular imports
        from . import CacheManager

        self.cache = CacheManager(backend=cache_backend, **cache_config)
        self.ttl = ttl
        self.max_entries = max_entries
        self.enable_semantic_cache = enable_semantic_cache and EMBEDDINGS_AVAILABLE
        self.similarity_threshold = similarity_threshold

        # Semantic cache components
        self.embeddings_model = None
        self.semantic_index: Dict[str, CachedResponse] = {}

        if self.enable_semantic_cache:
            self._initialize_semantic_cache()

        # Statistics
        self.stats = {
            "exact_hits": 0,
            "semantic_hits": 0,
            "misses": 0,
            "total_cost_saved": 0.0,
            "total_tokens_saved": 0,
        }

    def _initialize_semantic_cache(self) -> None:
        """Initialize semantic cache with embeddings model."""
        if not EMBEDDINGS_AVAILABLE:
            logger.warning("Embeddings not available, disabling semantic cache")
            self.enable_semantic_cache = False
            return

        try:
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Semantic cache initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize semantic cache: {e}")
            self.enable_semantic_cache = False

    def _generate_cache_key(self, messages: List[LLMMessage], **kwargs) -> str:
        """Generate cache key from messages and parameters"""
        key_data = {
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "params": {
                k: v
                for k, v in kwargs.items()
                if k in ["temperature", "max_tokens", "model"]
            },
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _extract_prompt_text(self, messages: List[LLMMessage]) -> str:
        """Extract text from messages for semantic comparison."""
        return " ".join([msg.content for msg in messages])

    async def get(self, messages: List[LLMMessage], **kwargs) -> Optional[LLMResponse]:
        """Get cached response for messages"""
        # Try exact match first
        cache_key = self._generate_cache_key(messages, **kwargs)
        cached_data = await self.cache.get(cache_key)

        if cached_data:
            cached_response = CachedResponse.from_dict(cached_data)
            cached_response.hit_count += 1

            # Update statistics
            self.stats["exact_hits"] += 1
            if (
                hasattr(cached_response.response, 'cost')
                and cached_response.response.cost
            ):
                self.stats["total_cost_saved"] += cached_response.response.cost
                cached_response.cost_saved += cached_response.response.cost

            if (
                hasattr(cached_response.response, 'usage')
                and cached_response.response.usage
            ):
                tokens = cached_response.response.usage.get("total_tokens", 0)
                self.stats["total_tokens_saved"] += tokens
                cached_response.tokens_saved += tokens

            # Update cache with incremented stats
            await self.cache.set(cache_key, cached_response.to_dict(), ttl=self.ttl)

            logger.debug("Cache hit (exact)")
            return cached_response.response

        # Try semantic match
        if self.enable_semantic_cache:
            semantic_match = await self._semantic_search(messages, **kwargs)
            if semantic_match:
                self.stats["semantic_hits"] += 1
                logger.debug(
                    f"Cache hit (semantic) with similarity {semantic_match[1]:.3f}"
                )
                return semantic_match[0].response

        self.stats["misses"] += 1
        return None

    async def set(
        self, messages: List[LLMMessage], response: LLMResponse, **kwargs
    ) -> None:
        """Cache LLM response"""
        cache_key = self._generate_cache_key(messages, **kwargs)
        prompt_text = self._extract_prompt_text(messages)

        # Create cached response
        cached_response = CachedResponse(
            response=response, prompt_hash=cache_key, prompt_text=prompt_text
        )

        # Add semantic embedding if enabled
        if self.enable_semantic_cache and self.embeddings_model:
            try:
                embedding = self.embeddings_model.encode(prompt_text)
                cached_response.embedding = embedding

                # Add to semantic index
                self.semantic_index[cache_key] = cached_response
            except Exception as e:
                logger.warning(f"Failed to generate embedding: {e}")

        # Store in cache
        await self.cache.set(cache_key, cached_response.to_dict(), ttl=self.ttl)

    async def _semantic_search(
        self, messages: List[LLMMessage], **kwargs
    ) -> Optional[Tuple[CachedResponse, float]]:
        """Search for semantically similar cached responses"""
        if not self.enable_semantic_cache or not self.embeddings_model:
            return None

        try:
            query_text = self._extract_prompt_text(messages)
            query_embedding = self.embeddings_model.encode(query_text)

            best_match = None
            best_similarity = 0.0

            for cached_response in self.semantic_index.values():
                if cached_response.embedding is not None:
                    similarity = np.dot(query_embedding, cached_response.embedding) / (
                        np.linalg.norm(query_embedding)
                        * np.linalg.norm(cached_response.embedding)
                    )

                    if (
                        similarity > best_similarity
                        and similarity >= self.similarity_threshold
                    ):
                        best_similarity = similarity
                        best_match = cached_response

            return (best_match, best_similarity) if best_match else None

        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_hits = self.stats["exact_hits"] + self.stats["semantic_hits"]
        total_requests = total_hits + self.stats["misses"]

        return {
            "exact_hits": self.stats["exact_hits"],
            "semantic_hits": self.stats["semantic_hits"],
            "misses": self.stats["misses"],
            "hit_rate": total_hits / total_requests if total_requests > 0 else 0,
            "semantic_hit_rate": (
                self.stats["semantic_hits"] / total_requests
                if total_requests > 0
                else 0
            ),
            "total_cost_saved": self.stats["total_cost_saved"],
            "total_tokens_saved": self.stats["total_tokens_saved"],
            "cache_size": len(self.semantic_index),
        }

    def cache_key_generator(self, key_prefix: str = ""):
        """Generate cache key function for decorators"""

        def _generate_key(func_name: str, *args, **kwargs) -> str:
            key_data = {
                "function": func_name,
                "args": args,
                "kwargs": kwargs,
                "prefix": key_prefix,
            }
            key_string = json.dumps(key_data, sort_keys=True, default=str)
            return hashlib.md5(key_string.encode()).hexdigest()

        return _generate_key

    def cached(self, ttl: Optional[int] = None, key_prefix: str = "", condition=None):
        """Decorator for caching function results"""
        cache_ttl = ttl or self.ttl
        key_generator = self.cache_key_generator(key_prefix)

        def decorator(func):
            async def async_wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = key_generator(func.__name__, *args, **kwargs)

                # Try to get from cache
                cached_result = await self.cache.get(cache_key)
                if cached_result is not None:
                    return cached_result

                # Execute function
                result = await func(*args, **kwargs)

                # Cache if condition is met
                if condition is None or condition(result):
                    await self.cache.set(cache_key, result, ttl=cache_ttl)

                return result

            def sync_wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = key_generator(func.__name__, *args, **kwargs)

                # Try to get from cache
                loop = asyncio.get_event_loop()
                cached_result = loop.run_until_complete(self.cache.get(cache_key))
                if cached_result is not None:
                    return cached_result

                # Execute function
                result = func(*args, **kwargs)

                # Cache if condition is met
                if condition is None or condition(result):
                    loop.run_until_complete(
                        self.cache.set(cache_key, result, ttl=cache_ttl)
                    )

                return result

            # Return appropriate wrapper based on function type
            import inspect

            if inspect.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator


class AdvancedLLMCache(LLMCache):
    """Advanced LLM cache with enhanced semantic capabilities"""

    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", index_type: str = "simple", **kwargs
    ):
        """Initialize advanced cache with better semantic search."""
        super().__init__(enable_semantic_cache=True, **kwargs)
        self.model_name = model_name
        self.index_type = index_type

        if self.enable_semantic_cache:
            self._initialize_advanced_index()

    def _initialize_advanced_index(self) -> None:
        """Initialize advanced semantic index (e.g., FAISS)."""
        try:
            # Upgrade to better model
            self.embeddings_model = SentenceTransformer(self.model_name)

            # TODO: Integrate FAISS or other vector index for better performance
            # This would improve search speed for large caches

            logger.info(f"Advanced semantic cache initialized with {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize advanced semantic cache: {e}")
            self.enable_semantic_cache = False

    async def batch_search(
        self, queries: List[List[LLMMessage]], **kwargs
    ) -> List[Optional[LLMResponse]]:
        """Batch semantic search for multiple queries"""
        results: List[Optional[LLMResponse]] = []

        if not self.enable_semantic_cache:
            return [None] * len(queries)

        # Extract all prompt texts
        prompt_texts = [self._extract_prompt_text(messages) for messages in queries]

        try:
            # Batch encode
            query_embeddings = self.embeddings_model.encode(prompt_texts)

            # Search for each query
            for i, query_embedding in enumerate(query_embeddings):
                best_match = None
                best_similarity = 0.0

                for cached_response in self.semantic_index.values():
                    if cached_response.embedding is not None:
                        similarity = np.dot(
                            query_embedding, cached_response.embedding
                        ) / (
                            np.linalg.norm(query_embedding)
                            * np.linalg.norm(cached_response.embedding)
                        )

                        if (
                            similarity > best_similarity
                            and similarity >= self.similarity_threshold
                        ):
                            best_similarity = similarity
                            best_match = cached_response

                results.append(best_match.response if best_match else None)

        except Exception as e:
            logger.warning(f"Batch semantic search failed: {e}")
            results = [None] * len(queries)

        return results

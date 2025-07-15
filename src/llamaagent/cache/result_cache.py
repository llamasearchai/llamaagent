"""
Result Caching Decorators and Utilities

Provides easy-to-use caching decorators for functions and methods.
"""

import asyncio
import functools
import hashlib
import inspect
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar, Union, cast

from . import CacheBackend, CacheManager

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class CacheOptions:
    """Options for result caching"""

    ttl: Optional[float] = 3600  # 1 hour default
    key_prefix: str = ""
    key_generator: Optional[Callable] = None
    condition: Optional[Callable[[Any], bool]] = None
    cache_none: bool = True
    cache_exceptions: bool = False
    backend: CacheBackend = CacheBackend.MEMORY
    invalidate_on: Optional[
        list[str]
    ] = None  # Function names that invalidate this cache


class ResultCache:
    """Result caching system with invalidation support"""

    def __init__(
        self, cache_manager: Optional[CacheManager] = None, default_ttl: float = 3600
    ):
        self.cache_manager = cache_manager or CacheManager()
        self.default_ttl = default_ttl
        self.invalidation_map: dict[
            str, set[str]
        ] = {}  # func_name -> set of cache prefixes

    def cache(
        self,
        ttl: Optional[float] = None,
        key: Optional[Union[str, Callable]] = None,
        condition: Optional[Callable[[Any], bool]] = None,
        cache_none: bool = True,
        cache_exceptions: bool = False,
        invalidate_on: Optional[list[str]] = None,
    ) -> Callable[[F], F]:
        """Decorator for caching function results"""

        def decorator(func: F) -> F:
            # Determine cache key prefix
            if isinstance(key, str):
                key_prefix = key
            else:
                key_prefix = f"{func.__module__}.{func.__qualname__}"

            # Register invalidation triggers
            if invalidate_on:
                for trigger_func in invalidate_on:
                    if trigger_func not in self.invalidation_map:
                        self.invalidation_map[trigger_func] = set()
                    self.invalidation_map[trigger_func].add(key_prefix)

            # Create wrapper based on function type
            if asyncio.iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    # Generate cache key
                    cache_key = self._generate_cache_key(
                        key_prefix, func, args, kwargs, key
                    )

                    # Try to get from cache
                    cached_result = await self.cache_manager.get(cache_key)
                    if cached_result is not None:
                        if isinstance(cached_result, dict) and cached_result.get(
                            "_exception"
                        ):
                            # Re-raise cached exception
                            raise Exception(cached_result["_exception"])
                        return cached_result

                    # Execute function
                    try:
                        result = await func(*args, **kwargs)

                        # Check caching conditions
                        should_cache = True
                        if result is None and not cache_none:
                            should_cache = False
                        if condition and not condition(result):
                            should_cache = False

                        # Cache result
                        if should_cache:
                            cache_ttl = ttl or self.default_ttl
                            await self.cache_manager.set(cache_key, result, cache_ttl)

                        return result

                    except Exception as e:
                        if cache_exceptions:
                            # Cache the exception
                            cached_exc = {"_exception": str(e)}
                            await self.cache_manager.set(
                                cache_key,
                                cached_exc,
                                min(
                                    300, ttl or self.default_ttl
                                ),  # Shorter TTL for exceptions
                            )
                        raise

                return cast(F, async_wrapper)
            else:

                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    # For sync functions, we need to run async operations
                    loop = asyncio.get_event_loop()

                    # Generate cache key
                    cache_key = self._generate_cache_key(
                        key_prefix, func, args, kwargs, key
                    )

                    # Try to get from cache
                    cached_result = loop.run_until_complete(
                        self.cache_manager.get(cache_key)
                    )
                    if cached_result is not None:
                        if isinstance(cached_result, dict) and cached_result.get(
                            "_exception"
                        ):
                            raise Exception(cached_result["_exception"])
                        return cached_result

                    # Execute function
                    try:
                        result = func(*args, **kwargs)

                        # Check caching conditions
                        should_cache = True
                        if result is None and not cache_none:
                            should_cache = False
                        if condition and not condition(result):
                            should_cache = False

                        # Cache result
                        if should_cache:
                            cache_ttl = ttl or self.default_ttl
                            loop.run_until_complete(
                                self.cache_manager.set(cache_key, result, cache_ttl)
                            )

                        return result

                    except Exception as e:
                        if cache_exceptions:
                            cached_exc = {"_exception": str(e)}
                            loop.run_until_complete(
                                self.cache_manager.set(
                                    cache_key,
                                    cached_exc,
                                    min(300, ttl or self.default_ttl),
                                )
                            )
                        raise

                return cast(F, sync_wrapper)

        return decorator

    def invalidate(
        self, func: Optional[Union[str, Callable]] = None, *args, **kwargs
    ) -> None:
        """Invalidate cached results"""
        if func is None:
            # Clear all cache
            asyncio.create_task(self.cache_manager.clear())
            return

        # Determine what to invalidate
        if isinstance(func, str):
            func_name = func
        elif callable(func):
            func_name = f"{func.__module__}.{func.__qualname__}"
        else:
            raise ValueError("func must be a string or callable")

        # Invalidate direct function cache
        if args or kwargs:
            # Invalidate specific call
            cache_key = self._generate_cache_key(func_name, func, args, kwargs)
            asyncio.create_task(self.cache_manager.delete(cache_key))
        else:
            # Invalidate all calls to this function
            # This would require pattern matching support in cache backend
            logger.warning(
                f"Pattern-based invalidation not fully implemented for {func_name}"
            )

        # Invalidate dependent caches
        if func_name in self.invalidation_map:
            for dependent_prefix in self.invalidation_map[func_name]:
                logger.info(f"Invalidating dependent cache: {dependent_prefix}")
                # This would also require pattern matching
                asyncio.create_task(self._invalidate_prefix(dependent_prefix))

    async def _invalidate_prefix(self, prefix: str) -> None:
        """Invalidate all cache entries with given prefix"""
        # This is a simplified implementation
        # In production, you'd want pattern-based deletion
        logger.warning(f"Prefix-based invalidation not fully implemented for {prefix}")

    def _generate_cache_key(
        self,
        prefix: str,
        func: Callable,
        args: tuple,
        kwargs: dict,
        key_func: Optional[Callable] = None,
    ) -> str:
        """Generate cache key for function call"""
        if key_func and callable(key_func):
            # Custom key generator
            return key_func(*args, **kwargs)

        # Default key generation
        key_parts = [prefix]

        # Handle method calls (skip 'self' or 'cls')
        if args and hasattr(args[0], "__class__"):
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            if params and params[0] in ("self", "cls"):
                # Include class name but skip instance
                key_parts.append(args[0].__class__.__name__)
                args = args[1:]

        # Add arguments
        for arg in args:
            key_parts.append(self._serialize_arg(arg))

        # Add keyword arguments
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={self._serialize_arg(v)}")

        # Create hash
        key_str = ":".join(key_parts)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _serialize_arg(self, arg: Any) -> str:
        """Serialize argument for cache key"""
        if isinstance(arg, (str, int, float, bool, type(None))):
            return str(arg)
        elif isinstance(arg, (list, tuple, dict)):
            return json.dumps(arg, sort_keys=True, default=str)
        elif hasattr(arg, "__dict__"):
            # For objects, use their dict representation
            return json.dumps(arg.__dict__, sort_keys=True, default=str)
        else:
            # Fallback to string representation
            return str(arg)


# Global result cache instance
_result_cache: Optional[ResultCache] = None


def get_result_cache() -> ResultCache:
    """Get global result cache instance"""
    global _result_cache
    if _result_cache is None:
        _result_cache = ResultCache()
    return _result_cache


# Convenience decorators


def cache_result(
    ttl: Optional[float] = None,
    key: Optional[Union[str, Callable]] = None,
    condition: Optional[Callable[[Any], bool]] = None,
    cache_none: bool = True,
    cache_exceptions: bool = False,
    invalidate_on: Optional[list[str]] = None,
) -> Callable[[F], F]:
    """Decorator for caching function results"""
    return get_result_cache().cache(
        ttl=ttl,
        key=key,
        condition=condition,
        cache_none=cache_none,
        cache_exceptions=cache_exceptions,
        invalidate_on=invalidate_on,
    )


def invalidate_cache(
    func: Optional[Union[str, Callable]] = None, *args, **kwargs
) -> None:
    """Invalidate cached results"""
    get_result_cache().invalidate(func, *args, **kwargs)


# Specialized cache decorators


def cache_expensive(ttl: float = 3600) -> Callable[[F], F]:
    """Cache decorator for expensive operations"""
    return cache_result(
        ttl=ttl, cache_none=False, condition=lambda result: result is not None
    )


def cache_immutable(ttl: float = 86400) -> Callable[[F], F]:
    """Cache decorator for immutable results (24 hour default)"""
    return cache_result(ttl=ttl, cache_none=True, cache_exceptions=False)


def cache_with_invalidation(
    ttl: float = 3600, invalidate_on: list[str] = None
) -> Callable[[F], F]:
    """Cache with automatic invalidation on specific function calls"""
    return cache_result(ttl=ttl, invalidate_on=invalidate_on or [])


# Method cache decorator for classes


class method_cache:
    """Cache decorator for class methods"""

    def __init__(
        self,
        ttl: Optional[float] = None,
        key_prefix: Optional[str] = None,
        include_instance: bool = True,
    ):
        self.ttl = ttl
        self.key_prefix = key_prefix
        self.include_instance = include_instance
        self.cache = get_result_cache()

    def __call__(self, method: F) -> F:
        @functools.wraps(method)
        async def async_wrapper(instance, *args, **kwargs):
            # Generate cache key
            prefix = (
                self.key_prefix or f"{instance.__class__.__name__}.{method.__name__}"
            )

            if self.include_instance:
                # Include instance ID in key
                instance_id = id(instance)
                cache_key = f"{prefix}:{instance_id}:{self.cache._serialize_arg(args)}:{self.cache._serialize_arg(kwargs)}"
            else:
                # Class-level cache
                cache_key = f"{prefix}:{self.cache._serialize_arg(args)}:{self.cache._serialize_arg(kwargs)}"

            # Try cache
            cached = await self.cache.cache_manager.get(cache_key)
            if cached is not None:
                return cached

            # Execute method
            if asyncio.iscoroutinefunction(method):
                result = await method(instance, *args, **kwargs)
            else:
                result = method(instance, *args, **kwargs)

            # Cache result
            await self.cache.cache_manager.set(
                cache_key, result, self.ttl or self.cache.default_ttl
            )

            return result

        @functools.wraps(method)
        def sync_wrapper(instance, *args, **kwargs):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(async_wrapper(instance, *args, **kwargs))

        if asyncio.iscoroutinefunction(method):
            return cast(F, async_wrapper)
        else:
            return cast(F, sync_wrapper)


# TTL strategies


class TTLStrategy:
    """TTL calculation strategies"""

    @staticmethod
    def exponential_backoff(
        base_ttl: float = 60, max_ttl: float = 3600, factor: float = 2.0
    ) -> Callable[[int], float]:
        """Exponential backoff TTL based on access count"""

        def calculate_ttl(access_count: int) -> float:
            ttl = base_ttl * (factor**access_count)
            return min(ttl, max_ttl)

        return calculate_ttl

    @staticmethod
    def time_based(
        morning_ttl: float = 3600,
        afternoon_ttl: float = 1800,
        evening_ttl: float = 900,
        night_ttl: float = 7200,
    ) -> Callable[[], float]:
        """TTL based on time of day"""

        def calculate_ttl() -> float:
            hour = time.localtime().tm_hour
            if 6 <= hour < 12:
                return morning_ttl
            elif 12 <= hour < 18:
                return afternoon_ttl
            elif 18 <= hour < 22:
                return evening_ttl
            else:
                return night_ttl

        return calculate_ttl

    @staticmethod
    def size_based(
        small_ttl: float = 3600,
        medium_ttl: float = 1800,
        large_ttl: float = 900,
        size_threshold_kb: tuple[float, float] = (10, 100),
    ) -> Callable[[Any], float]:
        """TTL based on result size"""

        def calculate_ttl(result: Any) -> float:
            try:
                import sys

                size_kb = sys.getsizeof(result) / 1024

                if size_kb < size_threshold_kb[0]:
                    return small_ttl
                elif size_kb < size_threshold_kb[1]:
                    return medium_ttl
                else:
                    return large_ttl
            except:
                return medium_ttl

        return calculate_ttl

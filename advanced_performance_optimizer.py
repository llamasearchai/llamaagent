#!/usr/bin/env python3
"""
Advanced Performance Optimization System for LlamaAgent

This module implements comprehensive performance optimizations including:
- List comprehension optimizations
- Memory management improvements
- Async operation enhancements
- Caching strategies
- Database query optimization
- Network request optimization

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import time
import functools
import weakref
import gc
from typing import Dict, List, Any, Optional, Callable, TypeVar, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import multiprocessing
from collections import defaultdict, deque
import heapq
import statistics
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    execution_time: float = 0.0
    memory_usage: int = 0
    cpu_usage: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    api_calls: int = 0
    database_queries: int = 0
    network_requests: int = 0
    errors: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class OptimizationConfig:
    """Configuration for performance optimizations."""
    enable_caching: bool = True
    enable_async_processing: bool = True
    enable_list_comprehensions: bool = True
    enable_memory_optimization: bool = True
    enable_query_optimization: bool = True
    max_workers: int = multiprocessing.cpu_count()
    cache_size: int = 1000
    cache_ttl: int = 3600  # seconds
    batch_size: int = 100
    timeout: float = 30.0


class AdvancedCache:
    """Advanced caching system with TTL and LRU eviction."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with TTL check."""
        with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            if time.time() - entry['timestamp'] > self.ttl:
                del self._cache[key]
                del self._access_times[key]
                return None
            
            self._access_times[key] = time.time()
            return entry['value']
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache with automatic eviction."""
        with self._lock:
            # Evict expired entries
            self._evict_expired()
            
            # Evict LRU entries if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()
            
            self._cache[key] = {
                'value': value,
                'timestamp': time.time()
            }
            self._access_times[key] = time.time()
    
    def _evict_expired(self) -> None:
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if current_time - entry['timestamp'] > self.ttl
        ]
        for key in expired_keys:
            del self._cache[key]
            del self._access_times[key]
    
    def _evict_lru(self) -> None:
        """Remove least recently used entry."""
        if not self._access_times:
            return
        
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        del self._cache[lru_key]
        del self._access_times[lru_key]
    
    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hit_rate': 0.0,  # Would need to track hits/misses
                'expired_count': 0  # Would need to track
            }


class AsyncBatchProcessor:
    """Asynchronous batch processing for improved performance."""
    
    def __init__(self, batch_size: int = 100, max_workers: int = 4):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self._semaphore = asyncio.Semaphore(max_workers)
    
    async def process_batch(self, items: List[T], processor: Callable[[T], Any]) -> List[Any]:
        """Process items in batches asynchronously."""
        results = []
        
        # Split into batches
        batches = [
            items[i:i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]
        
        # Process batches concurrently
        tasks = [
            self._process_single_batch(batch, processor)
            for batch in batches
        ]
        
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                logger.error(f"Batch processing error: {batch_result}")
                continue
            results.extend(batch_result)
        
        return results
    
    async def _process_single_batch(self, batch: List[T], processor: Callable[[T], Any]) -> List[Any]:
        """Process a single batch with semaphore control."""
        async with self._semaphore:
            if asyncio.iscoroutinefunction(processor):
                tasks = [processor(item) for item in batch]
                return await asyncio.gather(*tasks, return_exceptions=True)
            else:
                # Run sync processor in thread pool
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    tasks = [
                        loop.run_in_executor(executor, processor, item)
                        for item in batch
                    ]
                    return await asyncio.gather(*tasks, return_exceptions=True)


class MemoryOptimizer:
    """Memory optimization utilities."""
    
    @staticmethod
    def optimize_list_operations(data: List[Any], operation: str, condition: Optional[Callable] = None) -> List[Any]:
        """Optimize list operations using comprehensions."""
        if operation == "filter" and condition:
            return [item for item in data if condition(item)]
        elif operation == "map" and condition:
            return [condition(item) for item in data]
        elif operation == "filter_map" and condition:
            return [condition(item) for item in data if condition(item) is not None]
        else:
            return data
    
    @staticmethod
    def optimize_dict_operations(data: Dict[Any, Any], operation: str, condition: Optional[Callable] = None) -> Dict[Any, Any]:
        """Optimize dictionary operations using comprehensions."""
        if operation == "filter_keys" and condition:
            return {k: v for k, v in data.items() if condition(k)}
        elif operation == "filter_values" and condition:
            return {k: v for k, v in data.items() if condition(v)}
        elif operation == "transform_values" and condition:
            return {k: condition(v) for k, v in data.items()}
        else:
            return data
    
    @staticmethod
    def cleanup_memory() -> None:
        """Force garbage collection and memory cleanup."""
        gc.collect()
    
    @staticmethod
    def get_memory_usage() -> int:
        """Get current memory usage in bytes."""
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss


class QueryOptimizer:
    """Database and API query optimization."""
    
    def __init__(self, cache: AdvancedCache):
        self.cache = cache
        self._query_stats: Dict[str, List[float]] = defaultdict(list)
    
    async def optimized_query(self, query_key: str, query_func: Callable, *args, **kwargs) -> Any:
        """Execute query with caching and performance tracking."""
        # Check cache first
        cached_result = self.cache.get(query_key)
        if cached_result is not None:
            return cached_result
        
        # Execute query with timing
        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(query_func):
                result = await query_func(*args, **kwargs)
            else:
                result = query_func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            self._query_stats[query_key].append(execution_time)
            
            # Cache successful result
            self.cache.set(query_key, result)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._query_stats[f"{query_key}_error"].append(execution_time)
            raise
    
    def get_query_stats(self) -> Dict[str, Dict[str, float]]:
        """Get query performance statistics."""
        stats = {}
        for query_key, times in self._query_stats.items():
            if times:
                stats[query_key] = {
                    'avg_time': statistics.mean(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'total_calls': len(times)
                }
        return stats


class PerformanceMonitor:
    """Real-time performance monitoring."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._metrics: deque = deque(maxlen=window_size)
        self._lock = threading.Lock()
    
    def record_metric(self, metric: PerformanceMetrics) -> None:
        """Record a performance metric."""
        with self._lock:
            self._metrics.append(metric)
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        with self._lock:
            if not self._metrics:
                return {}
            
            recent_metrics = list(self._metrics)
            
            return {
                'avg_execution_time': statistics.mean(m.execution_time for m in recent_metrics),
                'avg_memory_usage': statistics.mean(m.memory_usage for m in recent_metrics),
                'avg_cpu_usage': statistics.mean(m.cpu_usage for m in recent_metrics),
                'total_api_calls': sum(m.api_calls for m in recent_metrics),
                'total_cache_hits': sum(m.cache_hits for m in recent_metrics),
                'total_cache_misses': sum(m.cache_misses for m in recent_metrics),
                'total_errors': sum(m.errors for m in recent_metrics),
                'cache_hit_rate': self._calculate_cache_hit_rate(recent_metrics),
                'error_rate': self._calculate_error_rate(recent_metrics)
            }
    
    def _calculate_cache_hit_rate(self, metrics: List[PerformanceMetrics]) -> float:
        """Calculate cache hit rate."""
        total_hits = sum(m.cache_hits for m in metrics)
        total_misses = sum(m.cache_misses for m in metrics)
        total_requests = total_hits + total_misses
        
        return total_hits / total_requests if total_requests > 0 else 0.0
    
    def _calculate_error_rate(self, metrics: List[PerformanceMetrics]) -> float:
        """Calculate error rate."""
        total_errors = sum(m.errors for m in metrics)
        total_operations = len(metrics)
        
        return total_errors / total_operations if total_operations > 0 else 0.0


def performance_decorator(monitor: Optional[PerformanceMonitor] = None):
    """Decorator for automatic performance monitoring."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = MemoryOptimizer.get_memory_usage()
            
            try:
                result = await func(*args, **kwargs)
                errors = 0
            except Exception as e:
                errors = 1
                raise
            finally:
                end_time = time.time()
                end_memory = MemoryOptimizer.get_memory_usage()
                
                if monitor:
                    metric = PerformanceMetrics(
                        execution_time=end_time - start_time,
                        memory_usage=end_memory - start_memory,
                        errors=errors
                    )
                    monitor.record_metric(metric)
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = MemoryOptimizer.get_memory_usage()
            
            try:
                result = func(*args, **kwargs)
                errors = 0
            except Exception as e:
                errors = 1
                raise
            finally:
                end_time = time.time()
                end_memory = MemoryOptimizer.get_memory_usage()
                
                if monitor:
                    metric = PerformanceMetrics(
                        execution_time=end_time - start_time,
                        memory_usage=end_memory - start_memory,
                        errors=errors
                    )
                    monitor.record_metric(metric)
            
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


class AdvancedPerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.cache = AdvancedCache(config.cache_size, config.cache_ttl)
        self.batch_processor = AsyncBatchProcessor(config.batch_size, config.max_workers)
        self.query_optimizer = QueryOptimizer(self.cache)
        self.monitor = PerformanceMonitor()
        self.memory_optimizer = MemoryOptimizer()
    
    async def optimize_list_processing(self, data: List[Any], operations: List[Dict[str, Any]]) -> List[Any]:
        """Optimize multiple list operations."""
        result = data
        
        for operation in operations:
            op_type = operation.get('type')
            condition = operation.get('condition')
            
            if self.config.enable_list_comprehensions:
                result = self.memory_optimizer.optimize_list_operations(
                    result, op_type, condition
                )
            else:
                # Fallback to traditional operations
                if op_type == "filter" and condition:
                    result = list(filter(condition, result))
                elif op_type == "map" and condition:
                    result = list(map(condition, result))
        
        return result
    
    async def optimize_async_operations(self, operations: List[Callable]) -> List[Any]:
        """Optimize multiple async operations."""
        if self.config.enable_async_processing:
            return await self.batch_processor.process_batch(
                operations, 
                lambda op: op() if callable(op) else op
            )
        else:
            # Sequential processing
            results = []
            for op in operations:
                if asyncio.iscoroutinefunction(op):
                    result = await op()
                else:
                    result = op()
                results.append(result)
            return results
    
    async def optimize_database_queries(self, queries: List[Dict[str, Any]]) -> List[Any]:
        """Optimize database queries with caching."""
        results = []
        
        for query in queries:
            query_key = query.get('key', str(hash(str(query))))
            query_func = query.get('function')
            args = query.get('args', [])
            kwargs = query.get('kwargs', {})
            
            result = await self.query_optimizer.optimized_query(
                query_key, query_func, *args, **kwargs
            )
            results.append(result)
        
        return results
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            'monitor_stats': self.monitor.get_current_stats(),
            'cache_stats': self.cache.stats(),
            'query_stats': self.query_optimizer.get_query_stats(),
            'memory_usage': self.memory_optimizer.get_memory_usage(),
            'config': {
                'caching_enabled': self.config.enable_caching,
                'async_processing_enabled': self.config.enable_async_processing,
                'list_comprehensions_enabled': self.config.enable_list_comprehensions,
                'max_workers': self.config.max_workers,
                'cache_size': self.config.cache_size,
                'batch_size': self.config.batch_size
            }
        }
    
    async def cleanup_and_optimize(self) -> None:
        """Perform cleanup and optimization tasks."""
        # Clear expired cache entries
        self.cache._evict_expired()
        
        # Force garbage collection
        self.memory_optimizer.cleanup_memory()
        
        # Log performance stats
        stats = self.get_performance_report()
        logger.info(f"Performance optimization completed: {stats['monitor_stats']}")


# Global optimizer instance
_global_optimizer: Optional[AdvancedPerformanceOptimizer] = None


def get_global_optimizer() -> AdvancedPerformanceOptimizer:
    """Get or create global optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        config = OptimizationConfig()
        _global_optimizer = AdvancedPerformanceOptimizer(config)
    return _global_optimizer


def optimize_performance(func: Optional[Callable] = None, *, monitor: bool = True):
    """Decorator for automatic performance optimization."""
    def decorator(f: Callable) -> Callable:
        optimizer = get_global_optimizer()
        
        if monitor:
            f = performance_decorator(optimizer.monitor)(f)
        
        return f
    
    if func is None:
        return decorator
    else:
        return decorator(func)


# Example usage and test functions
async def demonstrate_optimizations():
    """Demonstrate various performance optimizations."""
    
    print("Starting Advanced Performance Optimization Demo")
    print("=" * 50)
    
    # Initialize optimizer
    config = OptimizationConfig(
        enable_caching=True,
        enable_async_processing=True,
        enable_list_comprehensions=True,
        max_workers=4,
        cache_size=1000,
        batch_size=50
    )
    
    optimizer = AdvancedPerformanceOptimizer(config)
    
    # Test list processing optimization
    print("\nResponse Testing List Processing Optimization")
    large_list = list(range(10000))
    
    operations = [
        {'type': 'filter', 'condition': lambda x: x % 2 == 0},
        {'type': 'map', 'condition': lambda x: x * 2},
        {'type': 'filter', 'condition': lambda x: x < 1000}
    ]
    
    start_time = time.time()
    optimized_result = await optimizer.optimize_list_processing(large_list, operations)
    optimization_time = time.time() - start_time
    
    print(f"PASS Processed {len(large_list)} items -> {len(optimized_result)} results")
    print(f"⏱️  Optimization time: {optimization_time:.4f}s")
    
    # Test async operations optimization
    print("\n🔄 Testing Async Operations Optimization")
    
    async def sample_async_operation(delay: float = 0.01):
        await asyncio.sleep(delay)
        return f"Result after {delay}s"
    
    operations = [
        lambda: sample_async_operation(0.01),
        lambda: sample_async_operation(0.02),
        lambda: sample_async_operation(0.01),
        lambda: sample_async_operation(0.03)
    ]
    
    start_time = time.time()
    async_results = await optimizer.optimize_async_operations(operations)
    async_time = time.time() - start_time
    
    print(f"PASS Completed {len(operations)} async operations")
    print(f"⏱️  Total time: {async_time:.4f}s")
    
    # Test caching
    print("\n💾 Testing Caching System")
    
    def expensive_computation(n: int) -> int:
        time.sleep(0.01)  # Simulate expensive operation
        return n ** 2
    
    # First call (cache miss)
    start_time = time.time()
    result1 = optimizer.cache.get("test_key")
    if result1 is None:
        result1 = expensive_computation(42)
        optimizer.cache.set("test_key", result1)
    cache_miss_time = time.time() - start_time
    
    # Second call (cache hit)
    start_time = time.time()
    result2 = optimizer.cache.get("test_key")
    cache_hit_time = time.time() - start_time
    
    print(f"PASS Cache miss time: {cache_miss_time:.4f}s")
    print(f"PASS Cache hit time: {cache_hit_time:.4f}s")
    print(f"Starting Cache speedup: {cache_miss_time / cache_hit_time:.1f}x")
    
    # Get performance report
    print("\nRESULTS Performance Report")
    print("=" * 30)
    report = optimizer.get_performance_report()
    
    for category, stats in report.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        if isinstance(stats, dict):
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
    
    # Cleanup
    await optimizer.cleanup_and_optimize()
    
    print("\nSUCCESS Performance optimization demonstration completed!")


if __name__ == "__main__":
    asyncio.run(demonstrate_optimizations()) 
"""
Advanced performance optimization system for LlamaAgent.

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import asyncio
import concurrent.futures
import functools
import logging
import os
import queue
import threading
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""

    ASYNC_PARALLEL = "async_parallel"
    BATCH_PROCESSING = "batch_processing"
    LAZY_LOADING = "lazy_loading"
    RESOURCE_POOLING = "resource_pooling"
    ADAPTIVE = "adaptive"


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization."""

    execution_time: float = 0.0
    memory_usage: float = 0.0
    throughput: float = 0.0
    success_rate: float = 0.0
    error_rate: float = 0.0
    resource_utilization: float = 0.0

    def __post_init__(self):
        if self.success_rate == 0.0 and self.error_rate == 0.0:
            self.success_rate = 1.0


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""

    strategy: OptimizationStrategy = OptimizationStrategy.ASYNC_PARALLEL
    max_workers: int = 4
    batch_size: int = 10
    timeout: float = 30.0
    memory_limit: int = 1024 * 1024 * 1024  # 1GB
    enable_monitoring: bool = True
    enable_adaptive: bool = True
    cache_size: int = 1000


class ResourceMonitor:
    """Resource monitoring for optimization."""

    def __init__(self, check_interval: float = 1.0):
        self.check_interval = check_interval
        self.cpu_usage_history: List[float] = []
        self.memory_usage_history: List[float] = []
        self.last_check_time = time.time()
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None

    async def start_monitoring(self) -> None:
        """Start resource monitoring."""
        if not self._monitoring:
            self._monitoring = True
            self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                await self._collect_metrics()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")

    async def _collect_metrics(self) -> None:
        """Collect resource metrics."""
        try:
            import psutil

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.cpu_usage_history.append(cpu_percent)
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.memory_usage_history.append(memory_percent)
            # Keep only recent history
            max_history = 100
            if len(self.cpu_usage_history) > max_history:
                self.cpu_usage_history = self.cpu_usage_history[-max_history:]
            if len(self.memory_usage_history) > max_history:
                self.memory_usage_history = self.memory_usage_history[-max_history:]

        except ImportError:
            # psutil not available, use basic metrics
            pass

    def get_current_metrics(self) -> Dict[str, float]:
        """Get current resource metrics."""
        if not self.cpu_usage_history or not self.memory_usage_history:
            return {"cpu_usage": 0.0, "memory_usage": 0.0}

        return {
            "cpu_usage": self.cpu_usage_history[-1] if self.cpu_usage_history else 0.0,
            "memory_usage": self.memory_usage_history[-1]
            if self.memory_usage_history
            else 0.0,
            "avg_cpu_usage": sum(self.cpu_usage_history) / len(self.cpu_usage_history)
            if self.cpu_usage_history
            else 0.0,
            "avg_memory_usage": sum(self.memory_usage_history)
            / len(self.memory_usage_history)
            if self.memory_usage_history
            else 0.0,
        }


class ResourcePool:
    """Generic resource pool for object reuse."""

    def __init__(
        self, factory: Callable[[], Any], max_size: int = 10, timeout: float = 30.0
    ):
        self.factory = factory
        self.max_size = max_size
        self.timeout = timeout
        self._pool: queue.Queue[Any] = queue.Queue()
        self._all_resources: Set[Any] = set()
        self._lock = threading.Lock()

    def acquire(self) -> Any:
        """Acquire resource from pool."""
        try:
            # Try to get from pool
            resource = self._pool.get(timeout=0.1)
            return resource
        except queue.Empty:
            # Create new resource if under limit
            with self._lock:
                if len(self._all_resources) < self.max_size:
                    resource = self.factory()
                    self._all_resources.add(resource)
                    return resource

            # Wait for resource to become available
            return self._pool.get(timeout=self.timeout)

    def release(self, resource: Any) -> None:
        """Release resource back to pool."""
        self._pool.put(resource)

    @asynccontextmanager
    async def get_resource(self):
        """Async context manager for resource acquisition."""
        resource = await asyncio.to_thread(self.acquire)
        try:
            yield resource
        finally:
            await asyncio.to_thread(self.release, resource)


class LazyLoader:
    """Lazy loading implementation for deferred resource loading."""

    def __init__(self, loader_func: Callable[[], Any]):
        self.loader_func = loader_func
        self._value: Optional[Any] = None
        self._loaded = False
        self._lock = threading.Lock()

    @property
    def value(self) -> Any:
        """Get value, loading if necessary."""
        if not self._loaded:
            with self._lock:
                if not self._loaded:
                    self._value = self.loader_func()
                    self._loaded = True
        return self._value

    def reset(self) -> None:
        """Reset loader to unloaded state."""
        with self._lock:
            self._value = None
            self._loaded = False


class BatchProcessor(ABC):
    """Abstract base class for batch processing."""

    def __init__(self, batch_size: int = 10, max_wait_time: float = 1.0):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self._batch_queue: asyncio.Queue[Any] = asyncio.Queue()
        self._processing_task: Optional[asyncio.Task[None]] = None
        self._results: Dict[str, asyncio.Future[Any]] = {}

    async def submit(self, item: Any) -> Any:
        """Submit item for batch processing."""
        item_id = str(id(item))
        future: asyncio.Future[Any] = asyncio.Future()
        self._results[item_id] = future

        await self._batch_queue.put(item_id, item)
        if self._processing_task is None or self._processing_task.done():
            self._processing_task = asyncio.create_task(self._process_batches())
        return await future

    async def _process_batches(self) -> None:
        """Process items in batches."""
        while True:
            batch_items = []
            batch_ids = []

            # Collect batch
            try:
                # Wait for at least one item
                item_id, item = await asyncio.wait_for(
                    self._batch_queue.get(), timeout=self.max_wait_time
                )
                batch_items.append(item)
                batch_ids.append(item_id)
                # Collect additional items up to batch size
                for _ in range(self.batch_size - 1):
                    try:
                        item_id, item = await asyncio.wait_for(
                            self._batch_queue.get(), timeout=0.1
                        )
                        batch_items.append(item)
                        batch_ids.append(item_id)
                    except asyncio.TimeoutError:
                        break

            except asyncio.TimeoutError:
                continue

            # Process batch
            try:
                results = await self._process_batch(batch_items)
                # Set results
                for item_id, result in zip(batch_ids, results, strict=False):
                    if item_id in self._results:
                        self._results[item_id].set_result(result)
                        del self._results[item_id]

            except Exception as e:
                # Set exception for all items in batch
                for item_id in batch_ids:
                    if item_id in self._results:
                        self._results[item_id].set_exception(e)
                        del self._results[item_id]

    @abstractmethod
    async def _process_batch(self, batch_items: List[Any]) -> List[Any]:
        """Process a batch of items."""
        pass


class AsyncParallelizer:
    """Async parallelization utilities."""

    def __init__(self, max_concurrency: int = 10, timeout: Optional[float] = None):
        self.max_concurrency = max_concurrency
        self.timeout = timeout
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self.ordered = True

    async def map(
        self, func: Callable, items: List[Any], ordered: bool = True, **kwargs
    ) -> List[Any]:
        """Map function over items with parallelization."""
        self.ordered = ordered
        if self.ordered:
            return await self._ordered_map(func, items, **kwargs)
        else:
            return await self._unordered_map(func, items, **kwargs)

    async def _ordered_map(
        self, func: Callable, items: List[Any], **kwargs
    ) -> List[Any]:
        """Map maintaining order."""

        async def process_with_index(idx: int, item: Any) -> tuple:
            async with self._semaphore:
                result = await self._execute_func(func, item, **kwargs)
                return idx, result

        # Create tasks
        tasks = [
            asyncio.create_task(process_with_index(idx, item))
            for idx, item in enumerate(items)
        ]

        # Wait for results
        indexed_results = await asyncio.gather(*tasks)
        # Sort by index and return results
        sorted_results = sorted(indexed_results, key=lambda x: x[0])
        return [result for _, result in sorted_results]

    async def _unordered_map(
        self, func: Callable, items: List[Any], **kwargs
    ) -> List[Any]:
        """Map without maintaining order (faster)."""

        async def process_item(item: Any) -> Any:
            async with self._semaphore:
                return await self._execute_func(func, item, **kwargs)

        tasks = [asyncio.create_task(process_item(item)) for item in items]

        return await asyncio.gather(*tasks)

    async def _execute_func(self, func: Callable, item: Any, **kwargs) -> Any:
        """Execute function with timeout."""
        coro = (
            func(item, **kwargs)
            if asyncio.iscoroutinefunction(func)
            else asyncio.to_thread(func, item, **kwargs)
        )
        if self.timeout:
            return await asyncio.wait_for(coro, timeout=self.timeout)
        else:
            return await coro


class AdaptiveOptimizer:
    """Adaptive optimization strategy selector."""

    def __init__(self):
        self.strategies = list(OptimizationStrategy)
        self.strategy_weights = {strategy: 1.0 for strategy in self.strategies}
        self.performance_history: Dict[
            OptimizationStrategy, List[PerformanceMetrics]
        ] = {strategy: [] for strategy in self.strategies}
        self.exploration_rate = 0.1

    def select_strategy(self, context: Dict[str, Any]) -> OptimizationStrategy:
        """Select optimization strategy based on context and history."""
        # Exploration vs exploitation
        if np.random.random() < self.exploration_rate:
            # Explore: random selection
            return np.random.choice(self.strategies)
        else:
            # Exploit: weighted selection based on performance
            weights = list(self.strategy_weights.values())
            total_weight = sum(weights)
            probabilities = [w / total_weight for w in weights]

            return np.random.choice(self.strategies, p=probabilities)

    def update_performance(
        self, strategy: OptimizationStrategy, metrics: PerformanceMetrics
    ) -> None:
        """Update performance metrics for a strategy."""
        self.performance_history[strategy].append(metrics)
        # Update weights based on performance
        recent_metrics = self.performance_history[strategy][-10:]  # Last 10 runs
        if recent_metrics:
            avg_score = np.mean(
                [
                    m.throughput * m.success_rate / max(m.execution_time, 0.01)
                    for m in recent_metrics
                ]
            )
            self.strategy_weights[strategy] = max(0.1, avg_score)

    def get_recommendations(self) -> Dict[str, Any]:
        """Get optimization recommendations."""
        recommendations = {}

        for strategy in self.strategies:
            metrics = self.performance_history[strategy]
            if metrics:
                recent_metrics = metrics[-10:]
                recommendations[strategy.value] = {
                    "avg_execution_time": np.mean(
                        [m.execution_time for m in recent_metrics]
                    ),
                    "avg_throughput": np.mean([m.throughput for m in recent_metrics]),
                    "success_rate": np.mean([m.success_rate for m in recent_metrics]),
                    "weight": self.strategy_weights[strategy],
                }

        return recommendations


class PerformanceOptimizer:
    """
    Main performance optimization coordinator.

    Combines all optimization techniques for maximum performance.
    """

    def __init__(
        self, config: Optional[OptimizationConfig] = None, enable_adaptive: bool = True
    ):
        self.config = config or OptimizationConfig()
        self.max_workers = self.config.max_workers or (os.cpu_count() or 1) * 2

        # Initialize components
        self.adaptive_optimizer = AdaptiveOptimizer() if enable_adaptive else None
        self.parallelizer = AsyncParallelizer(max_concurrency=self.max_workers)
        self.resource_pools: Dict[str, ResourcePool] = {}
        self.lazy_loaders: Dict[str, LazyLoader] = {}
        self.batch_processors: Dict[str, BatchProcessor] = {}

        # Thread and process pools
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        )
        self.process_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=max(1, (os.cpu_count() or 1) // 2)
        )

        # Monitoring
        self.metrics_history: List[PerformanceMetrics] = []
        self.monitoring_task: Optional[asyncio.Task] = None

    async def optimize_async(
        self,
        func: Callable,
        items: List[Any],
        strategy: Optional[OptimizationStrategy] = None,
        **kwargs,
    ) -> List[Any]:
        """Optimize async execution of function over items."""
        # Ensure monitoring is started
        await self._ensure_monitoring_started()

        start_time = time.time()

        # Select strategy
        if strategy is None and self.adaptive_optimizer:
            context = {
                "item_count": len(items),
                "function_name": func.__name__
                if hasattr(func, '__name__')
                else str(func),
            }
            strategy = self.adaptive_optimizer.select_strategy(context)
        strategy = strategy or OptimizationStrategy.ASYNC_PARALLEL

        try:
            # Execute based on strategy
            if strategy == OptimizationStrategy.ASYNC_PARALLEL:
                results = await self.parallelizer.map(func, items, **kwargs)
            elif strategy == OptimizationStrategy.BATCH_PROCESSING:
                # Create temporary batch processor
                class TempBatchProcessor(BatchProcessor):
                    async def _process_batch(self, batch_items: List[Any]) -> List[Any]:
                        return await asyncio.gather(
                            *[func(item, **kwargs) for item in batch_items]
                        )

                processor = TempBatchProcessor()
                results = await asyncio.gather(
                    *[processor.submit(item) for item in items]
                )

            else:
                # Fallback to sequential
                results: List[Any] = []
                for item in items:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(item, **kwargs)
                    else:
                        result = await asyncio.to_thread(func, item, **kwargs)
                    results.append(result)
            # Update metrics
            duration = time.time() - start_time
            metrics = PerformanceMetrics(
                execution_time=duration,
                throughput=len(items) / duration if duration > 0 else 0,
                success_rate=1.0,  # Simplified
            )

            if self.adaptive_optimizer:
                self.adaptive_optimizer.update_performance(strategy, metrics)
            self.metrics_history.append(metrics)
            return results

        except Exception as e:
            logger.error(f"Optimization failed with strategy {strategy}: {e}")
            raise

    def optimize_cpu_bound(
        self, func: Callable, items: List[Any], **kwargs
    ) -> List[Any]:
        """Optimize CPU-bound operations using process pool."""
        # Use process pool for heavy computation
        futures = [self.process_pool.submit(func, item, **kwargs) for item in items]

        return [future.result() for future in concurrent.futures.as_completed(futures)]

    def create_resource_pool(
        self, name: str, factory: Callable[[], Any], max_size: int = 10, **kwargs
    ) -> ResourcePool:
        """Create a resource pool."""
        pool = ResourcePool(factory, max_size, **kwargs)
        self.resource_pools[name] = pool
        return pool

    def create_lazy_loader(
        self, name: str, loader_func: Callable[[], Any]
    ) -> LazyLoader:
        """Create a lazy loader."""
        loader = LazyLoader(loader_func)
        self.lazy_loaders[name] = loader
        return loader

    def create_batch_processor(
        self, name: str, processor_class: type, **kwargs
    ) -> BatchProcessor:
        """Create a batch processor."""
        processor = processor_class(**kwargs)
        self.batch_processors[name] = processor
        return processor

    async def _ensure_monitoring_started(self) -> None:
        """Ensure monitoring task is running."""
        if self.config.enable_monitoring and (
            self.monitoring_task is None or self.monitoring_task.done()
        ):
            self.monitoring_task = asyncio.create_task(self._monitor_performance())

    async def _monitor_performance(self) -> None:
        """Monitor performance metrics."""
        while True:
            try:
                # Collect metrics
                await asyncio.sleep(10)  # Monitor every 10 seconds

                # Log performance stats
                if self.metrics_history:
                    recent_metrics = self.metrics_history[-10:]
                    avg_execution_time = np.mean(
                        [m.execution_time for m in recent_metrics]
                    )
                    avg_throughput = np.mean([m.throughput for m in recent_metrics])
                    logger.info(
                        f"Performance Stats - Avg Execution Time: {avg_execution_time:.2f}s, "
                        f"Avg Throughput: {avg_throughput:.2f} items/s"
                    )
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get current optimization statistics."""
        stats = {
            "total_optimizations": len(self.metrics_history),
            "resource_pools": len(self.resource_pools),
            "lazy_loaders": len(self.lazy_loaders),
            "batch_processors": len(self.batch_processors),
        }

        if self.metrics_history:
            recent_metrics = self.metrics_history[-10:]
            stats["performance"] = {
                "avg_execution_time": np.mean(
                    [m.execution_time for m in recent_metrics]
                ),
                "avg_throughput": np.mean([m.throughput for m in recent_metrics]),
                "avg_success_rate": np.mean([m.success_rate for m in recent_metrics]),
            }

        if self.adaptive_optimizer:
            stats["recommendations"] = self.adaptive_optimizer.get_recommendations()

        return stats

    async def shutdown(self) -> None:
        """Shutdown optimization resources."""
        if self.monitoring_task:
            self.monitoring_task.cancel()

        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        # Cleanup batch processors
        for processor in self.batch_processors.values():
            if hasattr(processor, 'cleanup'):
                await processor.cleanup()


# Global optimizer instance
_global_optimizer: Optional[PerformanceOptimizer] = None


def get_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer()
    return _global_optimizer


async def optimize_parallel(
    func: Callable,
    items: List[Any],
    strategy: Optional[OptimizationStrategy] = None,
    **kwargs,
) -> List[Any]:
    """Convenience function for parallel optimization."""
    optimizer = get_optimizer()
    return await optimizer.optimize_async(func, items, strategy=strategy, **kwargs)


# Decorator for automatic optimization
def optimize(
    strategy: OptimizationStrategy = OptimizationStrategy.ASYNC_PARALLEL,
    max_workers: int = 4,
):
    """Decorator for automatic function optimization."""

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            optimizer = get_optimizer()
            optimizer.config.max_workers = max_workers

            # If first argument is a list, optimize over it
            if args and isinstance(args[0], list):
                items = args[0]
                remaining_args = args[1:]

                async def optimized_func(item):
                    return await func(item, *remaining_args, **kwargs)

                return await optimizer.optimize_async(
                    optimized_func, items, strategy=strategy
                )
            else:
                return await func(*args, **kwargs)

        return wrapper

    return decorator


if __name__ == "__main__":
    # Example usage
    async def example_function(x: int) -> int:
        await asyncio.sleep(0.1)  # Simulate work
        return x * 2

    async def main():
        items = list(range(100))
        # Test optimization
        start_time = time.time()
        results = await optimize_parallel(example_function, items)
        duration = time.time() - start_time

        print(f"Processed {len(items)} items in {duration:.2f}s")
        print(f"Results: {results[:10]}...")  # First 10 results

        # Test decorator
        @optimize(strategy=OptimizationStrategy.ASYNC_PARALLEL)
        async def decorated_function(items: List[int]) -> List[int]:
            return [x * 3 for x in items]

        results2 = await decorated_function(items)
        print(f"Decorated results: {results2[:10]}...")

    asyncio.run(main())

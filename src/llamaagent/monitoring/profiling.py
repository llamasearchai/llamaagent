"""
Advanced profiling system for LlamaAgent with multiple profiling backends.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import cProfile
import functools
import logging
import pstats
import threading
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# Optional imports
try:
    import line_profiler

    LINE_PROFILER_AVAILABLE = True
except ImportError:
    LINE_PROFILER_AVAILABLE = False

try:
    import memory_profiler

    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ProfileResult:
    """Result of a profiling session"""

    name: str
    duration_seconds: float
    cpu_time_seconds: float
    memory_start_mb: float
    memory_peak_mb: float
    memory_end_mb: float
    top_functions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "duration_seconds": self.duration_seconds,
            "cpu_time_seconds": self.cpu_time_seconds,
            "memory_start_mb": self.memory_start_mb,
            "memory_peak_mb": self.memory_peak_mb,
            "memory_end_mb": self.memory_end_mb,
            "top_functions": self.top_functions,
        }


class MemoryTracker:
    """Track memory allocations"""

    def __init__(self) -> None:
        self.snapshots: List[Tuple[str, Any]] = []
        self.enabled = False

    def start_tracking(self) -> None:
        """Start memory tracking"""
        if not self.enabled:
            tracemalloc.start()
            self.enabled = True

    def stop_tracking(self) -> None:
        """Stop memory tracking"""
        if self.enabled:
            tracemalloc.stop()
            self.enabled = False

    def take_snapshot(self, label: str) -> None:
        """Take a memory snapshot"""
        if not self.enabled:
            return
        snapshot = tracemalloc.take_snapshot()
        self.snapshots.append(label, snapshot)

    def get_top_allocations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top memory allocations"""
        if not self.snapshots:
            return []

        _, snapshot = self.snapshots[-1]
        top_stats = snapshot.statistics("lineno")

        results: List[Dict[str, Any]] = []
        for stat in top_stats[:limit]:
            results.append(
                {
                    "filename": stat.traceback.format()[0],
                    "size_mb": stat.size / (1024 * 1024),
                    "count": stat.count,
                }
            )

        return results


class Profiler:
    """Advanced profiling system"""

    def __init__(
        self,
        enable_memory_profiling: bool = True,
        enable_line_profiling: bool = False,
        top_n_functions: int = 20,
    ) -> None:
        self.enable_memory_profiling = enable_memory_profiling
        self.enable_line_profiling = enable_line_profiling
        self.top_n_functions = top_n_functions

        # Storage for profile results
        self.results: List[ProfileResult] = []
        self._lock = threading.Lock()

        # Line profiler instance
        self.line_profiler = (
            line_profiler.LineProfiler() if self.enable_line_profiling else None
        )

    def profile_function(self, name: Optional[str] = None):
        """Decorator for profiling functions"""

        def decorator(func: Callable) -> Callable:
            profile_name = name or f"{func.__module__}.{func.__name__}"

            if asyncio.iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    return await self._profile_async(
                        func, profile_name, *args, **kwargs
                    )

                return async_wrapper
            else:

                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    return self._profile_sync(func, profile_name, *args, **kwargs)

                return sync_wrapper

        return decorator

    async def _profile_async(self, func: Callable, name: str, *args, **kwargs) -> Any:
        """Profile an async function"""
        profiler = cProfile.Profile()

        # Memory tracking
        if self.enable_memory_profiling:
            tracemalloc.start()
            memory_start = self._get_memory_usage()
        else:
            memory_start = 0.0

        start_time = time.time()

        # Run function with profiling
        profiler.enable()
        try:
            result = await func(*args, **kwargs)
        finally:
            profiler.disable()

        end_time = time.time()

        # Memory tracking
        if self.enable_memory_profiling:
            _, peak = tracemalloc.get_traced_memory()
            memory_end = self._get_memory_usage()
            memory_peak = peak / (1024 * 1024)  # Convert to MB
            tracemalloc.stop()
        else:
            memory_end = memory_peak = 0.0

        # Process results
        self._process_profile(
            name, profiler, start_time, end_time, memory_start, memory_peak, memory_end
        )

        return result

    def _profile_sync(self, func: Callable, name: str, *args, **kwargs) -> Any:
        """Profile a sync function"""
        profiler = cProfile.Profile()

        if self.enable_memory_profiling:
            tracemalloc.start()
            memory_start = self._get_memory_usage()
        else:
            memory_start = 0.0

        start_time = time.time()

        profiler.enable()
        try:
            result = func(*args, **kwargs)
        finally:
            profiler.disable()

        end_time = time.time()

        if self.enable_memory_profiling:
            _, peak = tracemalloc.get_traced_memory()
            memory_end = self._get_memory_usage()
            memory_peak = peak / (1024 * 1024)
            tracemalloc.stop()
        else:
            memory_end = memory_peak = 0.0

        self._process_profile(
            name, profiler, start_time, end_time, memory_start, memory_peak, memory_end
        )

        return result

    def _process_profile(
        self,
        name: str,
        profiler: cProfile.Profile,
        start_time: float,
        end_time: float,
        memory_start: float,
        memory_peak: float,
        memory_end: float,
    ) -> None:
        """Process profile results"""

        # Create stats
        stats = pstats.Stats(profiler)
        stats.sort_stats(pstats.SortKey.CUMULATIVE)

        # Get top functions
        top_functions: List[Dict[str, Any]] = []
        stats_dict = stats.stats

        for (filename, line, func_name), (cc, nc, tt, ct, callers) in list(
            stats_dict.items()
        )[: self.top_n_functions]:
            top_functions.append(
                {
                    "function": f"{filename}:{line}({func_name})",
                    "cumulative_time": ct,
                    "total_time": tt,
                    "call_count": nc,
                }
            )

        # Create result
        result = ProfileResult(
            name=name,
            duration_seconds=end_time - start_time,
            cpu_time_seconds=(
                sum(stats_dict.values(), key=lambda x: x[3]) if stats_dict else 0
            ),
            memory_start_mb=memory_start,
            memory_peak_mb=memory_peak,
            memory_end_mb=memory_end,
            top_functions=top_functions,
        )

        with self._lock:
            self.results.append(result)

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0

    def get_results(self, name: Optional[str] = None) -> List[ProfileResult]:
        """Get profiling results"""
        with self._lock:
            if name:
                return [r for r in self.results if r.name == name]
            return self.results.copy()

    def clear_results(self) -> None:
        """Clear stored results"""
        with self._lock:
            self.results.clear()

    def generate_report(self, name: Optional[str] = None) -> str:
        """Generate profiling report"""
        results = self.get_results(name)

        if not results:
            return "No profiling results available"

        report = "Profiling Report\n" + "=" * 50 + "\n\n"

        for result in results:
            report += f"Function: {result.name}\n"
            report += f"Duration: {result.duration_seconds:.4f}s\n"
            report += f"CPU Time: {result.cpu_time_seconds:.4f}s\n"
            report += f"Memory Start: {result.memory_start_mb:.2f}MB\n"
            report += f"Memory Peak: {result.memory_peak_mb:.2f}MB\n"
            report += f"Memory End: {result.memory_end_mb:.2f}MB\n"
            report += "\nTop Functions:\n"

            for func in result.top_functions[:5]:
                report += f"  {func['function']}: {func['cumulative_time']:.4f}s\n"

            report += "\n" + "-" * 50 + "\n\n"

        return report


# Global profiler instance
_profiler: Optional[Profiler] = None


def get_profiler() -> Profiler:
    """Get the global profiler instance"""
    global _profiler
    if _profiler is None:
        _profiler = Profiler()
    return _profiler


# Convenience decorators
def profile_async(name: Optional[str] = None):
    """Decorator for profiling async functions"""
    profiler = get_profiler()
    return profiler.profile_function(name)


def profile_sync(name: Optional[str] = None):
    """Decorator for profiling sync functions"""
    profiler = get_profiler()
    return profiler.profile_function(name)

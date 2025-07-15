"""
Advanced Metrics Collection and Monitoring System

This module provides comprehensive metrics collection with Prometheus integration,
custom metrics storage, and advanced monitoring capabilities.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

# Optional imports
try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class MetricTypes(Enum):
    """Types of metrics"""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricDefinition:
    """Definition of a metric"""

    name: str
    type: MetricTypes
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms
    objectives: Optional[Dict[float, float]] = None


@dataclass
class MetricPoint:
    """Individual metric data point."""

    name: str
    value: Union[int, float]
    timestamp: datetime
    labels: Dict[str, str]
    metadata: Dict[str, Any]


class MetricsCollector:
    """
    Advanced metrics collector with Prometheus integration.

    Features:
    - Prometheus metrics export
    - Custom metrics storage
    - Background collection loops
    - System metrics monitoring
    - Performance tracking
    """

    def __init__(
        self,
        enable_prometheus: bool = True,
        history_size: int = 1000,
        collection_interval: int = 10,
    ):
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.history_size = history_size
        self.collection_interval = collection_interval

        # Prometheus metrics
        if self.enable_prometheus:
            self.registry = CollectorRegistry()
            self.metrics: Dict[str, Any] = {}
            self._initialize_default_metrics()

        # Custom metrics storage
        self.custom_metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)

        # Threading
        self._lock = threading.Lock()
        self._running = False
        self._collection_thread: Optional[threading.Thread] = None

        self.logger = logging.getLogger(__name__)

    def _initialize_default_metrics(self) -> None:
        """Initialize default Prometheus metrics."""
        if not self.enable_prometheus:
            return

        # System metrics
        self.metrics["system_cpu_percent"] = Gauge(
            "system_cpu_percent", "System CPU usage percentage", registry=self.registry
        )

        self.metrics["system_memory_usage_bytes"] = Gauge(
            "system_memory_usage_bytes",
            "System memory usage in bytes",
            registry=self.registry,
        )

        self.metrics["system_disk_usage_percent"] = Gauge(
            "system_disk_usage_percent",
            "System disk usage percentage",
            registry=self.registry,
        )

        # Application metrics
        self.metrics["agent_operations_total"] = Counter(
            "agent_operations_total",
            "Total number of agent operations",
            ["agent_id", "operation", "status"],
            registry=self.registry,
        )

        self.metrics["agent_execution_time"] = Histogram(
            "agent_execution_time_seconds",
            "Agent execution time in seconds",
            ["agent_id", "operation"],
            registry=self.registry,
        )

        self.metrics["system_requests_total"] = Counter(
            "system_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"],
            registry=self.registry,
        )

        self.metrics["system_request_duration"] = Histogram(
            "system_request_duration_seconds",
            "HTTP request duration in seconds",
            ["method", "endpoint"],
            registry=self.registry,
        )

    async def initialize(self) -> None:
        """Initialize the metrics collector."""
        self._running = True

        if self.enable_prometheus and PSUTIL_AVAILABLE:
            self._collection_thread = threading.Thread(
                target=self._collection_loop, daemon=True
            )
            self._collection_thread.start()

        self.logger.info("Metrics collector initialized")

    async def shutdown(self) -> None:
        """Shutdown the metrics collector."""
        self._running = False

        if self._collection_thread:
            self._collection_thread.join(timeout=5)

        self.logger.info("Metrics collector shutdown")

    def increment_counter(
        self,
        name: str,
        value: Union[int, float] = 1,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter metric."""
        with self._lock:
            # Custom counter
            key = f"{name}:{str(sorted(labels or {}).items())}"
            self.counters[key] += value

            # Prometheus counter
            if self.enable_prometheus and name in self.metrics:
                if labels:
                    self.metrics[name].labels(**labels).inc(value)
                else:
                    self.metrics[name].inc(value)

    def set_gauge(
        self,
        name: str,
        value: Union[int, float],
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge metric."""
        with self._lock:
            # Custom gauge
            key = f"{name}:{str(sorted(labels or {}).items())}"
            self.gauges[key] = value

            # Prometheus gauge
            if self.enable_prometheus and name in self.metrics:
                if labels:
                    self.metrics[name].labels(**labels).set(value)
                else:
                    self.metrics[name].set(value)

    def record_histogram(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a histogram metric."""
        with self._lock:
            # Custom histogram
            key = f"{name}:{str(sorted(labels or {}).items())}"
            self.histograms[key].append(value)

            # Keep only recent values
            if len(self.histograms[key]) > self.history_size:
                self.histograms[key] = self.histograms[key][-self.history_size :]

            # Prometheus histogram
            if self.enable_prometheus and name in self.metrics:
                if labels:
                    self.metrics[name].labels(**labels).observe(value)
                else:
                    self.metrics[name].observe(value)

    def record_custom_metric(
        self,
        name: str,
        value: Any,
        labels: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a custom metric."""
        point = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.now(timezone.utc),
            labels=labels or {},
            metadata=metadata or {},
        )

        with self._lock:
            self.custom_metrics[name].append(point)

    def get_counter(
        self, name: str, labels: Optional[Dict[str, str]] = None
    ) -> Union[int, float]:
        """Get counter value."""
        key = f"{name}:{str(sorted(labels or {}).items())}"
        return self.counters.get(key, 0)

    def get_gauge(
        self, name: str, labels: Optional[Dict[str, str]] = None
    ) -> Union[int, float]:
        """Get gauge value."""
        key = f"{name}:{str(sorted(labels or {}).items())}"
        return self.gauges.get(key, 0)

    def get_histogram_stats(
        self, name: str, labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """Get histogram statistics."""
        key = f"{name}:{str(sorted(labels or {}).items())}"
        values = self.histograms.get(key, [])

        if not values:
            return {"count": 0, "sum": 0, "avg": 0, "min": 0, "max": 0}

        return {
            "count": len(values),
            "sum": sum(values),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    def get_metric_history(self, name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get metric history."""
        with self._lock:
            points = list(self.custom_metrics.get(name, []))

        # Return most recent points
        points = points[-limit:] if len(points) > limit else points

        return [
            {
                "value": point.value,
                "timestamp": point.timestamp.isoformat(),
                "labels": point.labels,
                "metadata": point.metadata,
            }
            for point in points
        ]

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        with self._lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {
                    k: self.get_histogram_stats(k.split(":")[0])
                    for k in self.histograms
                },
                "custom_metrics": {k: len(v) for k, v in self.custom_metrics.items()},
            }

    def get_prometheus_metrics(self) -> Optional[str]:
        """Get metrics in Prometheus format."""
        if self.enable_prometheus:
            try:
                return generate_latest(self.registry).decode("utf-8")
            except Exception as e:
                self.logger.error(f"Failed to generate Prometheus metrics: {e}")
        return None

    def _collection_loop(self) -> None:
        """Background collection loop."""
        while self._running:
            try:
                self._collect_system_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                self.logger.error(f"Error in collection loop: {e}")
                time.sleep(self.collection_interval)

    def _collect_system_metrics(self) -> None:
        """Collect system metrics."""
        if not PSUTIL_AVAILABLE:
            return

        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.set_gauge("system_cpu_percent", cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()
            self.set_gauge("system_memory_usage_bytes", memory.used)

            # Disk usage
            disk = psutil.disk_usage("/")
            self.set_gauge("system_disk_usage_percent", disk.percent)

        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")

    def create_timer(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Create a timer context manager for measuring execution time."""
        return MetricTimer(self, name, labels)

    def track_request(
        self, method: str, endpoint: str, status_code: int, duration: float
    ) -> None:
        """Track HTTP request metrics."""
        labels = {
            "method": method,
            "endpoint": endpoint,
            "status": str(status_code),
        }

        self.increment_counter("system_requests_total", labels=labels)
        self.record_histogram(
            "system_request_duration",
            duration,
            labels={"method": method, "endpoint": endpoint},
        )

    def track_agent_operation(
        self, agent_id: str, operation: str, status: str, duration: float
    ) -> None:
        """Track agent operation metrics."""
        labels = {
            "agent_id": agent_id,
            "operation": operation,
            "status": status,
        }

        self.increment_counter("agent_operations_total", labels=labels)
        self.record_histogram(
            "agent_execution_time",
            duration,
            labels={"agent_id": agent_id, "operation": operation},
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            "total_counters": len(self.counters),
            "total_gauges": len(self.gauges),
            "total_histograms": len(self.histograms),
            "total_custom_metrics": len(self.custom_metrics),
            "prometheus_enabled": self.enable_prometheus,
            "collection_interval": self.collection_interval,
            "history_size": self.history_size,
        }


class MetricTimer:
    """Context manager for timing operations."""

    def __init__(
        self,
        collector: MetricsCollector,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ):
        self.collector = collector
        self.name = name
        self.labels = labels
        self.start_time = 0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.collector.record_histogram(self.name, duration, self.labels)


class PrometheusExporter:
    """Prometheus metrics exporter"""

    def __init__(self, collector: MetricsCollector, port: int = 8001):
        self.collector = collector
        self.port = port
        self.server = None

    async def start(self) -> None:
        """Start the metrics HTTP server"""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available, metrics export disabled")
            return

        try:
            from aiohttp import web

            async def metrics_handler(request):
                metrics = self.collector.get_prometheus_metrics()
                if metrics:
                    return web.Response(text=metrics, content_type=CONTENT_TYPE_LATEST)
                else:
                    return web.Response(text="Metrics not available", status=503)

            app = web.Application()
            app.router.add_get("/metrics", metrics_handler)

            runner = web.AppRunner(app)
            await runner.setup()
            self.server = web.TCPSite(runner, "0.0.0.0", self.port)
            await self.server.start()

            logger.info(
                f"Prometheus metrics available at http://0.0.0.0:{self.port}/metrics"
            )
        except Exception as e:
            logger.error(f"Failed to start Prometheus exporter: {e}")

    async def stop(self) -> None:
        """Stop the metrics server"""
        if self.server:
            await self.server.stop()


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


# Convenience functions
def increment_counter(
    name: str, value: Union[int, float] = 1, labels: Optional[Dict[str, str]] = None
) -> None:
    """Increment a counter metric"""
    get_metrics_collector().increment_counter(name, value, labels)


def set_gauge(
    name: str, value: Union[int, float], labels: Optional[Dict[str, str]] = None
) -> None:
    """Set a gauge metric"""
    get_metrics_collector().set_gauge(name, value, labels)


def record_histogram(
    name: str, value: float, labels: Optional[Dict[str, str]] = None
) -> None:
    """Record a histogram metric"""
    get_metrics_collector().record_histogram(name, value, labels)


def create_timer(name: str, labels: Optional[Dict[str, str]] = None) -> MetricTimer:
    """Create a timer context manager"""
    return get_metrics_collector().create_timer(name, labels)


async def time_async_operation(
    operation: Callable, name: str, labels: Optional[Dict[str, str]] = None
) -> Any:
    """Time an async operation"""
    start_time = time.time()
    try:
        result = await operation()
        return result
    finally:
        duration = time.time() - start_time
        get_metrics_collector().record_histogram(name, duration, labels)

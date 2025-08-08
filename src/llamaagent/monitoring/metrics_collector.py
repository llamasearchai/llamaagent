"""
Comprehensive Metrics Collection System for LlamaAgent

This module provides advanced metrics collection capabilities including:
- Prometheus metrics integration
- Custom business metrics
- Performance tracking
- Health monitoring
- Resource utilization tracking

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import psutil

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        Info,
        Summary,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be collected."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    INFO = "info"


@dataclass
class MetricDefinition:
    """Definition of a metric to be collected."""

    name: str
    description: str
    metric_type: MetricType
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms
    quantiles: Optional[Dict[float, float]] = None  # For summaries


class MetricsCollector:
    """
    Comprehensive metrics collector for LlamaAgent system.
    """

    def __init__(
        self, registry: Optional[Any] = None, enable_default_metrics: bool = True
    ):
        self.registry = registry
        self.enable_default_metrics = enable_default_metrics
        self.metrics: Dict[str, Any] = {}
        self.custom_metrics: Dict[str, Any] = {}

        # Initialize Prometheus if available
        if PROMETHEUS_AVAILABLE:
            if self.registry is None:
                self.registry = CollectorRegistry()
            self._initialize_prometheus_metrics()
        else:
            self._initialize_mock_metrics()

    def _initialize_prometheus_metrics(self):
        """Initialize core Prometheus metrics."""
        # HTTP request metrics
        self.http_requests_total = Counter(
            'llamaagent_http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry,
        )
        self.http_request_duration_seconds = Histogram(
            'llamaagent_http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry,
        )
        # Task metrics
        self.tasks_total = Counter(
            'llamaagent_tasks_total',
            'Total tasks submitted',
            ['task_type'],
            registry=self.registry,
        )
        self.tasks_completed_total = Counter(
            'llamaagent_tasks_completed_total',
            'Total tasks completed',
            ['task_type', 'status'],
            registry=self.registry,
        )
        self.tasks_failed_total = Counter(
            'llamaagent_tasks_failed_total',
            'Total tasks failed',
            ['task_type', 'error_type'],
            registry=self.registry,
        )
        self.task_duration_seconds = Histogram(
            'llamaagent_task_duration_seconds',
            'Task execution duration',
            ['task_type', 'agent_type'],
            buckets=[1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0],
            registry=self.registry,
        )
        self.task_queue_length = Gauge(
            'llamaagent_task_queue_length',
            'Current task queue length',
            registry=self.registry,
        )
        # Agent metrics
        self.active_agents = Gauge(
            'llamaagent_active_agents',
            'Number of active agents',
            registry=self.registry,
        )
        self.agent_health_status = Gauge(
            'llamaagent_agent_health_status',
            'Agent health status (1=healthy, 0=unhealthy)',
            ['agent_id', 'agent_type'],
            registry=self.registry,
        )

        self.agent_tasks_processed_total = Counter(
            'llamaagent_agent_tasks_processed_total',
            'Total tasks processed by agent',
            ['agent_id', 'agent_type'],
            registry=self.registry,
        )
        self.agent_response_duration_seconds = Histogram(
            'llamaagent_agent_response_duration_seconds',
            'Agent response duration',
            ['agent_id', 'agent_type'],
            buckets=[1.0, 5.0, 15.0, 30.0, 60.0, 120.0],
            registry=self.registry,
        )
        self.agent_memory_usage_bytes = Gauge(
            'llamaagent_agent_memory_usage_bytes',
            'Agent memory usage in bytes',
            ['agent_id'],
            registry=self.registry,
        )
        self.agent_cpu_usage_percent = Gauge(
            'llamaagent_agent_cpu_usage_percent',
            'Agent CPU usage percentage',
            ['agent_id'],
            registry=self.registry,
        )
        self.agent_task_queue_length = Gauge(
            'llamaagent_agent_task_queue_length',
            'Agent task queue length',
            ['agent_id'],
            registry=self.registry,
        )
        self.agent_success_rate = Gauge(
            'llamaagent_agent_success_rate',
            'Agent success rate percentage',
            ['agent_id'],
            registry=self.registry,
        )
        self.agent_errors_total = Counter(
            'llamaagent_agent_errors_total',
            'Total agent errors',
            ['agent_id', 'error_type'],
            registry=self.registry,
        )
        # LLM provider metrics
        self.llm_requests_total = Counter(
            'llamaagent_llm_requests_total',
            'Total LLM requests',
            ['provider', 'model'],
            registry=self.registry,
        )
        self.llm_request_duration_seconds = Histogram(
            'llamaagent_llm_request_duration_seconds',
            'LLM request duration',
            ['provider', 'model'],
            buckets=[1.0, 5.0, 15.0, 30.0, 60.0, 120.0],
            registry=self.registry,
        )
        self.llm_tokens_used_total = Counter(
            'llamaagent_llm_tokens_used_total',
            'Total LLM tokens used',
            ['provider', 'model', 'token_type'],
            registry=self.registry,
        )
        self.llm_errors_total = Counter(
            'llamaagent_llm_errors_total',
            'Total LLM errors',
            ['provider', 'error_type'],
            registry=self.registry,
        )
        self.llm_provider_health = Gauge(
            'llamaagent_llm_provider_health',
            'LLM provider health status (1=healthy, 0=unhealthy)',
            ['provider'],
            registry=self.registry,
        )

        # Database metrics
        self.db_connections_active = Gauge(
            'llamaagent_db_connections_active',
            'Active database connections',
            registry=self.registry,
        )
        self.db_connections_idle = Gauge(
            'llamaagent_db_connections_idle',
            'Idle database connections',
            registry=self.registry,
        )
        self.db_connections_max = Gauge(
            'llamaagent_db_connections_max',
            'Maximum database connections',
            registry=self.registry,
        )
        self.db_query_duration_seconds = Histogram(
            'llamaagent_db_query_duration_seconds',
            'Database query duration',
            ['operation'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
            registry=self.registry,
        )
        self.db_errors_total = Counter(
            'llamaagent_db_errors_total',
            'Total database errors',
            ['operation', 'error_type'],
            registry=self.registry,
        )
        # Cache metrics
        self.cache_hits_total = Counter(
            'llamaagent_cache_hits_total',
            'Total cache hits',
            ['cache_type'],
            registry=self.registry,
        )
        self.cache_misses_total = Counter(
            'llamaagent_cache_misses_total',
            'Total cache misses',
            ['cache_type'],
            registry=self.registry,
        )
        self.cache_hit_rate = Gauge(
            'llamaagent_cache_hit_rate',
            'Cache hit rate percentage',
            ['cache_type'],
            registry=self.registry,
        )
        # System metrics
        self.system_info = Info(
            'llamaagent_system_info', 'System information', registry=self.registry
        )
        self.process_cpu_usage = Gauge(
            'llamaagent_process_cpu_usage_percent',
            'Process CPU usage percentage',
            registry=self.registry,
        )
        self.process_memory_usage = Gauge(
            'llamaagent_process_memory_usage_bytes',
            'Process memory usage in bytes',
            registry=self.registry,
        )
        # Business metrics
        self.user_sessions_active = Gauge(
            'llamaagent_user_sessions_active',
            'Active user sessions',
            registry=self.registry,
        )
        self.api_rate_limit_exceeded_total = Counter(
            'llamaagent_api_rate_limit_exceeded_total',
            'Total rate limit exceeded events',
            ['endpoint', 'user_id'],
            registry=self.registry,
        )
        # Store all metrics for easy access
        self.metrics.update(
            {
                'http_requests_total': self.http_requests_total,
                'http_request_duration_seconds': self.http_request_duration_seconds,
                'tasks_total': self.tasks_total,
                'tasks_completed_total': self.tasks_completed_total,
                'tasks_failed_total': self.tasks_failed_total,
                'task_duration_seconds': self.task_duration_seconds,
                'task_queue_length': self.task_queue_length,
                'active_agents': self.active_agents,
                'agent_health_status': self.agent_health_status,
                'agent_tasks_processed_total': self.agent_tasks_processed_total,
                'agent_response_duration_seconds': self.agent_response_duration_seconds,
                'agent_memory_usage_bytes': self.agent_memory_usage_bytes,
                'agent_cpu_usage_percent': self.agent_cpu_usage_percent,
                'agent_task_queue_length': self.agent_task_queue_length,
                'agent_success_rate': self.agent_success_rate,
                'agent_errors_total': self.agent_errors_total,
                'llm_requests_total': self.llm_requests_total,
                'llm_request_duration_seconds': self.llm_request_duration_seconds,
                'llm_tokens_used_total': self.llm_tokens_used_total,
                'llm_errors_total': self.llm_errors_total,
                'llm_provider_health': self.llm_provider_health,
                'db_connections_active': self.db_connections_active,
                'db_connections_idle': self.db_connections_idle,
                'db_connections_max': self.db_connections_max,
                'db_query_duration_seconds': self.db_query_duration_seconds,
                'db_errors_total': self.db_errors_total,
                'cache_hits_total': self.cache_hits_total,
                'cache_misses_total': self.cache_misses_total,
                'cache_hit_rate': self.cache_hit_rate,
                'system_info': self.system_info,
                'process_cpu_usage': self.process_cpu_usage,
                'process_memory_usage': self.process_memory_usage,
                'user_sessions_active': self.user_sessions_active,
                'api_rate_limit_exceeded_total': self.api_rate_limit_exceeded_total,
            }
        )
        # Set system info
        self.system_info.info(
            {
                'version': os.getenv('LLAMAAGENT_VERSION', 'unknown'),
                'python_version': f"{psutil.Process().environ.get('PYTHON_VERSION', 'unknown')}",
                'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown',
                'platform': os.uname().sysname if hasattr(os, 'uname') else 'unknown',
            }
        )

    def _initialize_mock_metrics(self):
        """Initialize mock metrics when Prometheus is not available."""

        class MockMetric:
            def inc(self, *args, **kwargs):
                pass

            def set(self, *args, **kwargs):
                pass

            def observe(self, *args, **kwargs):
                pass

            def labels(self, *args, **kwargs):
                return self

            def info(self, *args, **kwargs):
                pass

        # Create mock metrics with the same names
        metric_names = [
            'http_requests_total',
            'http_request_duration_seconds',
            'tasks_total',
            'tasks_completed_total',
            'tasks_failed_total',
            'task_duration_seconds',
            'task_queue_length',
            'active_agents',
            'agent_health_status',
            'agent_tasks_processed_total',
            'agent_response_duration_seconds',
            'agent_memory_usage_bytes',
            'agent_cpu_usage_percent',
            'agent_task_queue_length',
            'agent_success_rate',
            'agent_errors_total',
            'llm_requests_total',
            'llm_request_duration_seconds',
            'llm_tokens_used_total',
            'llm_errors_total',
            'llm_provider_health',
            'db_connections_active',
            'db_connections_idle',
            'db_connections_max',
            'db_query_duration_seconds',
            'db_errors_total',
            'cache_hits_total',
            'cache_misses_total',
            'cache_hit_rate',
            'system_info',
            'process_cpu_usage',
            'process_memory_usage',
            'user_sessions_active',
            'api_rate_limit_exceeded_total',
        ]

        for name in metric_names:
            setattr(self, name, MockMetric())
            self.metrics[name] = getattr(self, name)

    def record_http_request(
        self, method: str, endpoint: str, status_code: int, duration: float
    ):
        """Record HTTP request metrics."""
        self.http_requests_total.labels(
            method=method, endpoint=endpoint, status_code=status_code
        ).inc()

        self.http_request_duration_seconds.labels(
            method=method, endpoint=endpoint
        ).observe(duration)

    def record_task_submission(self, task_type: str):
        """Record task submission."""
        self.tasks_total.labels(task_type=task_type).inc()

    def record_task_completion(
        self, task_type: str, status: str, duration: float, agent_type: str = "unknown"
    ):
        """Record task completion."""
        self.tasks_completed_total.labels(task_type=task_type, status=status).inc()

        self.task_duration_seconds.labels(
            task_type=task_type, agent_type=agent_type
        ).observe(duration)

    def record_task_failure(self, task_type: str, error_type: str):
        """Record task failure."""
        self.tasks_failed_total.labels(task_type=task_type, error_type=error_type).inc()

    def update_task_queue_length(self, length: int):
        """Update task queue length."""
        self.task_queue_length.set(length)

    def update_active_agents(self, count: int):
        """Update active agents count."""
        self.active_agents.set(count)

    def update_agent_health(self, agent_id: str, agent_type: str, is_healthy: bool):
        """Update agent health status."""
        self.agent_health_status.labels(agent_id=agent_id, agent_type=agent_type).set(
            1 if is_healthy else 0
        )

    def record_agent_task_processed(self, agent_id: str, agent_type: str):
        """Record agent task processing."""
        self.agent_tasks_processed_total.labels(
            agent_id=agent_id, agent_type=agent_type
        ).inc()

    def record_agent_response_time(
        self, agent_id: str, agent_type: str, duration: float
    ):
        """Record agent response time."""
        self.agent_response_duration_seconds.labels(
            agent_id=agent_id, agent_type=agent_type
        ).observe(duration)

    def update_agent_resource_usage(
        self, agent_id: str, memory_bytes: int, cpu_percent: float
    ):
        """Update agent resource usage."""
        self.agent_memory_usage_bytes.labels(agent_id=agent_id).set(memory_bytes)
        self.agent_cpu_usage_percent.labels(agent_id=agent_id).set(cpu_percent)

    def update_agent_queue_length(self, agent_id: str, length: int):
        """Update agent task queue length."""
        self.agent_task_queue_length.labels(agent_id=agent_id).set(length)

    def update_agent_success_rate(self, agent_id: str, rate: float):
        """Update agent success rate."""
        self.agent_success_rate.labels(agent_id=agent_id).set(rate)

    def record_agent_error(self, agent_id: str, error_type: str):
        """Record agent error."""
        self.agent_errors_total.labels(agent_id=agent_id, error_type=error_type).inc()

    def record_llm_request(
        self,
        provider: str,
        model: str,
        duration: float,
        tokens_used: int,
        token_type: str = "total",
    ):
        """Record LLM request metrics."""
        self.llm_requests_total.labels(provider=provider, model=model).inc()

        self.llm_request_duration_seconds.labels(
            provider=provider, model=model
        ).observe(duration)
        self.llm_tokens_used_total.labels(
            provider=provider, model=model, token_type=token_type
        ).inc(tokens_used)

    def record_llm_error(self, provider: str, error_type: str):
        """Record LLM error."""
        self.llm_errors_total.labels(provider=provider, error_type=error_type).inc()

    def update_llm_provider_health(self, provider: str, is_healthy: bool):
        """Update LLM provider health."""
        self.llm_provider_health.labels(provider=provider).set(1 if is_healthy else 0)

    def update_database_connections(self, active: int, idle: int, max_connections: int):
        """Update database connection metrics."""
        self.db_connections_active.set(active)
        self.db_connections_idle.set(idle)
        self.db_connections_max.set(max_connections)

    def record_database_query(self, operation: str, duration: float):
        """Record database query metrics."""
        self.db_query_duration_seconds.labels(operation=operation).observe(duration)

    def record_database_error(self, operation: str, error_type: str):
        """Record database error."""
        self.db_errors_total.labels(operation=operation, error_type=error_type).inc()

    def record_cache_hit(self, cache_type: str):
        """Record cache hit."""
        self.cache_hits_total.labels(cache_type=cache_type).inc()

    def record_cache_miss(self, cache_type: str):
        """Record cache miss."""
        self.cache_misses_total.labels(cache_type=cache_type).inc()

    def update_cache_hit_rate(self, cache_type: str, rate: float):
        """Update cache hit rate."""
        self.cache_hit_rate.labels(cache_type=cache_type).set(rate)

    def update_system_metrics(self):
        """Update system resource metrics."""
        try:
            process = psutil.Process()

            # CPU usage
            cpu_percent = process.cpu_percent()
            self.process_cpu_usage.set(cpu_percent)
            # Memory usage
            memory_info = process.memory_info()
            self.process_memory_usage.set(memory_info.rss)
        except Exception as e:
            logger.warning(f"Failed to update system metrics: {e}")

    def update_user_sessions(self, count: int):
        """Update active user sessions count."""
        self.user_sessions_active.set(count)

    def record_rate_limit_exceeded(self, endpoint: str, user_id: str):
        """Record rate limit exceeded event."""
        self.api_rate_limit_exceeded_total.labels(
            endpoint=endpoint, user_id=user_id
        ).inc()

    @asynccontextmanager
    async def time_operation(self, metric_name: str, **labels):
        """Context manager to time operations."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            if metric_name in self.metrics:
                metric = self.metrics[metric_name]
                if labels:
                    metric.labels(**labels).observe(duration)
                else:
                    metric.observe(duration)

    def create_custom_metric(self, definition: MetricDefinition):
        """Create a custom metric."""
        if definition.name in self.custom_metrics:
            return

        if definition.metric_type == MetricType.COUNTER:
            metric = Counter(
                definition.name,
                definition.description,
                definition.labels,
                registry=self.registry,
            )
        elif definition.metric_type == MetricType.GAUGE:
            metric = Gauge(
                definition.name,
                definition.description,
                definition.labels,
                registry=self.registry,
            )
        elif definition.metric_type == MetricType.HISTOGRAM:
            buckets = definition.buckets or [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
            metric = Histogram(
                definition.name,
                definition.description,
                definition.labels,
                buckets=buckets,
                registry=self.registry,
            )
        elif definition.metric_type == MetricType.SUMMARY:
            quantiles = definition.quantiles or {0.5: 0.05, 0.9: 0.01, 0.99: 0.001}
            metric = Summary(
                definition.name,
                definition.description,
                definition.labels,
                quantiles=quantiles,
                registry=self.registry,
            )
        elif definition.metric_type == MetricType.INFO:
            metric = Info(
                definition.name, definition.description, registry=self.registry
            )
        else:
            raise ValueError(f"Unsupported metric type: {definition.metric_type}")
        self.custom_metrics[definition.name] = metric

    def _create_mock_metric(self):
        """Create a mock metric for testing."""

        class MockMetric:
            def inc(self, *args, **kwargs):
                pass

            def set(self, *args, **kwargs):
                pass

            def observe(self, *args, **kwargs):
                pass

            def labels(self, *args, **kwargs):
                return self

            def info(self, *args, **kwargs):
                pass

        return MockMetric()

    def get_custom_metric(self, name: str):
        """Get a custom metric by name."""
        return self.custom_metrics.get(name)

    def export_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        if PROMETHEUS_AVAILABLE and self.registry:
            return generate_latest(self.registry).decode('utf-8')
        else:
            return "# Prometheus not available\n"

    def get_content_type(self) -> str:
        """Get the content type for metrics export."""
        return CONTENT_TYPE_LATEST if PROMETHEUS_AVAILABLE else "text/plain"

    async def start_background_collection(self):
        """Start background metrics collection."""
        asyncio.create_task(self._background_metrics_loop())

    async def _background_metrics_loop(self):
        """Background loop for collecting system metrics."""
        while True:
            try:
                self.update_system_metrics()
                await asyncio.sleep(30)  # Update every 30 seconds
            except Exception as e:
                logger.error(f"Error in background metrics collection: {e}")
                await asyncio.sleep(30)


# Global metrics collector instance
metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return metrics_collector

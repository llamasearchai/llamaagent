"""
Metrics system for tracking routing performance and learning from decisions.
"""

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExecutionRecord:
    """Record of a single task execution."""

    task_id: str
    provider_id: str
    timestamp: datetime
    success: bool
    latency_ms: float
    tokens_used: int = 0
    cost: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingRecord:
    """Record of a routing decision."""

    task_id: str
    provider_id: str
    timestamp: datetime
    routing_time_ms: float
    confidence: float
    strategy: str
    alternative_providers: List[Tuple[str, float]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderMetrics:
    """Aggregated metrics for a provider."""

    provider_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0
    error_types: Dict[str, int] = field(default_factory=dict)
    task_type_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    hourly_requests: Deque[Tuple[datetime, int]] = field(
        default_factory=lambda: deque(maxlen=24)
    )
    recent_latencies: Deque[float] = field(default_factory=lambda: deque(maxlen=100))

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    @property
    def avg_latency(self) -> float:
        """Calculate average latency."""
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests

    @property
    def avg_cost(self) -> float:
        """Calculate average cost per request."""
        if self.total_requests == 0:
            return 0.0
        return self.total_cost / self.total_requests

    @property
    def recent_avg_latency(self) -> float:
        """Calculate recent average latency."""
        if not self.recent_latencies:
            return self.avg_latency
        return sum(self.recent_latencies) / len(self.recent_latencies)

    def get_percentile_latency(self, percentile: int) -> float:
        """Get latency percentile (e.g., p95, p99)."""
        if not self.recent_latencies:
            return 0.0
        return np.percentile(list(self.recent_latencies), percentile)


@dataclass
class RoutingMetrics:
    """Overall routing system metrics."""

    total_routing_decisions: int = 0
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_cost: float = 0.0
    total_tokens: int = 0
    routing_strategy_usage: Dict[str, int] = field(default_factory=dict)
    task_type_distribution: Dict[str, int] = field(default_factory=dict)
    language_distribution: Dict[str, int] = field(default_factory=dict)
    avg_routing_confidence: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    fallback_invocations: int = 0
    ab_test_results: Dict[str, Dict[str, float]] = field(default_factory=dict)


class PerformanceTracker:
    """Track and analyze routing and execution performance."""

    def __init__(self, history_limit: int = 10000, metrics_file: Optional[Path] = None):
        """
        Initialize performance tracker.

        Args:
            history_limit: Maximum number of records to keep in memory
            metrics_file: Optional file to persist metrics
        """
        self.history_limit = history_limit
        self.metrics_file = metrics_file

        # In-memory storage
        self.execution_history: Deque[ExecutionRecord] = deque(maxlen=history_limit)
        self.routing_history: Deque[RoutingRecord] = deque(maxlen=history_limit)
        # Aggregated metrics
        self.provider_metrics: Dict[str, ProviderMetrics] = defaultdict(
            lambda: ProviderMetrics("")
        )
        self.routing_metrics = RoutingMetrics()

        # Real-time tracking
        self.active_requests: Dict[
            str, Tuple[str, float]
        ] = {}  # task_id -> (provider_id, start_time)

        # Load existing metrics if file exists
        if metrics_file and metrics_file.exists():
            self.load_metrics()

    def record_routing_decision(
        self,
        task_id: str,
        provider_id: str,
        routing_time: float,
        confidence: float,
        strategy: str = "unknown",
        alternatives: Optional[List[Tuple[str, float]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a routing decision."""
        record = RoutingRecord(
            task_id=task_id,
            provider_id=provider_id,
            timestamp=datetime.now(),
            routing_time_ms=routing_time * 1000,
            confidence=confidence,
            strategy=strategy,
            alternative_providers=alternatives or [],
            metadata=metadata or {},
        )

        self.routing_history.append(record)
        # Update metrics
        self.routing_metrics.total_routing_decisions += 1
        self.routing_metrics.routing_strategy_usage[strategy] = (
            self.routing_metrics.routing_strategy_usage.get(strategy, 0) + 1
        )

        # Update average confidence (running average)
        n = self.routing_metrics.total_routing_decisions
        self.routing_metrics.avg_routing_confidence = (
            (n - 1) * self.routing_metrics.avg_routing_confidence + confidence
        ) / n

        # Track task characteristics if in metadata
        if metadata:
            if "task_type" in metadata:
                task_type = metadata["task_type"]
                self.routing_metrics.task_type_distribution[task_type] = (
                    self.routing_metrics.task_type_distribution.get(task_type, 0) + 1
                )

            if "languages" in metadata:
                for lang in metadata["languages"]:
                    self.routing_metrics.language_distribution[lang] = (
                        self.routing_metrics.language_distribution.get(lang, 0) + 1
                    )

    def start_execution(self, task_id: str, provider_id: str) -> None:
        """Mark the start of a task execution."""
        self.active_requests[task_id] = (provider_id, time.time())

    def record_execution_result(
        self,
        task_id: str,
        provider_id: str,
        success: bool,
        latency: Optional[float] = None,
        tokens_used: int = 0,
        cost: float = 0.0,
        error: Optional[str] = None,
        attempt_number: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record the result of a task execution."""
        # Calculate latency if not provided
        if latency is None and task_id in self.active_requests:
            start_provider, start_time = self.active_requests[task_id]
            if start_provider == provider_id:
                latency = time.time() - start_time
                del self.active_requests[task_id]

        if latency is None:
            latency = 0.0

        # Create execution record
        record = ExecutionRecord(
            task_id=task_id,
            provider_id=provider_id,
            timestamp=datetime.now(),
            success=success,
            latency_ms=latency * 1000,
            tokens_used=tokens_used,
            cost=cost,
            error=error,
            metadata=metadata or {},
        )

        self.execution_history.append(record)
        # Update provider metrics
        if provider_id not in self.provider_metrics:
            self.provider_metrics[provider_id] = ProviderMetrics(provider_id)
        provider = self.provider_metrics[provider_id]
        provider.total_requests += 1

        if success:
            provider.successful_requests += 1
        else:
            provider.failed_requests += 1
            if error:
                error_type = error.split(":")[0] if ":" in error else "unknown"
                provider.error_types[error_type] = (
                    provider.error_types.get(error_type, 0) + 1
                )

        provider.total_latency_ms += latency * 1000
        provider.total_tokens += tokens_used
        provider.total_cost += cost
        provider.recent_latencies.append(latency * 1000)
        # Track task type performance
        if metadata and "task_type" in metadata:
            task_type = metadata["task_type"]
            if task_type not in provider.task_type_performance:
                provider.task_type_performance[task_type] = {
                    "total": 0,
                    "successful": 0,
                    "avg_latency": 0.0,
                }

            task_perf = provider.task_type_performance[task_type]
            task_perf["total"] += 1
            if success:
                task_perf["successful"] += 1

            # Update average latency for task type
            n = task_perf["total"]
            task_perf["avg_latency"] = (
                (n - 1) * task_perf["avg_latency"] + latency * 1000
            ) / n

        # Update hourly request tracking
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        if provider.hourly_requests and provider.hourly_requests[-1][0] == current_hour:
            provider.hourly_requests[-1] = (
                current_hour,
                provider.hourly_requests[-1][1] + 1,
            )
        else:
            provider.hourly_requests.append((current_hour, 1))
        # Update overall metrics
        self.routing_metrics.total_executions += 1
        if success:
            self.routing_metrics.successful_executions += 1
        else:
            self.routing_metrics.failed_executions += 1
            if attempt_number > 1:
                self.routing_metrics.fallback_invocations += 1

        self.routing_metrics.total_cost += cost
        self.routing_metrics.total_tokens += tokens_used

    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        self.routing_metrics.cache_hits += 1

    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        self.routing_metrics.cache_misses += 1

    def record_ab_test_result(
        self,
        test_name: str,
        variant: str,
        success: bool,
        metric_value: float,
    ) -> None:
        """Record A/B test result."""
        if test_name not in self.routing_metrics.ab_test_results:
            self.routing_metrics.ab_test_results[test_name] = {}

        if variant not in self.routing_metrics.ab_test_results[test_name]:
            self.routing_metrics.ab_test_results[test_name][variant] = {
                "total": 0,
                "successful": 0,
                "avg_metric": 0.0,
            }

        result = self.routing_metrics.ab_test_results[test_name][variant]
        result["total"] += 1
        if success:
            result["successful"] += 1

        # Update average metric
        n = result["total"]
        result["avg_metric"] = ((n - 1) * result["avg_metric"] + metric_value) / n

    def get_provider_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get current metrics for all providers."""
        metrics = {}

        for provider_id, provider in self.provider_metrics.items():
            metrics[provider_id] = {
                "success_rate": provider.success_rate,
                "avg_latency": provider.avg_latency,
                "recent_avg_latency": provider.recent_avg_latency,
                "p95_latency": provider.get_percentile_latency(95),
                "p99_latency": provider.get_percentile_latency(99),
                "avg_cost": provider.avg_cost,
                "total_requests": provider.total_requests,
                "recent_success_rate": self._get_recent_success_rate(provider_id),
            }

        return metrics

    def get_metrics(self) -> RoutingMetrics:
        """Get overall routing metrics."""
        return self.routing_metrics

    def get_provider_performance_by_task_type(
        self,
        provider_id: str,
        task_type: str,
    ) -> Optional[Dict[str, float]]:
        """Get provider performance for specific task type."""
        if provider_id not in self.provider_metrics:
            return None

        provider = self.provider_metrics[provider_id]
        if task_type not in provider.task_type_performance:
            return None

        perf = provider.task_type_performance[task_type]
        return {
            "success_rate": (
                perf["successful"] / perf["total"] if perf["total"] > 0 else 0.0
            ),
            "avg_latency": perf["avg_latency"],
            "total_requests": perf["total"],
        }

    def get_best_providers_for_task_type(
        self,
        task_type: str,
        min_requests: int = 10,
    ) -> List[Tuple[str, float]]:
        """Get best providers for a specific task type."""
        providers_scores = []

        for provider_id, provider in self.provider_metrics.items():
            if task_type in provider.task_type_performance:
                perf = provider.task_type_performance[task_type]
                if perf["total"] >= min_requests:
                    success_rate = perf["successful"] / perf["total"]
                    # Score based on success rate and latency
                    score = (
                        success_rate * 0.7
                        + (1.0 - min(perf["avg_latency"] / 5000, 1.0)) * 0.3
                    )
                    providers_scores.append((provider_id, score))
        # Sort by score descending
        providers_scores.sort(key=lambda x: x[1], reverse=True)
        return providers_scores

    def get_cost_analysis(
        self, time_period: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Analyze costs over time period."""
        if time_period:
            cutoff = datetime.now() - time_period
            relevant_records = [
                r for r in self.execution_history if r.timestamp >= cutoff
            ]
        else:
            relevant_records = list(self.execution_history)
        provider_costs = defaultdict(float)
        provider_tokens = defaultdict(int)
        for record in relevant_records:
            provider_costs[record.provider_id] += record.cost
            provider_tokens[record.provider_id] += record.tokens_used

        total_cost = sum(provider_costs.values())
        total_tokens = sum(provider_tokens.values())
        return {
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "provider_costs": dict(provider_costs),
            "provider_tokens": dict(provider_tokens),
            "cost_per_provider": {
                pid: cost / total_cost if total_cost > 0 else 0.0
                for pid, cost in provider_costs.items()
            },
            "avg_cost_per_request": (
                total_cost / len(relevant_records) if relevant_records else 0.0
            ),
        }

    def get_error_analysis(self) -> Dict[str, Any]:
        """Analyze errors across providers."""
        error_analysis = {}

        for provider_id, provider in self.provider_metrics.items():
            if provider.error_types:
                error_analysis[provider_id] = {
                    "total_errors": provider.failed_requests,
                    "error_rate": 1.0 - provider.success_rate,
                    "error_types": dict(provider.error_types),
                    "most_common_error": max(
                        provider.error_types, key=provider.error_types.get
                    ),
                }

        return error_analysis

    def get_load_distribution(self) -> Dict[str, float]:
        """Get load distribution across providers."""
        total_requests = sum(p.total_requests for p in self.provider_metrics.values())
        if total_requests == 0:
            return {}

        return {
            provider_id: provider.total_requests / total_requests
            for provider_id, provider in self.provider_metrics.items()
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.execution_history.clear()
        self.routing_history.clear()
        self.provider_metrics.clear()
        self.routing_metrics = RoutingMetrics()
        self.active_requests.clear()
        logger.info("Reset all metrics")

    def save_metrics(self, path: Optional[Path] = None) -> None:
        """Save metrics to file."""
        save_path = path or self.metrics_file
        if not save_path:
            return

        # Prepare data for serialization
        data = {
            "routing_metrics": {
                "total_routing_decisions": self.routing_metrics.total_routing_decisions,
                "total_executions": self.routing_metrics.total_executions,
                "successful_executions": self.routing_metrics.successful_executions,
                "failed_executions": self.routing_metrics.failed_executions,
                "total_cost": self.routing_metrics.total_cost,
                "total_tokens": self.routing_metrics.total_tokens,
                "routing_strategy_usage": dict(
                    self.routing_metrics.routing_strategy_usage
                ),
                "task_type_distribution": dict(
                    self.routing_metrics.task_type_distribution
                ),
                "language_distribution": dict(
                    self.routing_metrics.language_distribution
                ),
                "avg_routing_confidence": self.routing_metrics.avg_routing_confidence,
                "cache_hits": self.routing_metrics.cache_hits,
                "cache_misses": self.routing_metrics.cache_misses,
                "fallback_invocations": self.routing_metrics.fallback_invocations,
                "ab_test_results": dict(self.routing_metrics.ab_test_results),
            },
            "provider_metrics": {},
            "timestamp": datetime.now().isoformat(),
        }

        # Add provider metrics
        for provider_id, provider in self.provider_metrics.items():
            data["provider_metrics"][provider_id] = {
                "total_requests": provider.total_requests,
                "successful_requests": provider.successful_requests,
                "failed_requests": provider.failed_requests,
                "total_latency_ms": provider.total_latency_ms,
                "total_tokens": provider.total_tokens,
                "total_cost": provider.total_cost,
                "error_types": dict(provider.error_types),
                "task_type_performance": dict(provider.task_type_performance),
            }

        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved metrics to {save_path}")

    def load_metrics(self, path: Optional[Path] = None) -> None:
        """Load metrics from file."""
        load_path = path or self.metrics_file
        if not load_path or not load_path.exists():
            return

        try:
            with open(load_path, "r") as f:
                data = json.load(f)
            # Load routing metrics
            rm = data.get("routing_metrics", {})
            self.routing_metrics.total_routing_decisions = rm.get(
                "total_routing_decisions", 0
            )
            self.routing_metrics.total_executions = rm.get("total_executions", 0)
            self.routing_metrics.successful_executions = rm.get(
                "successful_executions", 0
            )
            self.routing_metrics.failed_executions = rm.get("failed_executions", 0)
            self.routing_metrics.total_cost = rm.get("total_cost", 0.0)
            self.routing_metrics.total_tokens = rm.get("total_tokens", 0)
            self.routing_metrics.routing_strategy_usage = rm.get(
                "routing_strategy_usage", {}
            )
            self.routing_metrics.task_type_distribution = rm.get(
                "task_type_distribution", {}
            )
            self.routing_metrics.language_distribution = rm.get(
                "language_distribution", {}
            )
            self.routing_metrics.avg_routing_confidence = rm.get(
                "avg_routing_confidence", 0.0
            )
            self.routing_metrics.cache_hits = rm.get("cache_hits", 0)
            self.routing_metrics.cache_misses = rm.get("cache_misses", 0)
            self.routing_metrics.fallback_invocations = rm.get(
                "fallback_invocations", 0
            )
            self.routing_metrics.ab_test_results = rm.get("ab_test_results", {})
            # Load provider metrics
            for provider_id, pm in data.get("provider_metrics", {}).items():
                provider = ProviderMetrics(provider_id)
                provider.total_requests = pm.get("total_requests", 0)
                provider.successful_requests = pm.get("successful_requests", 0)
                provider.failed_requests = pm.get("failed_requests", 0)
                provider.total_latency_ms = pm.get("total_latency_ms", 0.0)
                provider.total_tokens = pm.get("total_tokens", 0)
                provider.total_cost = pm.get("total_cost", 0.0)
                provider.error_types = pm.get("error_types", {})
                provider.task_type_performance = pm.get("task_type_performance", {})
                self.provider_metrics[provider_id] = provider

            logger.info(f"Loaded metrics from {load_path}")
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")

    def _get_recent_success_rate(
        self, provider_id: str, window_size: int = 20
    ) -> float:
        """Calculate recent success rate for a provider."""
        recent_executions = [
            r
            for r in list(self.execution_history)[-window_size:]
            if r.provider_id == provider_id
        ]

        if not recent_executions:
            return self.provider_metrics[provider_id].success_rate

        successful = sum(1 for r in recent_executions if r.success)
        return successful / len(recent_executions)

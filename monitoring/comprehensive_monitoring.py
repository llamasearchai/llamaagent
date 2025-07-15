#!/usr/bin/env python3
"""
Comprehensive Monitoring System for LlamaAgent

This module provides advanced monitoring capabilities including:
- Real-time performance metrics
- Health monitoring and alerting
- Business metrics tracking
- Custom dashboard generation
- Automated anomaly detection

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import statistics
import psutil
import os
from enum import Enum

# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Alert:
    """Alert data structure."""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    timestamp: datetime
    metric_name: str
    current_value: float
    threshold: float
    tags: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class HealthCheck:
    """Health check configuration."""
    name: str
    check_function: Callable[[], bool]
    interval: int = 30  # seconds
    timeout: int = 10   # seconds
    enabled: bool = True
    last_check: Optional[datetime] = None
    last_result: Optional[bool] = None
    consecutive_failures: int = 0


@dataclass
class MetricDefinition:
    """Metric definition for monitoring."""
    name: str
    metric_type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms
    quantiles: Optional[List[float]] = None  # For summaries


class MetricsCollector:
    """Advanced metrics collection system."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self.metrics: Dict[str, Any] = {}
        self.custom_metrics: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.RLock()
        
        # Initialize core metrics
        self._initialize_core_metrics()
    
    def _initialize_core_metrics(self):
        """Initialize core system metrics."""
        
        # Request metrics
        self.request_count = Counter(
            'llamaagent_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'llamaagent_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        # Agent metrics
        self.agent_executions = Counter(
            'llamaagent_agent_executions_total',
            'Total agent executions',
            ['agent_type', 'status'],
            registry=self.registry
        )
        
        self.agent_execution_time = Histogram(
            'llamaagent_agent_execution_seconds',
            'Agent execution time in seconds',
            ['agent_type'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )
        
        # LLM metrics
        self.llm_calls = Counter(
            'llamaagent_llm_calls_total',
            'Total LLM API calls',
            ['provider', 'model', 'status'],
            registry=self.registry
        )
        
        self.llm_tokens = Counter(
            'llamaagent_llm_tokens_total',
            'Total LLM tokens used',
            ['provider', 'model', 'type'],
            registry=self.registry
        )
        
        self.llm_response_time = Histogram(
            'llamaagent_llm_response_seconds',
            'LLM response time in seconds',
            ['provider', 'model'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
            registry=self.registry
        )
        
        # System metrics
        self.active_connections = Gauge(
            'llamaagent_active_connections',
            'Number of active connections',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'llamaagent_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.cpu_usage = Gauge(
            'llamaagent_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        # Business metrics
        self.benchmark_success_rate = Gauge(
            'llamaagent_benchmark_success_rate',
            'Benchmark success rate',
            ['benchmark_type'],
            registry=self.registry
        )
        
        self.cache_operations = Counter(
            'llamaagent_cache_operations_total',
            'Cache operations',
            ['operation', 'result'],
            registry=self.registry
        )
        
        # Error metrics
        self.errors = Counter(
            'llamaagent_errors_total',
            'Total errors',
            ['component', 'error_type'],
            registry=self.registry
        )
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics."""
        self.request_count.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_agent_execution(self, agent_type: str, status: str, duration: float):
        """Record agent execution metrics."""
        self.agent_executions.labels(agent_type=agent_type, status=status).inc()
        self.agent_execution_time.labels(agent_type=agent_type).observe(duration)
    
    def record_llm_call(self, provider: str, model: str, status: str, tokens: int, duration: float, token_type: str = "total"):
        """Record LLM call metrics."""
        self.llm_calls.labels(provider=provider, model=model, status=status).inc()
        self.llm_tokens.labels(provider=provider, model=model, type=token_type).inc(tokens)
        self.llm_response_time.labels(provider=provider, model=model).observe(duration)
    
    def record_benchmark_result(self, benchmark_type: str, success_rate: float):
        """Record benchmark results."""
        self.benchmark_success_rate.labels(benchmark_type=benchmark_type).set(success_rate)
    
    def record_cache_operation(self, operation: str, result: str):
        """Record cache operation."""
        self.cache_operations.labels(operation=operation, result=result).inc()
    
    def record_error(self, component: str, error_type: str):
        """Record error occurrence."""
        self.errors.labels(component=component, error_type=error_type).inc()
    
    def update_system_metrics(self):
        """Update system resource metrics."""
        # Memory usage
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        self.memory_usage.set(memory_info.rss)
        
        # CPU usage
        cpu_percent = process.cpu_percent()
        self.cpu_usage.set(cpu_percent)
    
    def record_custom_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record custom metric value."""
        with self._lock:
            metric_key = f"{name}:{json.dumps(labels or {}, sort_keys=True)}"
            self.custom_metrics[metric_key].append(value)
            
            # Keep only recent values (last 1000)
            if len(self.custom_metrics[metric_key]) > 1000:
                self.custom_metrics[metric_key] = self.custom_metrics[metric_key][-1000:]
    
    def get_custom_metric_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for custom metric."""
        with self._lock:
            all_values = []
            for key, values in self.custom_metrics.items():
                if key.startswith(f"{name}:"):
                    all_values.extend(values)
            
            if not all_values:
                return {}
            
            return {
                'count': len(all_values),
                'min': min(all_values),
                'max': max(all_values),
                'mean': statistics.mean(all_values),
                'median': statistics.median(all_values),
                'std_dev': statistics.stdev(all_values) if len(all_values) > 1 else 0.0
            }
    
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        return generate_latest(self.registry)


class HealthMonitor:
    """System health monitoring."""
    
    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_history: deque = deque(maxlen=1000)
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    def register_health_check(self, health_check: HealthCheck):
        """Register a health check."""
        self.health_checks[health_check.name] = health_check
        logger.info(f"Registered health check: {health_check.name}")
    
    def unregister_health_check(self, name: str):
        """Unregister a health check."""
        if name in self.health_checks:
            del self.health_checks[name]
            logger.info(f"Unregistered health check: {name}")
    
    async def start_monitoring(self):
        """Start health monitoring."""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("Health monitoring started")
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                await self._run_health_checks()
                await asyncio.sleep(5)  # Check every 5 seconds
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def _run_health_checks(self):
        """Run all enabled health checks."""
        current_time = datetime.now()
        
        for name, check in self.health_checks.items():
            if not check.enabled:
                continue
            
            # Check if it's time to run this check
            if (check.last_check and 
                (current_time - check.last_check).total_seconds() < check.interval):
                continue
            
            try:
                # Run health check with timeout
                result = await asyncio.wait_for(
                    asyncio.create_task(self._run_single_check(check)),
                    timeout=check.timeout
                )
                
                check.last_check = current_time
                check.last_result = result
                
                if result:
                    check.consecutive_failures = 0
                else:
                    check.consecutive_failures += 1
                
                # Record health check result
                self.health_history.append({
                    'timestamp': current_time,
                    'check_name': name,
                    'result': result,
                    'consecutive_failures': check.consecutive_failures
                })
                
            except asyncio.TimeoutError:
                logger.warning(f"Health check {name} timed out")
                check.last_result = False
                check.consecutive_failures += 1
            except Exception as e:
                logger.error(f"Health check {name} failed with error: {e}")
                check.last_result = False
                check.consecutive_failures += 1
    
    async def _run_single_check(self, check: HealthCheck) -> bool:
        """Run a single health check."""
        if asyncio.iscoroutinefunction(check.check_function):
            return await check.check_function()
        else:
            # Run sync function in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, check.check_function)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        healthy_checks = 0
        total_checks = 0
        failing_checks = []
        
        for name, check in self.health_checks.items():
            if not check.enabled:
                continue
            
            total_checks += 1
            if check.last_result:
                healthy_checks += 1
            else:
                failing_checks.append({
                    'name': name,
                    'consecutive_failures': check.consecutive_failures,
                    'last_check': check.last_check
                })
        
        overall_healthy = len(failing_checks) == 0
        
        return {
            'healthy': overall_healthy,
            'total_checks': total_checks,
            'healthy_checks': healthy_checks,
            'failing_checks': failing_checks,
            'health_score': healthy_checks / total_checks if total_checks > 0 else 1.0
        }


class AlertManager:
    """Alert management system."""
    
    def __init__(self):
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_handlers: List[Callable[[Alert], None]] = []
    
    def add_alert_rule(self, name: str, metric_name: str, condition: str, threshold: float, severity: AlertSeverity):
        """Add an alert rule."""
        self.alert_rules[name] = {
            'metric_name': metric_name,
            'condition': condition,  # 'gt', 'lt', 'eq'
            'threshold': threshold,
            'severity': severity
        }
        logger.info(f"Added alert rule: {name}")
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add an alert handler function."""
        self.alert_handlers.append(handler)
    
    def check_alerts(self, metrics: Dict[str, float]):
        """Check metrics against alert rules."""
        current_time = datetime.now()
        
        for rule_name, rule in self.alert_rules.items():
            metric_name = rule['metric_name']
            
            if metric_name not in metrics:
                continue
            
            current_value = metrics[metric_name]
            threshold = rule['threshold']
            condition = rule['condition']
            
            # Check condition
            triggered = False
            if condition == 'gt' and current_value > threshold:
                triggered = True
            elif condition == 'lt' and current_value < threshold:
                triggered = True
            elif condition == 'eq' and abs(current_value - threshold) < 0.001:
                triggered = True
            
            alert_id = f"{rule_name}:{metric_name}"
            
            if triggered:
                # Create or update alert
                if alert_id not in self.active_alerts:
                    alert = Alert(
                        id=alert_id,
                        title=f"Alert: {rule_name}",
                        description=f"Metric {metric_name} is {current_value}, threshold is {threshold}",
                        severity=rule['severity'],
                        timestamp=current_time,
                        metric_name=metric_name,
                        current_value=current_value,
                        threshold=threshold
                    )
                    
                    self.active_alerts[alert_id] = alert
                    self.alert_history.append(alert)
                    
                    # Notify handlers
                    for handler in self.alert_handlers:
                        try:
                            handler(alert)
                        except Exception as e:
                            logger.error(f"Error in alert handler: {e}")
                    
                    logger.warning(f"Alert triggered: {alert.title}")
            else:
                # Resolve alert if it exists
                if alert_id in self.active_alerts:
                    alert = self.active_alerts[alert_id]
                    alert.resolved = True
                    alert.resolved_at = current_time
                    
                    del self.active_alerts[alert_id]
                    logger.info(f"Alert resolved: {alert.title}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary."""
        active_alerts = list(self.active_alerts.values())
        
        severity_counts = defaultdict(int)
        for alert in active_alerts:
            severity_counts[alert.severity.value] += 1
        
        return {
            'total_active': len(active_alerts),
            'by_severity': dict(severity_counts),
            'recent_alerts': [asdict(alert) for alert in list(self.alert_history)[-10:]]
        }


class ComprehensiveMonitor:
    """Main monitoring coordinator."""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.health_monitor = HealthMonitor()
        self.alert_manager = AlertManager()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Initialize default health checks
        self._setup_default_health_checks()
        
        # Initialize default alert rules
        self._setup_default_alert_rules()
    
    def _setup_default_health_checks(self):
        """Setup default health checks."""
        
        def check_memory_usage():
            """Check if memory usage is reasonable."""
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            return memory_mb < 1000  # Less than 1GB
        
        def check_cpu_usage():
            """Check if CPU usage is reasonable."""
            process = psutil.Process(os.getpid())
            cpu_percent = process.cpu_percent()
            return cpu_percent < 80  # Less than 80%
        
        def check_disk_space():
            """Check if disk space is available."""
            disk_usage = psutil.disk_usage('/')
            free_percent = (disk_usage.free / disk_usage.total) * 100
            return free_percent > 10  # More than 10% free
        
        # Register health checks
        self.health_monitor.register_health_check(
            HealthCheck("memory_usage", check_memory_usage, interval=30)
        )
        self.health_monitor.register_health_check(
            HealthCheck("cpu_usage", check_cpu_usage, interval=30)
        )
        self.health_monitor.register_health_check(
            HealthCheck("disk_space", check_disk_space, interval=60)
        )
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules."""
        
        # Memory usage alert
        self.alert_manager.add_alert_rule(
            "high_memory_usage",
            "memory_usage_mb",
            "gt",
            800,  # 800MB
            AlertSeverity.WARNING
        )
        
        # CPU usage alert
        self.alert_manager.add_alert_rule(
            "high_cpu_usage",
            "cpu_usage_percent",
            "gt",
            75,  # 75%
            AlertSeverity.WARNING
        )
        
        # Error rate alert
        self.alert_manager.add_alert_rule(
            "high_error_rate",
            "error_rate_percent",
            "gt",
            5,  # 5%
            AlertSeverity.ERROR
        )
        
        # Response time alert
        self.alert_manager.add_alert_rule(
            "slow_response_time",
            "avg_response_time_seconds",
            "gt",
            2.0,  # 2 seconds
            AlertSeverity.WARNING
        )
    
    async def start_monitoring(self):
        """Start comprehensive monitoring."""
        if self._running:
            return
        
        self._running = True
        
        # Start health monitoring
        await self.health_monitor.start_monitoring()
        
        # Start metrics collection loop
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Comprehensive monitoring started")
    
    async def stop_monitoring(self):
        """Stop comprehensive monitoring."""
        self._running = False
        
        # Stop health monitoring
        await self.health_monitor.stop_monitoring()
        
        # Stop metrics collection
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Comprehensive monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                # Update system metrics
                self.metrics_collector.update_system_metrics()
                
                # Collect current metrics for alerting
                current_metrics = self._collect_current_metrics()
                
                # Check alerts
                self.alert_manager.check_alerts(current_metrics)
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)
    
    def _collect_current_metrics(self) -> Dict[str, float]:
        """Collect current metric values for alerting."""
        metrics = {}
        
        # System metrics
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        metrics['memory_usage_mb'] = memory_info.rss / 1024 / 1024
        metrics['cpu_usage_percent'] = process.cpu_percent()
        
        # Add custom metrics
        for metric_name in ['error_rate_percent', 'avg_response_time_seconds']:
            stats = self.metrics_collector.get_custom_metric_stats(metric_name)
            if stats:
                metrics[metric_name] = stats.get('mean', 0.0)
        
        return metrics
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        return {
            'health_status': self.health_monitor.get_health_status(),
            'alert_summary': self.alert_manager.get_alert_summary(),
            'system_metrics': self._collect_current_metrics(),
            'custom_metrics': {
                name: self.metrics_collector.get_custom_metric_stats(name)
                for name in ['response_time', 'throughput', 'success_rate']
            }
        }
    
    def export_metrics(self) -> str:
        """Export all metrics in Prometheus format."""
        return self.metrics_collector.export_metrics()


# Global monitoring instance
_global_monitor: Optional[ComprehensiveMonitor] = None


def get_global_monitor() -> ComprehensiveMonitor:
    """Get or create global monitoring instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = ComprehensiveMonitor()
    return _global_monitor


# Decorator for automatic monitoring
def monitor_performance(metric_name: Optional[str] = None):
    """Decorator for automatic performance monitoring."""
    def decorator(func: Callable) -> Callable:
        monitor = get_global_monitor()
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            function_name = metric_name or func.__name__
            
            try:
                result = await func(*args, **kwargs)
                status = "success"
                return result
            except Exception as e:
                status = "error"
                monitor.metrics_collector.record_error(func.__module__, type(e).__name__)
                raise
            finally:
                duration = time.time() - start_time
                monitor.metrics_collector.record_custom_metric(
                    f"{function_name}_duration", duration
                )
                monitor.metrics_collector.record_custom_metric(
                    f"{function_name}_status", 1 if status == "success" else 0
                )
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            function_name = metric_name or func.__name__
            
            try:
                result = func(*args, **kwargs)
                status = "success"
                return result
            except Exception as e:
                status = "error"
                monitor.metrics_collector.record_error(func.__module__, type(e).__name__)
                raise
            finally:
                duration = time.time() - start_time
                monitor.metrics_collector.record_custom_metric(
                    f"{function_name}_duration", duration
                )
                monitor.metrics_collector.record_custom_metric(
                    f"{function_name}_status", 1 if status == "success" else 0
                )
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Example usage
async def demonstrate_monitoring():
    """Demonstrate monitoring capabilities."""
    
    print("RESULTS Comprehensive Monitoring System Demo")
    print("=" * 50)
    
    # Initialize monitoring
    monitor = ComprehensiveMonitor()
    
    # Add custom alert handler
    def alert_handler(alert: Alert):
        print(f"URGENT ALERT: {alert.title} - {alert.description}")
    
    monitor.alert_manager.add_alert_handler(alert_handler)
    
    # Start monitoring
    await monitor.start_monitoring()
    
    print("PASS Monitoring started")
    
    # Simulate some metrics
    for i in range(10):
        # Record some sample metrics
        monitor.metrics_collector.record_custom_metric("response_time", 0.1 + i * 0.05)
        monitor.metrics_collector.record_custom_metric("throughput", 100 - i * 5)
        monitor.metrics_collector.record_custom_metric("success_rate", 95 + i)
        
        await asyncio.sleep(1)
    
    # Get dashboard data
    dashboard_data = monitor.get_dashboard_data()
    
    print("\nPerformance Dashboard Data:")
    print(f"Health Status: {dashboard_data['health_status']['healthy']}")
    print(f"Active Alerts: {dashboard_data['alert_summary']['total_active']}")
    print(f"System Metrics: {dashboard_data['system_metrics']}")
    
    # Export metrics
    metrics_output = monitor.export_metrics()
    newline_count = len(metrics_output.split('\n'))
    print(f"\nRESULTS Exported {newline_count} metric lines")
    
    # Stop monitoring
    await monitor.stop_monitoring()
    print("PASS Monitoring stopped")


if __name__ == "__main__":
    import functools
    asyncio.run(demonstrate_monitoring()) 
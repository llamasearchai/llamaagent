"""
Health Check System for LlamaAgent

This module provides comprehensive health monitoring for all system components.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    component: str
    status: HealthStatus
    message: str
    timestamp: datetime
    response_time_ms: float
    metadata: Dict[str, Any]


class HealthChecker:
    """Comprehensive health checker for LlamaAgent system."""

    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.logger = logging.getLogger(__name__)

        # Health check results
        self.last_results: Dict[str, HealthCheckResult] = {}
        self.health_history: List[HealthCheckResult] = []

        # Component registry
        self.components: Dict[str, Any] = {}
        self.check_functions: Dict[str, Any] = {}

        # Background monitoring
        self._monitor_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

    async def initialize(self) -> None:
        """Initialize the health checker."""
        self.logger.info("Initializing Health Checker")

        # Register default health checks
        self._register_default_checks()

        # Start background monitoring
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def shutdown(self) -> None:
        """Shutdown the health checker."""
        self.logger.info("Shutting down Health Checker")
        self._shutdown_event.set()

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    def register_component(
        self, name: str, component: Any, check_func: Optional[Any] = None
    ) -> None:
        """Register a component for health checking."""
        self.components[name] = component
        if check_func:
            self.check_functions[name] = check_func
        self.logger.info(f"Registered component for health checking: {name}")

    def unregister_component(self, name: str) -> None:
        """Unregister a component."""
        self.components.pop(name, None)
        self.check_functions.pop(name, None)
        self.last_results.pop(name, None)
        self.logger.info(f"Unregistered component: {name}")

    async def check_health(
        self, component_name: Optional[str] = None
    ) -> Dict[str, HealthCheckResult]:
        """Perform health checks on specified component or all components."""
        if component_name:
            if component_name in self.components:
                result = await self._check_component(component_name)
                return {component_name: result}
            else:
                return {}
        else:
            results = {}
            for name in self.components:
                result = await self._check_component(name)
                results[name] = result
            return results

    async def _check_component(self, name: str) -> HealthCheckResult:
        """Check health of a specific component."""
        start_time = time.time()

        try:
            component = self.components[name]
            check_func = self.check_functions.get(name)

            if check_func:
                # Use custom check function
                status, message, metadata = await self._call_check_function(
                    check_func, component
                )
            else:
                # Use default check
                status, message, metadata = await self._default_health_check(
                    name, component
                )

            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            result = HealthCheckResult(
                component=name,
                status=status,
                message=message,
                timestamp=datetime.now(timezone.utc),
                response_time_ms=response_time,
                metadata=metadata,
            )

            # Store result
            self.last_results[name] = result
            self.health_history.append(result)

            # Keep history size manageable
            if len(self.health_history) > 1000:
                self.health_history = self.health_history[-500:]

            return result

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            result = HealthCheckResult(
                component=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                timestamp=datetime.now(timezone.utc),
                response_time_ms=response_time,
                metadata={"error": str(e)},
            )

            self.last_results[name] = result
            self.health_history.append(result)
            self.logger.error(f"Health check failed for {name}: {e}")

            return result

    async def _call_check_function(self, check_func: Any, component: Any) -> tuple:
        """Call a custom health check function."""
        if asyncio.iscoroutinefunction(check_func):
            return await check_func(component)
        else:
            return check_func(component)

    async def _default_health_check(self, name: str, component: Any) -> tuple:
        """Default health check implementation."""
        try:
            # Check if component has a health check method
            if hasattr(component, "health_check"):
                if asyncio.iscoroutinefunction(component.health_check):
                    result = await component.health_check()
                else:
                    result = component.health_check()

                if isinstance(result, bool):
                    status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                    message = "OK" if result else "Health check failed"
                    metadata = {}
                elif isinstance(result, dict):
                    status = HealthStatus(result.get("status", "healthy"))
                    message = result.get("message", "OK")
                    metadata = result.get("metadata", {})
                else:
                    status = HealthStatus.HEALTHY
                    message = str(result)
                    metadata = {}

                return status, message, metadata

            # Basic connectivity check
            if hasattr(component, "ping"):
                if asyncio.iscoroutinefunction(component.ping):
                    result = await component.ping()
                else:
                    result = component.ping()

                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                message = "Ping successful" if result else "Ping failed"
                return status, message, {}

            # If no specific health check, assume healthy if object exists
            return HealthStatus.HEALTHY, "Component exists", {}

        except Exception as e:
            return HealthStatus.UNHEALTHY, f"Error: {str(e)}", {"error": str(e)}

    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                await self.check_health()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Error in health monitor loop: {e}")
                await asyncio.sleep(5)  # Short delay before retry

    def _register_default_checks(self) -> None:
        """Register default health checks."""

        # Database health check
        async def check_database(db_component: Any) -> tuple:
            try:
                if hasattr(db_component, "execute"):
                    # Try a simple query
                    result = await db_component.execute("SELECT 1")
                    return (
                        HealthStatus.HEALTHY,
                        "Database query successful",
                        {"result": result},
                    )
                else:
                    return HealthStatus.HEALTHY, "Database component available", {}
            except Exception as e:
                return (
                    HealthStatus.UNHEALTHY,
                    f"Database error: {str(e)}",
                    {"error": str(e)},
                )

        # Redis health check
        async def check_redis(redis_component: Any) -> tuple:
            try:
                if hasattr(redis_component, "ping"):
                    result = await redis_component.ping()
                    return (
                        HealthStatus.HEALTHY,
                        "Redis ping successful",
                        {"ping": result},
                    )
                else:
                    return HealthStatus.HEALTHY, "Redis component available", {}
            except Exception as e:
                return (
                    HealthStatus.UNHEALTHY,
                    f"Redis error: {str(e)}",
                    {"error": str(e)},
                )

        # LLM Provider health check
        async def check_llm_provider(llm_component: Any) -> tuple:
            try:
                if hasattr(llm_component, "health_check"):
                    result = await llm_component.health_check()
                    return (
                        HealthStatus.HEALTHY,
                        "LLM provider healthy",
                        {"result": result},
                    )
                else:
                    return HealthStatus.HEALTHY, "LLM provider available", {}
            except Exception as e:
                return (
                    HealthStatus.UNHEALTHY,
                    f"LLM provider error: {str(e)}",
                    {"error": str(e)},
                )

        # Store default check functions
        self.check_functions.update(
            {
                "database": check_database,
                "redis": check_redis,
                "llm_provider": check_llm_provider,
            }
        )

    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        if not self.last_results:
            return {
                "status": HealthStatus.UNKNOWN.value,
                "message": "No health checks performed yet",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # Determine overall status
        statuses = [result.status for result in self.last_results.values()]

        if all(status == HealthStatus.HEALTHY for status in statuses):
            overall_status = HealthStatus.HEALTHY
            message = "All components healthy"
        elif any(status == HealthStatus.UNHEALTHY for status in statuses):
            overall_status = HealthStatus.UNHEALTHY
            unhealthy_components = [
                name
                for name, result in self.last_results.items()
                if result.status == HealthStatus.UNHEALTHY
            ]
            message = f"Unhealthy components: {', '.join(unhealthy_components)}"
        elif any(status == HealthStatus.DEGRADED for status in statuses):
            overall_status = HealthStatus.DEGRADED
            degraded_components = [
                name
                for name, result in self.last_results.items()
                if result.status == HealthStatus.DEGRADED
            ]
            message = f"Degraded components: {', '.join(degraded_components)}"
        else:
            overall_status = HealthStatus.UNKNOWN
            message = "System status unknown"

        return {
            "status": overall_status.value,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "component_count": len(self.last_results),
            "components": {
                name: {
                    "status": result.status.value,
                    "message": result.message,
                    "response_time_ms": result.response_time_ms,
                    "last_checked": result.timestamp.isoformat(),
                }
                for name, result in self.last_results.items()
            },
        }

    def get_component_health(self, component_name: str) -> Optional[Dict[str, Any]]:
        """Get health status of a specific component."""
        result = self.last_results.get(component_name)
        if not result:
            return None

        return {
            "status": result.status.value,
            "message": result.message,
            "response_time_ms": result.response_time_ms,
            "last_checked": result.timestamp.isoformat(),
            "metadata": result.metadata,
        }

    def get_health_history(
        self, component_name: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get health check history."""
        if component_name:
            history = [
                {
                    "component": result.component,
                    "status": result.status.value,
                    "message": result.message,
                    "response_time_ms": result.response_time_ms,
                    "timestamp": result.timestamp.isoformat(),
                    "metadata": result.metadata,
                }
                for result in self.health_history
                if result.component == component_name
            ]
        else:
            history = [
                {
                    "component": result.component,
                    "status": result.status.value,
                    "message": result.message,
                    "response_time_ms": result.response_time_ms,
                    "timestamp": result.timestamp.isoformat(),
                    "metadata": result.metadata,
                }
                for result in self.health_history
            ]

        return history[-limit:]

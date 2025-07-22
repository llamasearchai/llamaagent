"""
Service Mesh Implementation for LlamaAgent

This module provides service discovery, load balancing, and circuit breaker
functionality for distributed services.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

try:
    import redis
except ImportError:
    redis = None

try:
    from opentelemetry import trace

    tracer = trace.get_tracer(__name__)
except ImportError:
    # Fallback tracer
    class MockTracer:
        def start_as_current_span(self, name: str):
            return MockSpan()

    class MockSpan:
        def set_attribute(self, key: str, value: Any):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    tracer = MockTracer()


class ServiceStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class LoadBalancingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    RANDOM = "random"
    LEAST_CONNECTIONS = "least_connections"


@dataclass
class ServiceEndpoint:
    """Service endpoint definition"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    service_name: str = ""
    host: str = "localhost"
    port: int = 8000
    protocol: str = "http"
    version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    health_check_path: str = "/health"
    capabilities: Set[str] = field(default_factory=set)
    weight: int = 1
    max_connections: int = 100
    current_connections: int = 0
    status: ServiceStatus = ServiceStatus.UNKNOWN
    last_health_check: Optional[datetime] = None
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def url(self) -> str:
        """Get service URL"""
        return f"{self.protocol}://{self.host}:{self.port}"

    @property
    def health_check_url(self) -> str:
        """Get health check URL"""
        return f"{self.url}{self.health_check_path}"


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""

    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    half_open_max_calls: int = 3
    timeout: int = 30  # seconds


class CircuitBreaker:
    """Circuit breaker implementation"""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half_open
        self.half_open_calls = 0

    def can_execute(self) -> bool:
        """Check if operation can be executed"""
        if self.state == "closed":
            return True

        if self.state == "open":
            if (
                self.last_failure_time
                and (
                    datetime.now(timezone.utc) - self.last_failure_time
                ).total_seconds()
                > self.config.recovery_timeout
            ):
                self.state = "half_open"
                self.half_open_calls = 0
                return True
            return False

        if self.state == "half_open":
            return self.half_open_calls < self.config.half_open_max_calls

        return False

    def record_success(self) -> None:
        """Record successful operation"""
        if self.state == "half_open":
            self.half_open_calls += 1
            if self.half_open_calls >= self.config.half_open_max_calls:
                self.state = "closed"
                self.failure_count = 0
        else:
            self.failure_count = max(0, self.failure_count - 1)

    def record_failure(self) -> None:
        """Record failed operation"""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)
        if self.failure_count >= self.config.failure_threshold:
            self.state = "open"
        elif self.state == "half_open":
            self.state = "open"


class ServiceMesh:
    """
    Distributed service mesh for service discovery, load balancing,
    and circuit breaking.
    """

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        default_lb_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN,
        health_check_interval: int = 30,
        enable_circuit_breaker: bool = True,
    ):
        self.redis_client = redis_client
        if not self.redis_client and redis:
            try:
                self.redis_client = redis.Redis(decode_responses=True)
            except Exception:
                self.redis_client = None

        self.default_lb_strategy = default_lb_strategy
        self.health_check_interval = health_check_interval
        self.enable_circuit_breaker = enable_circuit_breaker

        # Service registry
        self.services: Dict[str, List[ServiceEndpoint]] = defaultdict(list)
        self.service_endpoints: Dict[str, ServiceEndpoint] = {}

        # Load balancing state
        self.lb_state: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # Circuit breakers
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

        # Rate limiters
        self.rate_limiters: Dict[str, Any] = {}

        # Monitoring
        self.metrics: Dict[str, Any] = defaultdict(int)

        # State
        self._running = False
        self._health_check_task: Optional[asyncio.Task[None]] = None
        self._cleanup_task: Optional[asyncio.Task[None]] = None

        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Setup logging"""
        logger = logging.getLogger("ServiceMesh")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    async def start(self) -> None:
        """Start service mesh"""
        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        # Load existing services from Redis
        await self._load_services_from_redis()

        self.logger.info("Service mesh started")

    async def stop(self) -> None:
        """Stop service mesh"""
        self._running = False

        if self._health_check_task:
            self._health_check_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()

        self.logger.info("Service mesh stopped")

    async def register_service(self, endpoint: ServiceEndpoint) -> None:
        """Register a service endpoint"""
        with tracer.start_as_current_span("register_service") as span:
            span.set_attribute("service.name", endpoint.service_name)
            span.set_attribute("service.id", endpoint.id)

            # Add to in-memory registry
            self.services[endpoint.service_name].append(endpoint)
            self.service_endpoints[endpoint.id] = endpoint

            # Initialize circuit breaker if enabled
            if self.enable_circuit_breaker:
                cb_config = CircuitBreakerConfig()
                self.circuit_breakers[endpoint.id] = CircuitBreaker(cb_config)

            # Persist to Redis if available
            if self.redis_client:
                await self._persist_service(endpoint)

            self.logger.info(
                f"Registered service: {endpoint.service_name} at {endpoint.url}"
            )

    async def deregister_service(self, service_id: str) -> None:
        """Deregister a service endpoint"""
        with tracer.start_as_current_span("deregister_service") as span:
            span.set_attribute("service.id", service_id)

            if service_id in self.service_endpoints:
                endpoint = self.service_endpoints[service_id]

                # Remove from in-memory registry
                self.services[endpoint.service_name] = [
                    ep
                    for ep in self.services[endpoint.service_name]
                    if ep.id != service_id
                ]
                del self.service_endpoints[service_id]

                # Remove circuit breaker
                if service_id in self.circuit_breakers:
                    del self.circuit_breakers[service_id]

                # Remove from Redis if available
                if self.redis_client:
                    try:
                        self.redis_client.delete(f"service:{service_id}")
                        self.redis_client.srem(
                            f"services:{endpoint.service_name}", service_id
                        )
                    except Exception as e:
                        self.logger.error(f"Failed to remove service from Redis: {e}")

                self.logger.info(f"Deregistered service: {service_id}")

    async def discover_services(self, service_name: str) -> List[ServiceEndpoint]:
        """Discover service endpoints by name"""
        with tracer.start_as_current_span("discover_services") as span:
            span.set_attribute("service.name", service_name)

            # Filter healthy endpoints
            healthy_endpoints = [
                ep
                for ep in self.services.get(service_name, [])
                if ep.status == ServiceStatus.HEALTHY
            ]

            span.set_attribute("service.endpoints_count", len(healthy_endpoints))
            return healthy_endpoints

    async def get_service_endpoint(
        self, service_name: str, strategy: Optional[LoadBalancingStrategy] = None
    ) -> Optional[ServiceEndpoint]:
        """Get service endpoint using load balancing"""
        with tracer.start_as_current_span("get_service_endpoint") as span:
            span.set_attribute("service.name", service_name)

            endpoints = await self.discover_services(service_name)
            if not endpoints:
                return None

            strategy = strategy or self.default_lb_strategy

            # Apply load balancing strategy
            if strategy == LoadBalancingStrategy.ROUND_ROBIN:
                endpoint = self._round_robin_select(service_name, endpoints)
            elif strategy == LoadBalancingStrategy.WEIGHTED:
                endpoint = self._weighted_select(endpoints)
            elif strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                endpoint = self._least_connections_select(endpoints)
            else:  # RANDOM
                import random

                endpoint = random.choice(endpoints)

            span.set_attribute("service.selected_endpoint", endpoint.id)
            return endpoint

    def _round_robin_select(
        self, service_name: str, endpoints: List[ServiceEndpoint]
    ) -> ServiceEndpoint:
        """Round-robin load balancing"""
        if service_name not in self.lb_state:
            self.lb_state[service_name]["current_index"] = 0

        current_index = self.lb_state[service_name]["current_index"]
        selected = endpoints[current_index % len(endpoints)]

        self.lb_state[service_name]["current_index"] = (current_index + 1) % len(
            endpoints
        )
        return selected

    def _weighted_select(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Weighted load balancing"""
        total_weight = sum(ep.weight for ep in endpoints)
        if total_weight == 0:
            return endpoints[0]

        import random

        target = random.randint(1, total_weight)

        current_weight = 0
        for endpoint in endpoints:
            current_weight += endpoint.weight
            if current_weight >= target:
                return endpoint

        return endpoints[-1]

    def _least_connections_select(
        self, endpoints: List[ServiceEndpoint]
    ) -> ServiceEndpoint:
        """Least connections load balancing"""
        return min(endpoints, key=lambda ep: ep.current_connections)

    async def execute_with_circuit_breaker(
        self, service_id: str, operation: Callable, *args, **kwargs
    ) -> Any:
        """Execute operation with circuit breaker protection"""
        if not self.enable_circuit_breaker or service_id not in self.circuit_breakers:
            return await operation(*args, **kwargs)

        cb = self.circuit_breakers[service_id]

        if not cb.can_execute():
            raise Exception(f"Circuit breaker open for service {service_id}")

        try:
            result = await operation(*args, **kwargs)
            cb.record_success()
            return result
        except Exception as e:
            cb.record_failure()
            raise e

    async def _health_check_loop(self) -> None:
        """Background health checking"""
        while self._running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                await asyncio.sleep(5)

    async def _perform_health_checks(self) -> None:
        """Perform health checks on all services"""
        for endpoint in self.service_endpoints.values():
            try:
                # Simple health check - in production use proper HTTP client
                try:
                    import httpx

                    async with httpx.AsyncClient(timeout=5.0) as client:
                        response = await client.get(endpoint.health_check_url)
                        if response.status_code == 200:
                            endpoint.status = ServiceStatus.HEALTHY
                        else:
                            endpoint.status = ServiceStatus.UNHEALTHY
                except ImportError:
                    # Fallback - mark as healthy if no health check client available
                    endpoint.status = ServiceStatus.HEALTHY
            except Exception:
                endpoint.status = ServiceStatus.UNHEALTHY

            endpoint.last_health_check = datetime.now(timezone.utc)
            # Update in Redis
            if self.redis_client:
                await self._persist_service(endpoint)

    async def _cleanup_loop(self) -> None:
        """Background cleanup of stale services"""
        while self._running:
            try:
                await self._cleanup_stale_services()
                await asyncio.sleep(300)  # Every 5 minutes
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(60)

    async def _cleanup_stale_services(self) -> None:
        """Remove stale service registrations"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=10)
        stale_services: List[str] = []

        for service_id, endpoint in self.service_endpoints.items():
            if endpoint.last_health_check and endpoint.last_health_check < cutoff_time:
                stale_services.append(service_id)

        for service_id in stale_services:
            await self.deregister_service(service_id)
            self.logger.info(f"Cleaned up stale service: {service_id}")

    async def _persist_service(self, endpoint: ServiceEndpoint) -> None:
        """Persist service to Redis"""
        if not self.redis_client:
            return

        try:
            service_data = {
                "id": endpoint.id,
                "service_name": endpoint.service_name,
                "host": endpoint.host,
                "port": endpoint.port,
                "protocol": endpoint.protocol,
                "version": endpoint.version,
                "metadata": str(endpoint.metadata),
                "health_check_path": endpoint.health_check_path,
                "capabilities": list(endpoint.capabilities),
                "weight": endpoint.weight,
                "max_connections": endpoint.max_connections,
                "current_connections": endpoint.current_connections,
                "status": endpoint.status.value,
                "last_health_check": (
                    endpoint.last_health_check.isoformat()
                    if endpoint.last_health_check
                    else None
                ),
                "registered_at": endpoint.registered_at.isoformat(),
            }

            # Store service data with TTL
            self.redis_client.hmset(f"service:{endpoint.id}", service_data)
            self.redis_client.expire(f"service:{endpoint.id}", 3600)  # 1 hour TTL

            # Add to service name index
            self.redis_client.sadd(f"services:{endpoint.service_name}", endpoint.id)

        except Exception as e:
            self.logger.error(f"Failed to persist service to Redis: {e}")

    async def _load_services_from_redis(self) -> None:
        """Load services from Redis on startup"""
        if not self.redis_client:
            return

        try:
            # Get all service keys
            service_keys = self.redis_client.keys("service:*")

            for key in service_keys:
                try:
                    service_data = self.redis_client.hgetall(key)
                    if not service_data:
                        continue

                    # Reconstruct service endpoint
                    endpoint = ServiceEndpoint(
                        id=service_data["id"],
                        service_name=service_data["service_name"],
                        host=service_data["host"],
                        port=int(service_data["port"]),
                        protocol=service_data["protocol"],
                        version=service_data["version"],
                        health_check_path=service_data["health_check_path"],
                        weight=int(service_data["weight"]),
                        max_connections=int(service_data["max_connections"]),
                        current_connections=int(service_data["current_connections"]),
                        status=ServiceStatus(service_data["status"]),
                    )

                    # Add to registries
                    self.services[endpoint.service_name].append(endpoint)
                    self.service_endpoints[endpoint.id] = endpoint

                except Exception as e:
                    self.logger.error(f"Failed to load service from Redis: {e}")

        except Exception as e:
            self.logger.error(f"Failed to load services from Redis: {e}")

    async def get_service_stats(self) -> Dict[str, Any]:
        """Get service mesh statistics"""
        stats = {
            "total_services": len(self.service_endpoints),
            "services_by_name": {
                name: len(endpoints) for name, endpoints in self.services.items()
            },
            "services_by_status": defaultdict(int),
            "circuit_breakers": len(self.circuit_breakers),
            "metrics": dict(self.metrics),
        }

        # Count services by status
        for endpoint in self.service_endpoints.values():
            stats["services_by_status"][endpoint.status.value] += 1

        return stats

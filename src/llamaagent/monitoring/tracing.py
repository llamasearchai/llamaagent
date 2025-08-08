#!/usr/bin/env python3
"""
Distributed tracing module for LlamaAgent.

This module provides distributed tracing capabilities using OpenTelemetry
with fallback support when OpenTelemetry is not available.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import functools
import logging
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from types import TracebackType
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, cast

# Type variables for decorators
F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)

# Check OpenTelemetry availability and create fallback stubs
_opentelemetry_available = False
try:
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor  # type: ignore
    from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor  # type: ignore
    from opentelemetry.instrumentation.redis import RedisInstrumentor  # type: ignore
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.trace import SpanKind, Status, StatusCode

    _opentelemetry_available = True
except ImportError:
    logger.info("OpenTelemetry not available, using fallback tracing")

    # Create fallback stubs for missing imports
    class _FallbackSpanKind:
        INTERNAL = "internal"
        CLIENT = "client"
        SERVER = "server"
        PRODUCER = "producer"
        CONSUMER = "consumer"

    class _FallbackStatusCode:
        OK = "ok"
        ERROR = "error"

    class _FallbackStatus:
        def __init__(self, status_code: str, description: Optional[str] = None):
            self.status_code = status_code
            self.description = description

    class _FallbackTrace:
        def get_current_span(self):
            return None

        def set_tracer_provider(self, provider: Any) -> None:
            pass

        def get_tracer(self, name: str) -> Any:
            return None

    class _FallbackResource:
        @staticmethod
        def create(attributes: Dict[str, Any]) -> Any:
            return None

    class _FallbackTracerProvider:
        def __init__(self, resource: Any = None):
            pass

        def add_span_processor(self, processor: Any) -> None:
            pass

    class _FallbackExporter:
        def __init__(self, *args: Any, **kwargs: Any):
            pass

    class _FallbackInstrumentor:
        def instrument(self) -> None:
            pass

    class _FallbackBatchSpanProcessor:
        def __init__(self, exporter: Any):
            pass

    # Assign fallback implementations
    SpanKind = _FallbackSpanKind()  # type: ignore
    StatusCode = _FallbackStatusCode()  # type: ignore
    Status = _FallbackStatus  # type: ignore
    trace = _FallbackTrace()  # type: ignore
    Resource = _FallbackResource()  # type: ignore
    TracerProvider = _FallbackTracerProvider  # type: ignore
    JaegerExporter = _FallbackExporter  # type: ignore
    OTLPSpanExporter = _FallbackExporter  # type: ignore
    ConsoleSpanExporter = _FallbackExporter  # type: ignore
    BatchSpanProcessor = _FallbackBatchSpanProcessor  # type: ignore
    AioHttpClientInstrumentor = _FallbackInstrumentor  # type: ignore
    Psycopg2Instrumentor = _FallbackInstrumentor  # type: ignore
    RedisInstrumentor = _FallbackInstrumentor  # type: ignore


# Context variable for current span
current_span_context: ContextVar[Optional["SpanContext"]] = ContextVar(
    "current_span", default=None
)


class SpanContext:
    """Context for a tracing span"""

    def __init__(
        self,
        span_id: str,
        trace_id: str,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.span_id = span_id
        self.trace_id = trace_id
        self.name = name
        self.attributes = attributes or {}
        self.start_time = datetime.now(timezone.utc)
        self.events: List[Dict[str, Any]] = []
        self.status: Optional[str] = None
        self.error: Optional[str] = None


class LocalSpanManager:
    """Manager for local span context"""

    def __init__(self, manager: "TracingManager", context: SpanContext) -> None:
        self.manager = manager
        self.context = context

    def __enter__(self) -> "LocalSpanManager":
        current_span_context.set(self.context)
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if exc_type:
            self.context.status = "error"
            self.context.error = str(exc_val) if exc_val else "Unknown error"
        else:
            self.context.status = "ok"
        current_span_context.set(None)

    async def __aenter__(self) -> "LocalSpanManager":
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)

    def set_status(self, status: str, description: Optional[str] = None) -> None:
        """Set the status of the span"""
        self.context.status = status
        if description:
            self.context.error = description


class TracingManager:
    """Manages distributed tracing for the application"""

    def __init__(
        self,
        service_name: str = "llamaagent",
        jaeger_endpoint: Optional[str] = None,
        otlp_endpoint: Optional[str] = None,
        enable_console: bool = False,
        sample_rate: float = 1.0,
    ) -> None:
        self.service_name = service_name
        self.jaeger_endpoint = jaeger_endpoint
        self.otlp_endpoint = otlp_endpoint
        self.enable_console = enable_console
        self.sample_rate = sample_rate
        self.tracer = None
        self.initialized = False

        # Fallback storage when OpenTelemetry is not available
        self.local_traces: List[SpanContext] = []

        self._initialize()

    def _initialize(self) -> None:
        """Initialize the tracing system"""
        if not _opentelemetry_available:
            logger.warning("OpenTelemetry not available, using fallback tracing")
            return

        try:
            # Create resource
            resource = Resource.create({"service.name": self.service_name})

            # Create tracer provider
            provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(provider)  # type: ignore

            # Configure exporters
            if self.jaeger_endpoint:
                jaeger_exporter = JaegerExporter(
                    agent_host_name=(
                        self.jaeger_endpoint.split(":")[0]
                        if ":" in self.jaeger_endpoint
                        else self.jaeger_endpoint
                    ),
                    agent_port=(
                        int(self.jaeger_endpoint.split(":")[1])
                        if ":" in self.jaeger_endpoint
                        else 6831
                    ),
                )
                provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))  # type: ignore

            if self.otlp_endpoint:
                otlp_exporter = OTLPSpanExporter(
                    endpoint=self.otlp_endpoint, insecure=True
                )
                provider.add_span_processor(BatchSpanProcessor(otlp_exporter))  # type: ignore

            if self.enable_console:
                console_exporter = ConsoleSpanExporter()
                provider.add_span_processor(BatchSpanProcessor(console_exporter))  # type: ignore

            # Get tracer
            self.tracer = trace.get_tracer(__name__)
            self.initialized = True

            # Setup auto-instrumentation
            self._setup_auto_instrumentation()

        except Exception as e:
            logger.error(f"Failed to initialize tracing: {e}")

    def _setup_auto_instrumentation(self) -> None:
        """Setup automatic instrumentation for common libraries"""
        if not _opentelemetry_available:
            return

        try:
            # Auto-instrument HTTP client
            AioHttpClientInstrumentor().instrument()  # type: ignore

            # Auto-instrument PostgreSQL
            Psycopg2Instrumentor().instrument()  # type: ignore

            # Auto-instrument Redis
            RedisInstrumentor().instrument()  # type: ignore

        except Exception as e:
            logger.warning(f"Failed to set up auto-instrumentation: {e}")

    def create_span(
        self,
        name: str,
        kind: Optional[Any] = None,  # Use Any to avoid SpanKind type issues
        attributes: Optional[Dict[str, Any]] = None,
        parent_context: Optional[SpanContext] = None,
    ) -> LocalSpanManager:
        """Create a new span"""
        if self.initialized and self.tracer and _opentelemetry_available:
            # Use OpenTelemetry
            span_kind = kind or SpanKind.INTERNAL  # type: ignore

            # Get parent context - simplified implementation
            span = self.tracer.start_as_current_span(
                name,
                kind=span_kind,  # type: ignore
                attributes=attributes or {},
            )

            # Create our context wrapper with safe access
            try:
                span_context = SpanContext(
                    span_id=str(getattr(span.get_span_context(), 'span_id', uuid.uuid4())),  # type: ignore
                    trace_id=str(getattr(span.get_span_context(), 'trace_id', uuid.uuid4())),  # type: ignore
                    name=name,
                    attributes=attributes or {},
                )
            except (AttributeError, TypeError):
                # Fallback if span context access fails
                span_context = SpanContext(
                    span_id=str(uuid.uuid4()),
                    trace_id=str(uuid.uuid4()),
                    name=name,
                    attributes=attributes or {},
                )

            return LocalSpanManager(self, span_context)
        else:
            # Fallback implementation
            span_id = str(uuid.uuid4())
            trace_id = parent_context.trace_id if parent_context else str(uuid.uuid4())

            span_context = SpanContext(
                span_id=span_id,
                trace_id=trace_id,
                name=name,
                attributes=attributes or {},
            )

            self.local_traces.append(span_context)
            current_span_context.set(span_context)

            return LocalSpanManager(self, span_context)

    def set_span_status(self, status: str, description: Optional[str] = None) -> None:
        """Set the status of the current span"""
        if self.initialized and _opentelemetry_available:
            span = trace.get_current_span()
            if span and hasattr(span, 'is_recording') and span.is_recording():
                status_code = StatusCode.OK if status == "ok" else StatusCode.ERROR  # type: ignore
                span.set_status(Status(status_code, description))  # type: ignore
        else:
            # Fallback
            ctx = current_span_context.get()
            if ctx:
                ctx.status = status
                if description:
                    ctx.error = description

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to the current span"""
        if self.initialized and _opentelemetry_available:
            span = trace.get_current_span()
            if span and hasattr(span, 'is_recording') and span.is_recording():
                span.add_event(name, attributes=attributes or {})
        else:
            # Fallback
            ctx = current_span_context.get()
            if ctx:
                ctx.events.append(
                    {
                        "name": name,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "attributes": attributes or {},
                    }
                )

    def get_current_trace_id(self) -> Optional[str]:
        """Get the current trace ID"""
        if self.initialized and _opentelemetry_available:
            span = trace.get_current_span()
            if span and hasattr(span, 'is_recording') and span.is_recording():
                try:
                    return format(span.get_span_context().trace_id, "032x")
                except (AttributeError, TypeError):
                    pass

        # Fallback
        ctx = current_span_context.get()
        return ctx.trace_id if ctx else None

    def export_traces(self) -> List[Dict[str, Any]]:
        """Export local traces (fallback mode)"""
        traces: List[Dict[str, Any]] = []
        for span in self.local_traces:
            traces.append(
                {
                    "span_id": span.span_id,
                    "trace_id": span.trace_id,
                    "name": span.name,
                    "start_time": span.start_time.isoformat(),
                    "attributes": span.attributes,
                    "events": span.events,
                    "status": span.status,
                    "error": span.error,
                }
            )
        return traces


# Global tracing manager
_tracing_manager: Optional[TracingManager] = None


def get_tracing_manager() -> TracingManager:
    """Get the global tracing manager"""
    global _tracing_manager
    if _tracing_manager is None:
        _tracing_manager = TracingManager()
    return _tracing_manager


def trace_async(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> Callable[[F], F]:
    """Decorator for tracing async functions"""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            span_name = name or func.__name__
            manager = get_tracing_manager()

            # Add function info to attributes
            span_attributes = attributes or {}
            span_attributes.update(
                {
                    "function.name": func.__name__,
                    "function.module": func.__module__,
                }
            )

            with manager.create_span(
                span_name, attributes=span_attributes
            ) as span_manager:
                try:
                    result = await func(*args, **kwargs)
                    span_manager.set_status("ok")
                    return result
                except Exception as e:
                    span_manager.set_status("error", str(e))
                    raise

        return cast(F, wrapper)

    return decorator


def trace_sync(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> Callable[[F], F]:
    """Decorator for tracing sync functions"""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            span_name = name or func.__name__
            manager = get_tracing_manager()

            # Add function info to attributes
            span_attributes = attributes or {}
            span_attributes.update(
                {
                    "function.name": func.__name__,
                    "function.module": func.__module__,
                }
            )

            with manager.create_span(
                span_name, attributes=span_attributes
            ) as span_manager:
                try:
                    result = func(*args, **kwargs)
                    span_manager.set_status("ok")
                    return result
                except Exception as e:
                    span_manager.set_status("error", str(e))
                    raise

        return cast(F, wrapper)

    return decorator


def set_trace_attribute(key: str, value: Any) -> None:
    """Set an attribute on the current trace"""
    ctx = current_span_context.get()
    if ctx:
        ctx.attributes[key] = value


def add_trace_event(name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
    """Add an event to the current trace"""
    get_tracing_manager().add_event(name, attributes)

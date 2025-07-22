#!/usr/bin/env python3
"""
Structured Logging Module for LlamaAgent

Provides comprehensive logging capabilities with structured output,
context management, and performance monitoring.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import functools
import json
import logging
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Dict, Iterator, Optional, Type

try:
    import psutil

    _psutil_available = True
    del psutil  # Remove from module namespace
except ImportError:
    _psutil_available = False


@dataclass
class LogContext:
    """Context information for structured logging."""

    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    agent_name: Optional[str] = None
    task_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None}


# Context variables for logging context
log_context: ContextVar[Dict[str, Any]] = ContextVar("log_context", default={})


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""

    def __init__(self, include_traceback: bool = True):
        super().__init__()
        self.include_traceback = include_traceback

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add context if available
        ctx = log_context.get()
        if ctx:
            log_data["context"] = json.dumps(ctx, default=str)

        # Add extra fields - safely handle dictionary type
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if key not in log_data and not key.startswith("_"):
                    # Convert complex objects to string representation
                    if isinstance(value, dict):
                        log_data[key] = json.dumps(value, default=str)
                    else:
                        log_data[key] = str(value)

        # Add exception info if present
        if record.exc_info and self.include_traceback:
            exc_type, exc_value, _ = record.exc_info
            if exc_type is not None:
                log_data["exception"] = json.dumps(
                    {
                        "type": exc_type.__name__,
                        "message": str(exc_value),
                        "traceback": self.formatException(record.exc_info),
                    },
                    default=str,
                )

        return json.dumps(log_data, default=str)


class StructuredLogger:
    """Structured logger with context support."""

    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

    def _log(self, level: int, message: str, **kwargs: Any) -> None:
        """Internal log method."""
        # Merge context
        ctx = log_context.get().copy()
        ctx.update(kwargs)

        # Create log record with extra fields
        self.logger.log(level, message, extra=ctx)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self._log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, message, **kwargs)

    def with_context(self, **kwargs: Any) -> "ContextualLogger":
        """Create a contextual logger."""
        return ContextualLogger(self.logger, LogContext(**kwargs))


class ContextualLogger:
    """Logger with context."""

    def __init__(self, logger: logging.Logger, context: LogContext) -> None:
        self.logger = logger
        self.context = context
        self._previous_context: Dict[str, Any] = {}

    def __enter__(self) -> "ContextualLogger":
        # Store previous context
        self._previous_context = log_context.get()

        # Merge contexts
        new_context = self._previous_context.copy()
        new_context.update(self.context.to_dict())
        log_context.set(new_context)

        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        # Restore previous context
        log_context.set(self._previous_context)

    def debug(self, message: str, **kwargs: Any) -> None:
        self.logger.debug(message, extra=kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        self.logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self.logger.warning(message, extra=kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self.logger.error(message, extra=kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        self.logger.critical(message, extra=kwargs)


def log_performance(
    operation_name: str,
    include_memory: bool = False,
    level: int = logging.INFO,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for logging performance of operations."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            logger = logging.getLogger(func.__module__ if func.__module__ else __name__)

            # Collect initial metrics
            initial_memory: Optional[float] = None
            if include_memory and _psutil_available:
                try:
                    import psutil

                    process = psutil.Process()
                    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
                except Exception as e:
                    # Memory monitoring is not critical, but log the issue
                    logger.debug(f"Unable to collect initial memory metrics: {e}")

            try:
                result: Any = await func(*args, **kwargs)
                duration = time.time() - start_time

                # Collect final metrics
                extra: Dict[str, Any] = {
                    "operation": operation_name,
                    "duration_seconds": duration,
                    "status": "success",
                }

                if initial_memory is not None and _psutil_available:
                    try:
                        import psutil

                        process = psutil.Process()
                        final_memory = process.memory_info().rss / 1024 / 1024
                        extra["memory_usage_mb"] = final_memory - initial_memory
                    except Exception as e:
                        # Memory monitoring is not critical, but log the issue
                        logger.debug(f"Unable to collect final memory metrics: {e}")

                logger.log(
                    level,
                    f"Operation '{operation_name}' completed in {duration:.3f}s",
                    extra=extra,
                )
                return result

            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Operation '{operation_name}' failed after {duration:.3f}s",
                    extra={
                        "operation": operation_name,
                        "duration_seconds": duration,
                        "status": "error",
                        "error": str(e),
                    },
                    exc_info=e,
                )
                raise

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            logger = logging.getLogger(func.__module__ if func.__module__ else __name__)

            # Similar implementation for sync functions
            try:
                result: Any = func(*args, **kwargs)
                duration = time.time() - start_time

                logger.log(
                    level,
                    f"Operation '{operation_name}' completed in {duration:.3f}s",
                    extra={
                        "operation": operation_name,
                        "duration_seconds": duration,
                        "status": "success",
                    },
                )
                return result

            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Operation '{operation_name}' failed after {duration:.3f}s",
                    extra={
                        "operation": operation_name,
                        "duration_seconds": duration,
                        "status": "error",
                        "error": str(e),
                    },
                    exc_info=e,
                )
                raise

        # Return appropriate wrapper
        if hasattr(func, "__code__") and func.__code__.co_flags & 0x80:  # CO_COROUTINE
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


@contextmanager
def log_context_manager(**kwargs: Any) -> Iterator[None]:
    """Context manager for logging context."""
    context = LogContext(
        trace_id=kwargs.get("trace_id"),
        span_id=kwargs.get("span_id"),
        user_id=kwargs.get("user_id"),
        session_id=kwargs.get("session_id"),
        request_id=kwargs.get("request_id"),
        agent_name=kwargs.get("agent_name"),
        task_id=kwargs.get("task_id"),
        extra=kwargs.get("extra", {}),
    )
    previous_context = log_context.get()

    # Merge contexts
    new_context = previous_context.copy()
    new_context.update(context.to_dict())
    log_context.set(new_context)

    try:
        yield
    finally:
        log_context.set(previous_context)


def configure_logging(
    level: str = "INFO",
    enable_console: bool = True,
    enable_file: bool = True,
    file_path: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    enable_performance_logging: bool = True,
) -> None:
    """Configure global logging settings."""
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    root_logger.handlers.clear()  # Clear existing handlers

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        if enable_performance_logging:
            console_handler.setFormatter(StructuredFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
        root_logger.addHandler(console_handler)

    # File handler
    if enable_file and file_path:
        from logging.handlers import RotatingFileHandler

        # Create directory if needed
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        file_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(file_handler)

    # Configure specific loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    logger = get_logger(__name__)
    logger.info(
        "Logging configured",
        log_level=level,
        console_enabled=enable_console,
        file_enabled=enable_file,
        file_path=file_path,
    )


# Convenience function for getting logger
def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance."""
    return StructuredLogger(name)


# Export main components
__all__ = [
    "StructuredLogger",
    "ContextualLogger",
    "LogContext",
    "log_performance",
    "log_context_manager",
    "configure_logging",
    "get_logger",
]

"""
Comprehensive Error Handling System

This module provides robust error handling with retry mechanisms,
circuit breakers, fallback strategies, and error categorization.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

# Optional imports
try:
    from tenacity import (
        RetryError,
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        stop_after_delay,
        wait_exponential,
        wait_random_exponential,
    )

    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False

    # Create simple tenacity replacements
    class RetryError(Exception):
        pass

    def retry(**kwargs):
        def decorator(func):
            return func

        return decorator

    def stop_after_attempt(n):
        return n

    def stop_after_delay(d):
        return d

    def wait_exponential(**kwargs):
        return None

    def wait_random_exponential(**kwargs):
        return None

    def retry_if_exception_type(exc_type):
        return exc_type


logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""

    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    COMPENSATE = "compensate"
    ESCALATE = "escalate"
    IGNORE = "ignore"


@dataclass
class ErrorContext:
    """Context information for errors."""

    error_type: Type[Exception]
    error_message: str
    error_trace: str
    timestamp: float
    severity: ErrorSeverity
    component: str
    operation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryResult:
    """Result of an error recovery attempt."""

    success: bool
    result: Any = None
    error: Optional[Exception] = None
    strategy_used: Optional[RecoveryStrategy] = None
    retry_count: int = 0
    recovery_time: float = 0.0
    message: str = ""


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Circuit breaker implementation for preventing cascading failures.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None

    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker to a function."""
        if asyncio.iscoroutinefunction(func):

            async def async_wrapper(*args, **kwargs):
                return await self.async_call(func, *args, **kwargs)

            return async_wrapper
        else:

            def sync_wrapper(*args, **kwargs):
                return self.call(func, *args, **kwargs)

            return sync_wrapper

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise Exception(f"Circuit breaker is OPEN for {func.__name__}")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e

    async def async_call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise Exception(f"Circuit breaker is OPEN for {func.__name__}")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit."""
        return (
            self.last_failure_time
            and time.time() - self.last_failure_time >= self.recovery_timeout
        )

    def _on_success(self) -> None:
        """Handle successful call."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 3:  # Require 3 successes to close
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0

    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            self.success_count = 0
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN


class ErrorHandler:
    """
    Comprehensive error handling system with multiple recovery strategies.

    Features:
    - Intelligent retry mechanisms
    - Circuit breaker pattern
    - Fallback strategies
    - Error categorization and routing
    - Compensation mechanisms
    - Error escalation
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        enable_circuit_breaker: bool = True,
        enable_metrics: bool = True,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.enable_circuit_breaker = enable_circuit_breaker
        self.enable_metrics = enable_metrics

        # Circuit breakers by component
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

        # Error tracking
        self.error_history: List[ErrorContext] = []
        self.error_handlers: Dict[Type[Exception], Callable] = {}
        self.fallback_handlers: Dict[str, Callable] = {}

        # Recovery strategies
        self.recovery_strategies: Dict[Type[Exception], RecoveryStrategy] = {
            ConnectionError: RecoveryStrategy.RETRY,
            TimeoutError: RecoveryStrategy.RETRY,
            ValueError: RecoveryStrategy.FALLBACK,
            KeyError: RecoveryStrategy.FALLBACK,
            RuntimeError: RecoveryStrategy.CIRCUIT_BREAK,
        }

    def register_error_handler(
        self, error_type: Type[Exception], handler: Callable
    ) -> None:
        """Register a custom error handler for a specific exception type."""
        self.error_handlers[error_type] = handler

    def register_fallback_handler(self, component: str, handler: Callable) -> None:
        """Register a fallback handler for a component."""
        self.fallback_handlers[component] = handler

    def handle_error(
        self,
        error: Exception,
        component: str,
        operation: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """
        Handle an error with the appropriate strategy.

        Args:
            error: The exception that occurred
            component: The component where the error occurred
            operation: The operation that failed
            context: Additional context information

        Returns:
            Result if error was handled, None otherwise
        """
        error_context = ErrorContext(
            error_type=type(error),
            error_message=str(error),
            error_trace=str(error.__traceback__ if error.__traceback__ else ""),
            timestamp=time.time(),
            severity=self._determine_severity(error),
            component=component,
            operation=operation,
            metadata=context or {},
        )

        # Store error in history
        self.error_history.append(error_context)

        # Log error
        self.logger.error(
            f"Error in {component}.{operation}: {str(error)}",
            exc_info=True,
            extra={"component": component, "operation": operation},
        )

        # Check for custom error handler
        if type(error) in self.error_handlers:
            try:
                return self.error_handlers[type(error)](error, error_context)
            except Exception as handler_error:
                self.logger.error(f"Error handler failed: {handler_error}")

        # Apply recovery strategy
        strategy = self.recovery_strategies.get(type(error), RecoveryStrategy.ESCALATE)

        if strategy == RecoveryStrategy.FALLBACK:
            return self._handle_fallback(error_context)
        elif strategy == RecoveryStrategy.CIRCUIT_BREAK:
            self._handle_circuit_break(error_context)
        elif strategy == RecoveryStrategy.COMPENSATE:
            return self._handle_compensation(error_context)

        return None

    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """Determine the severity of an error."""
        if isinstance(error, (SystemExit, KeyboardInterrupt)):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorSeverity.HIGH
        elif isinstance(error, (ValueError, KeyError)):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW

    def _handle_fallback(self, error_context: ErrorContext) -> Optional[Any]:
        """Handle error with fallback strategy."""
        if error_context.component in self.fallback_handlers:
            try:
                return self.fallback_handlers[error_context.component](error_context)
            except Exception as fallback_error:
                self.logger.error(f"Fallback handler failed: {fallback_error}")
        return None

    def _handle_circuit_break(self, error_context: ErrorContext) -> None:
        """Handle error with circuit breaker."""
        if not self.enable_circuit_breaker:
            return

        component = error_context.component
        if component not in self.circuit_breakers:
            self.circuit_breakers[component] = CircuitBreaker()

        # The circuit breaker will handle the failure internally
        self.circuit_breakers[component]._on_failure()

    def _handle_compensation(self, error_context: ErrorContext) -> Optional[Any]:
        """Handle error with compensation logic."""
        # This is a placeholder for compensation logic
        # In a real implementation, this would undo any partial changes
        self.logger.info(f"Compensating for error in {error_context.component}")
        return None

    def get_circuit_breaker(self, component: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker for a component."""
        return self.circuit_breakers.get(component)

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        if not self.error_history:
            return {
                "total_errors": 0,
                "by_severity": {},
                "by_component": {},
                "by_type": {},
                "recent_errors": [],
            }

        total_errors = len(self.error_history)

        # Group by severity
        by_severity: Dict[str, int] = {}
        for error in self.error_history:
            severity = error.severity.value
            by_severity[severity] = by_severity.get(severity, 0) + 1

        # Group by component
        by_component: Dict[str, int] = {}
        for error in self.error_history:
            component = error.component
            by_component[component] = by_component.get(component, 0) + 1

        # Group by error type
        by_type: Dict[str, int] = {}
        for error in self.error_history:
            error_type = error.error_type.__name__
            by_type[error_type] = by_type.get(error_type, 0) + 1

        # Recent errors
        recent_errors = [
            {
                "timestamp": error.timestamp,
                "component": error.component,
                "operation": error.operation,
                "error_type": error.error_type.__name__,
                "message": error.error_message,
                "severity": error.severity.value,
            }
            for error in self.error_history[-10:]
        ]

        return {
            "total_errors": total_errors,
            "by_severity": by_severity,
            "by_component": by_component,
            "by_type": by_type,
            "recent_errors": recent_errors,
            "circuit_breakers": {
                name: breaker.state.value
                for name, breaker in self.circuit_breakers.items()
            },
        }

    def reset_circuit_breaker(self, component: str) -> bool:
        """Reset circuit breaker for a component."""
        if component in self.circuit_breakers:
            self.circuit_breakers[component].state = CircuitBreakerState.CLOSED
            self.circuit_breakers[component].failure_count = 0
            self.circuit_breakers[component].success_count = 0
            self.circuit_breakers[component].last_failure_time = None
            return True
        return False

    def clear_error_history(self) -> None:
        """Clear error history."""
        self.error_history.clear()


# Global error handler instance
default_error_handler = ErrorHandler()


def handle_errors(
    component: str,
    operation: str = "unknown",
    fallback_result: Any = None,
    suppress_errors: bool = False,
):
    """
    Decorator for automatic error handling.

    Args:
        component: Component name for error tracking
        operation: Operation name for error tracking
        fallback_result: Result to return if error occurs
        suppress_errors: Whether to suppress errors and return fallback
    """

    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):

            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    result = default_error_handler.handle_error(
                        e, component, operation, {"args": args, "kwargs": kwargs}
                    )

                    if result is not None:
                        return result
                    elif suppress_errors:
                        return fallback_result
                    else:
                        raise

            return async_wrapper
        else:

            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    result = default_error_handler.handle_error(
                        e, component, operation, {"args": args, "kwargs": kwargs}
                    )

                    if result is not None:
                        return result
                    elif suppress_errors:
                        return fallback_result
                    else:
                        raise

            return sync_wrapper

    return decorator


def with_circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: Type[Exception] = Exception,
):
    """
    Decorator to add circuit breaker protection to a function.

    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Time to wait before attempting recovery
        expected_exception: Exception type to catch
    """

    def decorator(func: Callable) -> Callable:
        breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
        )
        return breaker(func)

    return decorator


def with_retry(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    Decorator to add retry logic to a function.

    Args:
        max_attempts: Maximum number of retry attempts
        backoff_factor: Backoff multiplier for retry delays
        exceptions: Exception types to retry on
    """

    def decorator(func: Callable) -> Callable:
        if TENACITY_AVAILABLE:
            # Use tenacity if available
            retry_decorator = retry(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(multiplier=backoff_factor),
                retry=retry_if_exception_type(exceptions),
                reraise=True,
            )
            return retry_decorator(func)
        else:
            # Simple retry implementation
            if asyncio.iscoroutinefunction(func):

                async def async_wrapper(*args, **kwargs):
                    for attempt in range(max_attempts):
                        try:
                            return await func(*args, **kwargs)
                        except exceptions:
                            if attempt == max_attempts - 1:
                                raise
                            await asyncio.sleep(backoff_factor**attempt)

                return async_wrapper
            else:

                def sync_wrapper(*args, **kwargs):
                    for attempt in range(max_attempts):
                        try:
                            return func(*args, **kwargs)
                        except exceptions:
                            if attempt == max_attempts - 1:
                                raise
                            time.sleep(backoff_factor**attempt)

                return sync_wrapper

    return decorator


def get_error_handler() -> ErrorHandler:
    """Get the default error handler instance."""
    return default_error_handler


def with_error_handling(
    component: str,
    operation: str = "unknown",
    fallback_result: Any = None,
    suppress_errors: bool = False,
):
    """
    Decorator for comprehensive error handling.

    Args:
        component: Component name for error tracking
        operation: Operation name for error tracking
        fallback_result: Result to return if error occurs
        suppress_errors: Whether to suppress errors and return fallback
    """
    return handle_errors(component, operation, fallback_result, suppress_errors)

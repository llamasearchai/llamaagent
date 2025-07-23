"""
Monitoring Middleware for FastAPI Integration

This module provides middleware for comprehensive monitoring and metrics collection
for FastAPI applications in the LlamaAgent system.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import json
import logging
import time
import uuid
from typing import Any, Callable, Dict, Optional

from fastapi import FastAPI, Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse


# Simple metrics collector implementation
class MetricsCollector:
    """Simple metrics collector for monitoring."""

    def __init__(self):
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0.0

    def record_http_request(
        self, method: str, endpoint: str, status_code: int, duration: float
    ):
        """Record HTTP request metrics."""
        self.request_count += 1
        self.total_response_time += duration

        if status_code >= 400:
            self.error_count += 1

        # Store metrics
        key = f"{method}:{endpoint}:{status_code}"
        if key not in self.metrics:
            self.metrics[key] = {"count": 0, "total_duration": 0.0}

        self.metrics[key]["count"] += 1
        self.metrics[key]["total_duration"] += duration

    def record_rate_limit_exceeded(self, endpoint: str, user_id: str):
        """Record rate limit exceeded event."""
        key = f"rate_limit:{endpoint}:{user_id}"
        if key not in self.metrics:
            self.metrics[key] = {"count": 0}
        self.metrics[key]["count"] += 1

    def export_metrics(self) -> str:
        """Export metrics in text format."""
        lines = []
        lines.append(f"# Total requests: {self.request_count}")
        lines.append(f"# Total errors: {self.error_count}")
        lines.append(
            f"# Average response time: {self.total_response_time / max(self.request_count, 1):.3f}s"
        )

        for key, data in self.metrics.items():
            lines.append(f"{key}: {data}")
        return "\n".join(lines)

    def get_content_type(self) -> str:
        """Get content type for metrics."""
        return "text/plain"


# Global metrics collector instance
_metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return _metrics_collector


logger = logging.getLogger(__name__)


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware for collecting HTTP request metrics.
    """

    def __init__(self, app: FastAPI):
        super().__init__(app)
        self.metrics = get_metrics_collector()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        request_id = str(uuid.uuid4())
        # Add request ID to headers for tracing
        request.state.request_id = request_id

        # Extract request information
        method = request.method
        path = request.url.path

        # Normalize path for metrics (remove path parameters)
        normalized_path = self._normalize_path(path)
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            logger.error(f"Request {request_id} failed: {e}")
            status_code = 500
            response = JSONResponse(
                status_code=500,
                content={"error": "Internal server error", "request_id": request_id},
            )
        # Calculate duration
        duration = time.time() - start_time

        # Record metrics
        self.metrics.record_http_request(
            method=method,
            endpoint=normalized_path,
            status_code=status_code,
            duration=duration,
        )
        # Add response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration:.3f}s"

        return response

    def _normalize_path(self, path: str) -> str:
        """Normalize path by removing path parameters."""
        # Common path normalizations
        normalizations = [
            (r'/api/v\d+', '/api/v*'),
            (r'/agents/[^/]+', '/agents/{id}'),
            (r'/tasks/[^/]+', '/tasks/{id}'),
            (r'/users/[^/]+', '/users/{id}'),
            (r'/sessions/[^/]+', '/sessions/{id}'),
        ]

        normalized = path
        for pattern, replacement in normalizations:
            import re

            normalized = re.sub(pattern, replacement, normalized)
        return normalized


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for comprehensive request logging.
    """

    def __init__(self, app: FastAPI, log_body: bool = False, max_body_size: int = 1024):
        super().__init__(app)
        self.log_body = log_body
        self.max_body_size = max_body_size

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))

        # Log request start
        request_info = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "headers": dict(request.headers),
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
        }

        # Optionally log request body
        if self.log_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if len(body) <= self.max_body_size:
                    try:
                        request_info["body"] = json.loads(body.decode())
                    except:
                        request_info["body"] = body.decode()[: self.max_body_size]
                else:
                    request_info["body"] = f"[Body too large: {len(body)} bytes]"
            except Exception as e:
                request_info["body"] = f"[Error reading body: {e}]"

        logger.info(f"Request started: {json.dumps(request_info, default=str)}")

        try:
            response = await call_next(request)
            status_code = response.status_code
            success = True
        except Exception as e:
            logger.error(f"Request {request_id} failed: {e}")
            status_code = 500
            success = False
            response = JSONResponse(
                status_code=500,
                content={"error": "Internal server error", "request_id": request_id},
            )
        # Calculate duration
        duration = time.time() - start_time

        # Log request completion
        response_info = {
            "request_id": request_id,
            "status_code": status_code,
            "duration_ms": round(duration * 1000, 2),
            "success": success,
        }

        log_level = logging.INFO if success else logging.ERROR
        logger.log(log_level, f"Request completed: {json.dumps(response_info)}")

        return response


class HealthCheckMiddleware(BaseHTTPMiddleware):
    """
    Middleware for health check monitoring.
    """

    def __init__(self, app: FastAPI, health_check_paths: Optional[list] = None):
        super().__init__(app)
        self.health_check_paths = health_check_paths or ["/health", "/healthz", "/ping"]
        self.metrics = get_metrics_collector()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip detailed monitoring for health check endpoints
        if request.url.path in self.health_check_paths:
            return await call_next(request)
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for rate limiting with metrics integration.
    """

    def __init__(
        self,
        app: FastAPI,
        requests_per_minute: int = 60,
        burst_size: int = 10,
        redis_client: Optional[Any] = None,
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.redis_client = redis_client
        self.metrics = get_metrics_collector()

        # In-memory rate limiting for simple cases
        self.request_counts: Dict[str, Dict[str, Any]] = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        user_id = request.headers.get("X-User-ID", client_ip)
        # Check rate limit
        if await self._is_rate_limited(user_id, request.url.path):
            self.metrics.record_rate_limit_exceeded(
                endpoint=request.url.path, user_id=user_id
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Limit: {self.requests_per_minute} requests per minute",
                },
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                },
            )

        response = await call_next(request)
        # Add rate limit headers
        remaining = await self._get_remaining_requests(user_id)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response

    async def _is_rate_limited(self, user_id: str, endpoint: str) -> bool:
        """Check if user is rate limited."""
        current_time = time.time()
        minute_window = int(current_time // 60)
        # Clean old entries
        self._cleanup_old_entries(current_time)
        # Get current count for this user in this minute
        user_key = f"{user_id}:{minute_window}"
        if user_key not in self.request_counts:
            self.request_counts[user_key] = {
                "count": 0,
                "first_request": current_time,
                "last_request": current_time,
            }

        user_data = self.request_counts[user_key]

        # Check if rate limit exceeded
        if user_data["count"] >= self.requests_per_minute:
            return True

        # Increment count
        user_data["count"] += 1
        user_data["last_request"] = current_time

        return False

    async def _get_remaining_requests(self, user_id: str) -> int:
        """Get remaining requests for user."""
        current_time = time.time()
        minute_window = int(current_time // 60)
        user_key = f"{user_id}:{minute_window}"

        if user_key in self.request_counts:
            return max(
                0, self.requests_per_minute - self.request_counts[user_key]["count"]
            )
        return self.requests_per_minute

    def _cleanup_old_entries(self, current_time: float):
        """Clean up old rate limit entries."""
        cutoff_time = current_time - 120  # Keep 2 minutes of data

        keys_to_remove = []
        for key, data in self.request_counts.items():
            if data["last_request"] < cutoff_time:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del self.request_counts[key]


class ErrorTrackingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for comprehensive error tracking and alerting.
    """

    def __init__(self, app: FastAPI, enable_alerting: bool = True):
        super().__init__(app)
        self.enable_alerting = enable_alerting
        self.metrics = get_metrics_collector()

        # Error tracking
        self.error_counts: Dict[str, int] = {}
        self.last_cleanup = time.time()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
        try:
            response = await call_next(request)
            # Track client errors (4xx)
            if 400 <= response.status_code < 500:
                await self._track_error(
                    error_type="client_error",
                    status_code=response.status_code,
                    endpoint=request.url.path,
                    request_id=request_id,
                )
            return response

        except Exception as e:
            # Track server errors (5xx)
            await self._track_error(
                error_type="server_error",
                status_code=500,
                endpoint=request.url.path,
                request_id=request_id,
                exception=e,
            )
            logger.error(
                f"Unhandled exception in request {request_id}: {e}", exc_info=True
            )
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "request_id": request_id,
                    "message": "An unexpected error occurred. Please try again later.",
                },
            )

    async def _track_error(
        self,
        error_type: str,
        status_code: int,
        endpoint: str,
        request_id: str,
        exception: Optional[Exception] = None,
    ):
        """Track error occurrence."""
        current_time = time.time()

        # Clean up old error counts
        if current_time - self.last_cleanup > 300:  # 5 minutes
            self._cleanup_error_counts(current_time)
            self.last_cleanup = current_time

        # Track error
        error_key = f"{error_type}:{status_code}:{endpoint}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1

        # Log error details
        error_info = {
            "request_id": request_id,
            "error_type": error_type,
            "status_code": status_code,
            "endpoint": endpoint,
            "count": self.error_counts[error_key],
        }

        if exception:
            error_info["exception"] = str(exception)
            error_info["exception_type"] = type(exception).__name__

        logger.warning(f"Error tracked: {json.dumps(error_info)}")

        # Alert if error rate is high
        if (
            self.enable_alerting and self.error_counts[error_key] > 10
        ):  # More than 10 errors
            await self._send_alert(error_key, self.error_counts[error_key])

    def _cleanup_error_counts(self, current_time: float):
        """Clean up old error counts."""
        # Reset counts every 5 minutes for rate calculation
        self.error_counts.clear()

    async def _send_alert(self, error_key: str, count: int):
        """Send alert for high error rate."""
        # This could integrate with alerting systems like PagerDuty, Slack, etc.
        logger.critical(
            f"HIGH ERROR RATE ALERT: {error_key} - {count} errors in 5 minutes"
        )


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Security middleware for monitoring and protection.
    """

    def __init__(self, app: FastAPI, enable_security_headers: bool = True):
        super().__init__(app)
        self.enable_security_headers = enable_security_headers
        self.suspicious_requests: Dict[str, int] = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check for suspicious patterns
        await self._check_suspicious_activity(request)
        response = await call_next(request)
        # Add security headers
        if self.enable_security_headers:
            self._add_security_headers(response)
        return response

    async def _check_suspicious_activity(self, request: Request):
        """Check for suspicious activity patterns."""
        client_ip = request.client.host if request.client else "unknown"

        # Check for common attack patterns
        suspicious_patterns = [
            "../",
            "..\\",
            "<script",
            "javascript:",
            "eval(",
            "union select",
            "drop table",
            "insert into",
            "exec(",
            "system(",
            "cmd.exe",
        ]

        request_data = str(request.url) + str(request.headers)
        for pattern in suspicious_patterns:
            if pattern.lower() in request_data.lower():
                self.suspicious_requests[client_ip] = (
                    self.suspicious_requests.get(client_ip, 0) + 1
                )
                logger.warning(
                    f"Suspicious request from {client_ip}: {pattern} detected"
                )
                break

        # Alert on repeated suspicious activity
        if self.suspicious_requests.get(client_ip, 0) > 5:
            logger.critical(
                f"SECURITY ALERT: Multiple suspicious requests from {client_ip}"
            )

    def _add_security_headers(self, response: Response):
        """Add security headers to response."""
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';",
        }

        for header, value in security_headers.items():
            response.headers[header] = value


def setup_monitoring_middleware(app: FastAPI, config: Optional[Dict[str, Any]] = None):
    """
    Setup comprehensive monitoring middleware for FastAPI app.
    """
    config = config or {}

    # Add middleware in reverse order (last added is executed first)

    # Security middleware
    if config.get("enable_security", True):
        app.add_middleware(SecurityMiddleware)
    # Error tracking middleware
    if config.get("enable_error_tracking", True):
        app.add_middleware(ErrorTrackingMiddleware)
    # Rate limiting middleware
    if config.get("enable_rate_limiting", True):
        app.add_middleware(
            RateLimitMiddleware, requests_per_minute=config.get("rate_limit_rpm", 60)
        )

    # Request logging middleware
    if config.get("enable_request_logging", True):
        app.add_middleware(
            RequestLoggingMiddleware, log_body=config.get("log_request_body", False)
        )

    # Health check middleware
    app.add_middleware(HealthCheckMiddleware)
    # Metrics middleware (should be last/first to execute)
    app.add_middleware(MetricsMiddleware)
    logger.info("Monitoring middleware setup completed")


# Export metrics endpoint
async def metrics_endpoint():
    """Endpoint for Prometheus metrics scraping."""
    metrics = get_metrics_collector()
    return Response(
        content=metrics.export_metrics(), media_type=metrics.get_content_type()
    )

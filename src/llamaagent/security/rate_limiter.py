"""
Rate Limiting System for LlamaAgent

This module provides comprehensive rate limiting with multiple algorithms
and configurable rules for API protection.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class RateLimitAlgorithm(Enum):
    """Rate limiting algorithms."""

    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"


@dataclass
class RateLimitRule:
    """Configuration for a rate limit rule."""

    requests: int
    window_seconds: int
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW
    burst_limit: Optional[int] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class FixedWindow:
    """Fixed window rate limiting algorithm."""

    def __init__(self, rule: RateLimitRule):
        self.rule = rule
        self.windows: Dict[str, Dict[str, Any]] = {}

    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed."""
        current_time = time.time()
        window_start = (
            int(current_time // self.rule.window_seconds) * self.rule.window_seconds
        )

        if identifier not in self.windows:
            self.windows[identifier] = {"start": window_start, "count": 0}

        window = self.windows[identifier]

        # Reset window if expired
        if window["start"] < window_start:
            window["start"] = window_start
            window["count"] = 0

        # Check limit
        if window["count"] >= self.rule.requests:
            return False

        window["count"] += 1
        return True

    def get_retry_after(self, identifier: str) -> float:
        """Get retry after time in seconds."""
        current_time = time.time()
        window_start = (
            int(current_time // self.rule.window_seconds) * self.rule.window_seconds
        )
        next_window = window_start + self.rule.window_seconds
        return next_window - current_time


class SlidingWindow:
    """Sliding window rate limiting algorithm."""

    def __init__(self, rule: RateLimitRule):
        self.rule = rule
        self.requests: Dict[str, deque] = defaultdict(lambda: deque())

    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed."""
        current_time = time.time()
        window_start = current_time - self.rule.window_seconds

        # Clean old requests
        requests = self.requests[identifier]
        while requests and requests[0] < window_start:
            requests.popleft()

        # Check limit
        if len(requests) >= self.rule.requests:
            return False

        requests.append(current_time)
        return True

    def get_retry_after(self, identifier: str) -> float:
        """Get retry after time in seconds."""
        requests = self.requests.get(identifier, deque())
        if requests:
            oldest_request = requests[0]
            return oldest_request + self.rule.window_seconds - time.time()
        return 0


class TokenBucket:
    """Token bucket rate limiting algorithm."""

    def __init__(self, rule: RateLimitRule):
        self.rule = rule
        self.buckets: Dict[str, Dict[str, Any]] = {}
        self.refill_rate = rule.requests / rule.window_seconds

    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed."""
        current_time = time.time()

        if identifier not in self.buckets:
            self.buckets[identifier] = {
                "tokens": float(self.rule.requests),
                "last_refill": current_time,
            }

        bucket = self.buckets[identifier]

        # Refill tokens
        time_passed = current_time - bucket["last_refill"]
        tokens_to_add = time_passed * self.refill_rate
        bucket["tokens"] = min(self.rule.requests, bucket["tokens"] + tokens_to_add)
        bucket["last_refill"] = current_time

        # Check if token available
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True

        return False

    def get_retry_after(self, identifier: str) -> float:
        """Get retry after time in seconds."""
        return 1.0 / self.refill_rate


class RateLimiter:
    """
    Rate limiter with configurable algorithms and rules.
    """

    def __init__(self):
        self.rules: Dict[str, RateLimitRule] = {}
        self.algorithms: Dict[str, Any] = {}
        self.active_identifiers: Dict[str, int] = defaultdict(int)

    def add_rule(self, name: str, rule: RateLimitRule) -> None:
        """Add a rate limiting rule."""
        self.rules[name] = rule

        # Create algorithm instance
        if rule.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            self.algorithms[name] = FixedWindow(rule)
        elif rule.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            self.algorithms[name] = SlidingWindow(rule)
        elif rule.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            self.algorithms[name] = TokenBucket(rule)
        else:
            raise ValueError(f"Unknown algorithm: {rule.algorithm}")

    def remove_rule(self, name: str) -> bool:
        """Remove a rate limiting rule."""
        if name in self.rules:
            del self.rules[name]
            del self.algorithms[name]
            return True
        return False

    def is_allowed(self, rule_name: str, identifier: str) -> bool:
        """Check if request is allowed for a rule and identifier."""
        if rule_name not in self.algorithms:
            logger.warning(f"Rate limit rule not found: {rule_name}")
            return True  # Allow if rule doesn't exist

        algorithm = self.algorithms[rule_name]
        allowed = algorithm.is_allowed(identifier)

        if allowed:
            self.active_identifiers[identifier] += 1

        return allowed

    def get_retry_after(self, rule_name: str, identifier: str) -> float:
        """Get retry after time in seconds."""
        if rule_name not in self.algorithms:
            return 0

        algorithm = self.algorithms[rule_name]
        return algorithm.get_retry_after(identifier)

    def enforce_limit(self, rule_name: str, identifier: str) -> None:
        """Enforce rate limit, raise exception if exceeded."""
        if not self.is_allowed(rule_name, identifier):
            retry_after = self.get_retry_after(rule_name, identifier)
            raise RateLimitExceeded(
                f"Rate limit exceeded for {rule_name}", retry_after=retry_after
            )

    def get_status(self, rule_name: str, identifier: str) -> Dict[str, Any]:
        """Get rate limit status for a rule and identifier."""
        if rule_name not in self.rules:
            return {"error": "Rule not found"}

        rule = self.rules[rule_name]
        algorithm = self.algorithms[rule_name]

        # Get current usage based on algorithm type
        if isinstance(algorithm, SlidingWindow):
            current_time = time.time()
            window_start = current_time - rule.window_seconds
            requests = algorithm.requests.get(identifier, deque())
            current_count = sum(1 for req_time in requests if req_time > window_start)
        elif isinstance(algorithm, FixedWindow):
            window = algorithm.windows.get(identifier, {"count": 0})
            current_count = window["count"]
        elif isinstance(algorithm, TokenBucket):
            bucket = algorithm.buckets.get(identifier, {"tokens": rule.requests})
            current_count = rule.requests - int(bucket["tokens"])
        else:
            current_count = 0

        return {
            "rule_name": rule_name,
            "identifier": identifier,
            "limit": rule.requests,
            "window_seconds": rule.window_seconds,
            "current_count": current_count,
            "remaining": max(0, rule.requests - current_count),
            "retry_after": (
                self.get_retry_after(rule_name, identifier)
                if current_count >= rule.requests
                else 0
            ),
        }

    def cleanup_expired(self, max_age_seconds: int = 3600) -> None:
        """Clean up expired data."""
        current_time = time.time()
        cutoff_time = current_time - max_age_seconds

        for algorithm in self.algorithms.values():
            if isinstance(algorithm, SlidingWindow):
                for identifier in list(algorithm.requests.keys()):
                    requests = algorithm.requests[identifier]
                    while requests and requests[0] < cutoff_time:
                        requests.popleft()
                    if not requests:
                        del algorithm.requests[identifier]

            elif isinstance(algorithm, FixedWindow):
                for identifier in list(algorithm.windows.keys()):
                    window = algorithm.windows[identifier]
                    if window["start"] < cutoff_time:
                        del algorithm.windows[identifier]

        # Clean active identifiers
        self.active_identifiers = defaultdict(int)

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            "total_rules": len(self.rules),
            "active_identifiers": len(self.active_identifiers),
            "rules": {
                name: {
                    "requests": rule.requests,
                    "window_seconds": rule.window_seconds,
                    "algorithm": rule.algorithm.value,
                }
                for name, rule in self.rules.items()
            },
        }


class AsyncRateLimiter:
    """
    Async wrapper for rate limiter with additional async features.
    """

    def __init__(self):
        self.rate_limiter = RateLimiter()
        self._lock = asyncio.Lock()

    async def add_rule(self, name: str, rule: RateLimitRule) -> None:
        """Add a rate limiting rule."""
        async with self._lock:
            self.rate_limiter.add_rule(name, rule)

    async def remove_rule(self, name: str) -> bool:
        """Remove a rate limiting rule."""
        async with self._lock:
            return self.rate_limiter.remove_rule(name)

    async def is_allowed(self, rule_name: str, identifier: str) -> bool:
        """Check if request is allowed for a rule and identifier."""
        async with self._lock:
            return self.rate_limiter.is_allowed(rule_name, identifier)

    async def enforce_limit(self, rule_name: str, identifier: str) -> None:
        """Enforce rate limit, raise exception if exceeded."""
        async with self._lock:
            self.rate_limiter.enforce_limit(rule_name, identifier)

    async def get_status(self, rule_name: str, identifier: str) -> Dict[str, Any]:
        """Get rate limit status for a rule and identifier."""
        async with self._lock:
            return self.rate_limiter.get_status(rule_name, identifier)

    async def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        async with self._lock:
            return self.rate_limiter.get_stats()

    async def cleanup_expired(self, max_age_seconds: int = 3600) -> None:
        """Clean up expired data."""
        async with self._lock:
            self.rate_limiter.cleanup_expired(max_age_seconds)


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create global rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()

        # Add default rules
        _rate_limiter.add_rule(
            "api_general",
            RateLimitRule(
                requests=100,
                window_seconds=3600,
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
            ),
        )

        _rate_limiter.add_rule(
            "auth",
            RateLimitRule(
                requests=5,
                window_seconds=300,
                algorithm=RateLimitAlgorithm.FIXED_WINDOW,
            ),
        )

        _rate_limiter.add_rule(
            "heavy_ops",
            RateLimitRule(
                requests=10,
                window_seconds=3600,
                algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
            ),
        )

    return _rate_limiter


# Convenience functions
def is_rate_limited(rule_name: str, identifier: str) -> bool:
    """Check if identifier is rate limited."""
    return not get_rate_limiter().is_allowed(rule_name, identifier)


def enforce_rate_limit(rule_name: str, identifier: str) -> None:
    """Enforce rate limit, raise exception if exceeded."""
    get_rate_limiter().enforce_limit(rule_name, identifier)

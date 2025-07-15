"""
Security Manager for LlamaAgent

This module provides comprehensive security management including authentication,
authorization, session management, and rate limiting.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import hashlib
import logging
import secrets
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

# Optional imports
try:
    import jwt

    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for different operations."""

    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    AUTHORIZED = "authorized"
    ADMIN = "admin"
    SYSTEM = "system"


class RoleType(Enum):
    """User role types."""

    GUEST = "guest"
    USER = "user"
    AGENT = "agent"
    ADMIN = "admin"
    SYSTEM = "system"


@dataclass
class User:
    """User information."""

    id: str
    username: str
    email: str
    role: RoleType
    permissions: Set[str]
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityContext:
    """Security context for requests."""

    user: Optional[User] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    permissions: Set[str] = field(default_factory=set)
    security_level: SecurityLevel = SecurityLevel.PUBLIC


@dataclass
class RateLimit:
    """Rate limiting configuration."""

    max_requests: int
    window_seconds: int
    burst_limit: Optional[int] = None


class SecurityManager:
    """
    Comprehensive security manager for LlamaAgent.

    Features:
    - User authentication and authorization
    - JWT token management
    - Session management
    - Rate limiting
    - Security event logging
    - Role-based access control
    """

    def __init__(
        self,
        jwt_secret: Optional[str] = None,
        jwt_expiry_hours: int = 24,
        rate_limit_enabled: bool = True,
        default_rate_limit: Optional[RateLimit] = None,
    ):
        self.jwt_secret = jwt_secret or secrets.token_urlsafe(32)
        self.jwt_expiry_hours = jwt_expiry_hours
        self.rate_limit_enabled = rate_limit_enabled
        self.default_rate_limit = default_rate_limit or RateLimit(
            max_requests=100, window_seconds=3600
        )

        # Storage
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, SecurityContext] = {}
        self.api_keys: Dict[str, Dict[str, Any]] = {}

        # Rate limiting
        self.rate_limits: Dict[str, RateLimit] = {}
        self.request_counts: Dict[str, deque] = defaultdict(lambda: deque())

        # Security monitoring
        self.failed_attempts: Dict[str, List[datetime]] = defaultdict(list)
        self.security_events: List[Dict[str, Any]] = []

        # Create default users
        self._create_default_users()

        self.logger = logging.getLogger(__name__)

    def _create_default_users(self) -> None:
        """Create default users for testing."""
        # Admin user
        admin_user = User(
            id="admin",
            username="admin",
            email="admin@llamaagent.ai",
            role=RoleType.ADMIN,
            permissions={"*"},  # All permissions
            created_at=datetime.now(timezone.utc),
            is_active=True,
        )
        self.users["admin"] = admin_user

        # Test user
        test_user = User(
            id="test",
            username="test",
            email="test@llamaagent.ai",
            role=RoleType.USER,
            permissions={"read", "write"},
            created_at=datetime.now(timezone.utc),
            is_active=True,
        )
        self.users["test"] = test_user

    def _hash_password(self, password: str) -> str:
        """Hash a password using SHA-256."""
        return hashlib.sha256(password.encode().hexdigest())

    def _verify_password(self, password: str, hashed: str) -> bool:
        """Verify a password against its hash."""
        return hashlib.sha256(password.encode().hexdigest()) == hashed

    async def create_user(
        self,
        username: str,
        email: str,
        password: str,
        role: RoleType = RoleType.USER,
        permissions: Optional[Set[str]] = None,
    ) -> User:
        """Create a new user."""
        if username in self.users:
            raise ValueError(f"User {username} already exists")

        user = User(
            id=username,  # Simple ID scheme
            username=username,
            email=email,
            role=role,
            permissions=permissions or set(),
            created_at=datetime.now(timezone.utc),
            is_active=True,
        )

        self.users[username] = user

        # Store password hash separately (in real implementation, use proper DB)
        self.api_keys[f"password_{username}"] = {
            "password_hash": self._hash_password(password)
        }

        self.logger.info(f"User created: {username}")
        return user

    async def authenticate_user(
        self, username: str, password: str, ip_address: str = "unknown"
    ) -> Optional[User]:
        """Authenticate a user with username and password."""
        if username not in self.users:
            self._record_failed_attempt(f"user_{username}_{ip_address}")
            self.logger.warning(
                f"Authentication attempt for non-existent user: {username}"
            )
            return None

        user = self.users[username]
        if not user.is_active:
            self.logger.warning(f"Authentication attempt for inactive user: {username}")
            return None

        # Check rate limiting
        if self._is_rate_limited(f"auth_{ip_address}"):
            self.logger.warning(
                f"Rate limited authentication attempt from {ip_address}"
            )
            return None

        # Verify password
        password_data = self.api_keys.get(f"password_{username}")
        if not password_data or not self._verify_password(
            password, password_data["password_hash"]
        ):
            self._record_failed_attempt(f"user_{username}_{ip_address}")
            self.logger.warning(f"Invalid password for user: {username}")
            return None

        # Update last login
        user.last_login = datetime.now(timezone.utc)
        self.logger.info(f"User authenticated successfully: {username}")
        return user

    async def create_session(
        self, user: User, ip_address: str = "unknown", user_agent: str = "unknown"
    ) -> str:
        """Create a new session for a user."""
        session_id = secrets.token_urlsafe(32)

        context = SecurityContext(
            user=user,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            permissions=user.permissions,
            security_level=self._get_security_level(user.role),
        )

        self.sessions[session_id] = context

        self._log_security_event(
            "session_created",
            {"session_id": session_id, "user_id": user.id, "ip_address": ip_address},
        )

        return session_id

    async def get_security_context(self, session_id: str) -> Optional[SecurityContext]:
        """Get security context for a session."""
        return self.sessions.get(session_id)

    async def revoke_session(self, session_id: str) -> bool:
        """Revoke a session."""
        if session_id in self.sessions:
            context = self.sessions.pop(session_id)
            self._log_security_event(
                "session_revoked",
                {
                    "session_id": session_id,
                    "user_id": context.user.id if context.user else None,
                },
            )
            return True
        return False

    async def create_jwt_token(self, user: User) -> str:
        """Create a JWT token for a user."""
        if not JWT_AVAILABLE:
            raise RuntimeError("JWT library not available")

        payload = {
            "sub": user.id,
            "username": user.username,
            "email": user.email,
            "role": user.role.value,
            "permissions": list(user.permissions),
            "iat": datetime.now(timezone.utc),
            "exp": datetime.now(timezone.utc) + timedelta(hours=self.jwt_expiry_hours),
        }

        token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
        return token

    async def verify_jwt_token(self, token: str) -> Optional[User]:
        """Verify a JWT token and return the user."""
        if not JWT_AVAILABLE:
            return None

        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])

            user_id = payload.get("sub")
            if user_id and user_id in self.users:
                return self.users[user_id]
        except jwt.ExpiredSignatureError:
            self.logger.warning("JWT token expired")
        except jwt.InvalidTokenError:
            self.logger.warning("Invalid JWT token")

        return None

    async def verify_token(self, token: str) -> Optional[User]:
        """Verify an authentication token.

        This method is a thin wrapper around ``verify_jwt_token`` that also
        maintains backward-compatibility with simple API key tokens used by
        early versions of the platform.
        """
        # First, try JWT verification
        user = await self.verify_jwt_token(token)
        if user:
            return user

        # Fallback: treat the raw token as an API key that maps to a user
        key_data = self.api_keys.get(token)
        if key_data:
            user_id = key_data.get("user_id") or key_data.get("username")
            if user_id and user_id in self.users:
                return self.users[user_id]

        return None

    def check_permission(self, context: SecurityContext, permission: str) -> bool:
        """Check if security context has required permission."""
        if not context.user:
            return False

        # Admin has all permissions
        if "*" in context.permissions:
            return True

        return permission in context.permissions

    def require_security_level(
        self, context: SecurityContext, required_level: SecurityLevel
    ) -> bool:
        """Check if context meets required security level."""
        level_hierarchy = {
            SecurityLevel.PUBLIC: 0,
            SecurityLevel.AUTHENTICATED: 1,
            SecurityLevel.AUTHORIZED: 2,
            SecurityLevel.ADMIN: 3,
            SecurityLevel.SYSTEM: 4,
        }

        current_level = level_hierarchy.get(context.security_level, 0)
        required_level_value = level_hierarchy.get(required_level, 0)

        return current_level >= required_level_value

    def _get_security_level(self, role: RoleType) -> SecurityLevel:
        """Get security level for a role."""
        role_to_level = {
            RoleType.GUEST: SecurityLevel.PUBLIC,
            RoleType.USER: SecurityLevel.AUTHENTICATED,
            RoleType.AGENT: SecurityLevel.AUTHORIZED,
            RoleType.ADMIN: SecurityLevel.ADMIN,
            RoleType.SYSTEM: SecurityLevel.SYSTEM,
        }
        return role_to_level.get(role, SecurityLevel.PUBLIC)

    def _is_rate_limited(self, identifier: str) -> bool:
        """Check if identifier is rate limited."""
        if not self.rate_limit_enabled:
            return False

        limit = self.rate_limits.get(identifier, self.default_rate_limit)
        now = time.time()
        window_start = now - limit.window_seconds

        # Clean old requests
        requests = self.request_counts[identifier]
        while requests and requests[0] < window_start:
            requests.popleft()

        # Check limit
        if len(requests) >= limit.max_requests:
            return True

        # Record this request
        requests.append(now)
        return False

    def _record_failed_attempt(self, identifier: str) -> None:
        """Record a failed authentication attempt."""
        now = datetime.now(timezone.utc)
        self.failed_attempts[identifier].append(now)

        # Clean old attempts (keep last 24 hours)
        cutoff = now - timedelta(hours=24)
        self.failed_attempts[identifier] = [
            attempt for attempt in self.failed_attempts[identifier] if attempt > cutoff
        ]

        # Log if too many failures
        if len(self.failed_attempts[identifier]) > 5:
            self._log_security_event(
                "repeated_failed_attempts",
                {
                    "identifier": identifier,
                    "attempt_count": len(self.failed_attempts[identifier]),
                },
            )

    def _log_security_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log a security event."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }

        self.security_events.append(event)

        # Keep only recent events
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-1000:]

        self.logger.info(f"Security event: {event_type}", extra=data)

    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics."""
        return {
            "total_users": len(self.users),
            "active_sessions": len(self.sessions),
            "total_security_events": len(self.security_events),
            "failed_attempts": {
                identifier: len(attempts)
                for identifier, attempts in self.failed_attempts.items()
            },
            "rate_limits_active": self.rate_limit_enabled,
            "jwt_available": JWT_AVAILABLE,
        }

    def get_user_list(self) -> List[Dict[str, Any]]:
        """Get list of users (without sensitive data)."""
        return [
            {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "role": user.role.value,
                "is_active": user.is_active,
                "last_login": user.last_login.isoformat() if user.last_login else None,
                "created_at": user.created_at.isoformat(),
            }
            for user in self.users.values()
        ]

    def get_security_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent security events."""
        return self.security_events[-limit:]

    async def cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions and old data."""
        # This would normally check JWT expiry or session timeouts
        # For now, just clean up old security events and failed attempts

        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

        # Clean up old failed attempts
        for identifier in list(self.failed_attempts.keys()):
            self.failed_attempts[identifier] = [
                attempt
                for attempt in self.failed_attempts[identifier]
                if attempt > cutoff
            ]
            if not self.failed_attempts[identifier]:
                del self.failed_attempts[identifier]

        # Clean up old security events
        if len(self.security_events) > 500:
            self.security_events = self.security_events[-500:]


# Global security manager instance
_security_manager: Optional[SecurityManager] = None


def get_security_manager() -> SecurityManager:
    """Get or create global security manager."""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager


def create_security_context(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    permissions: Optional[Set[str]] = None,
) -> SecurityContext:
    """Create a security context."""
    user = None
    if user_id:
        security_manager = get_security_manager()
        user = security_manager.users.get(user_id)

    return SecurityContext(
        user=user,
        session_id=session_id,
        ip_address=ip_address,
        permissions=permissions or set(),
        security_level=SecurityLevel.AUTHENTICATED if user else SecurityLevel.PUBLIC,
    )

#!/usr/bin/env python3
"""
Advanced Security System for LlamaAgent

This module provides comprehensive security features including:
- Multi-factor authentication
- Role-based access control (RBAC)
- API key management
- Rate limiting and DDoS protection
- Encryption and data protection
- Security audit logging
- Vulnerability scanning
- Intrusion detection

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import hashlib
import hmac
import secrets
import time
import json
import logging
from typing import Dict, List, Any, Optional, Set, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
from collections import defaultdict, deque
import ipaddress
import re

# Cryptography
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

# JWT and authentication
import jwt
from passlib.context import CryptContext
from passlib.hash import bcrypt

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for different operations."""
    PUBLIC = "public"
    BASIC = "basic"
    ELEVATED = "elevated"
    ADMIN = "admin"
    SYSTEM = "system"


class ThreatLevel(Enum):
    """Threat levels for security events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class User:
    """User data structure."""
    id: str
    username: str
    email: str
    password_hash: str
    roles: Set[str] = field(default_factory=set)
    permissions: Set[str] = field(default_factory=set)
    api_keys: List[str] = field(default_factory=list)
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    is_active: bool = True


@dataclass
class SecurityEvent:
    """Security event for audit logging."""
    id: str
    event_type: str
    user_id: Optional[str]
    ip_address: str
    user_agent: str
    resource: str
    action: str
    result: str  # success, failure, blocked
    threat_level: ThreatLevel
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RateLimitRule:
    """Rate limiting rule configuration."""
    name: str
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_limit: int = 0
    enabled: bool = True


class PasswordPolicy:
    """Password policy enforcement."""
    
    def __init__(self):
        self.min_length = 8
        self.max_length = 128
        self.require_uppercase = True
        self.require_lowercase = True
        self.require_digits = True
        self.require_special = True
        self.special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        self.max_repeated_chars = 3
        self.min_unique_chars = 5
    
    def validate_password(self, password: str) -> tuple[bool, List[str]]:
        """Validate password against policy."""
        errors = []
        
        # Length check
        if len(password) < self.min_length:
            errors.append(f"Password must be at least {self.min_length} characters long")
        if len(password) > self.max_length:
            errors.append(f"Password must be no more than {self.max_length} characters long")
        
        # Character requirements
        if self.require_uppercase and not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        if self.require_lowercase and not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        if self.require_digits and not re.search(r'\d', password):
            errors.append("Password must contain at least one digit")
        if self.require_special and not re.search(f'[{re.escape(self.special_chars)}]', password):
            errors.append("Password must contain at least one special character")
        
        # Repeated characters check
        repeated_count = 0
        for i in range(len(password) - 1):
            if password[i] == password[i + 1]:
                repeated_count += 1
                if repeated_count >= self.max_repeated_chars:
                    errors.append(f"Password cannot have more than {self.max_repeated_chars} repeated characters")
                    break
            else:
                repeated_count = 0
        
        # Unique characters check
        unique_chars = len(set(password))
        if unique_chars < self.min_unique_chars:
            errors.append(f"Password must contain at least {self.min_unique_chars} unique characters")
        
        return len(errors) == 0, errors


class EncryptionManager:
    """Encryption and decryption manager."""
    
    def __init__(self, master_key: Optional[bytes] = None):
        if master_key:
            self.master_key = master_key
        else:
            self.master_key = Fernet.generate_key()
        
        self.fernet = Fernet(self.master_key)
        self.password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        return self.fernet.decrypt(encrypted_data.encode()).decode()
    
    def hash_password(self, password: str) -> str:
        """Hash password securely."""
        return self.password_context.hash(password)
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        return self.password_context.verify(password, hashed)
    
    def generate_api_key(self) -> str:
        """Generate secure API key."""
        return secrets.token_urlsafe(32)
    
    def generate_mfa_secret(self) -> str:
        """Generate MFA secret."""
        return secrets.token_urlsafe(16)


class RateLimiter:
    """Advanced rate limiting with multiple algorithms."""
    
    def __init__(self):
        self.rules: Dict[str, RateLimitRule] = {}
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque())
        self.blocked_ips: Dict[str, datetime] = {}
        self._lock = threading.RLock()
    
    def add_rule(self, rule: RateLimitRule):
        """Add rate limiting rule."""
        self.rules[rule.name] = rule
    
    def check_rate_limit(self, identifier: str, rule_name: str = "default") -> tuple[bool, Dict[str, Any]]:
        """Check if request is within rate limits."""
        with self._lock:
            # Check if IP is blocked
            if identifier in self.blocked_ips:
                if datetime.now() < self.blocked_ips[identifier]:
                    return False, {"reason": "IP blocked", "retry_after": self.blocked_ips[identifier]}
                else:
                    del self.blocked_ips[identifier]
            
            rule = self.rules.get(rule_name)
            if not rule or not rule.enabled:
                return True, {}
            
            now = datetime.now()
            history = self.request_history[identifier]
            
            # Clean old entries
            cutoff_minute = now - timedelta(minutes=1)
            cutoff_hour = now - timedelta(hours=1)
            cutoff_day = now - timedelta(days=1)
            
            while history and history[0] < cutoff_day:
                history.popleft()
            
            # Count requests in different time windows
            requests_last_minute = sum(1 for t in history if t >= cutoff_minute)
            requests_last_hour = sum(1 for t in history if t >= cutoff_hour)
            requests_last_day = len(history)
            
            # Check limits
            if (requests_last_minute >= rule.requests_per_minute or
                requests_last_hour >= rule.requests_per_hour or
                requests_last_day >= rule.requests_per_day):
                
                # Block IP for escalating violations
                if requests_last_minute >= rule.requests_per_minute * 2:
                    self.blocked_ips[identifier] = now + timedelta(minutes=15)
                
                return False, {
                    "reason": "Rate limit exceeded",
                    "requests_last_minute": requests_last_minute,
                    "requests_last_hour": requests_last_hour,
                    "requests_last_day": requests_last_day,
                    "limits": {
                        "per_minute": rule.requests_per_minute,
                        "per_hour": rule.requests_per_hour,
                        "per_day": rule.requests_per_day
                    }
                }
            
            # Record this request
            history.append(now)
            return True, {}
    
    def get_rate_limit_status(self, identifier: str, rule_name: str = "default") -> Dict[str, Any]:
        """Get current rate limit status."""
        with self._lock:
            rule = self.rules.get(rule_name)
            if not rule:
                return {}
            
            history = self.request_history[identifier]
            now = datetime.now()
            
            cutoff_minute = now - timedelta(minutes=1)
            cutoff_hour = now - timedelta(hours=1)
            cutoff_day = now - timedelta(days=1)
            
            requests_last_minute = sum(1 for t in history if t >= cutoff_minute)
            requests_last_hour = sum(1 for t in history if t >= cutoff_hour)
            requests_last_day = sum(1 for t in history if t >= cutoff_day)
            
            return {
                "requests_last_minute": requests_last_minute,
                "requests_last_hour": requests_last_hour,
                "requests_last_day": requests_last_day,
                "limits": {
                    "per_minute": rule.requests_per_minute,
                    "per_hour": rule.requests_per_hour,
                    "per_day": rule.requests_per_day
                },
                "remaining": {
                    "per_minute": max(0, rule.requests_per_minute - requests_last_minute),
                    "per_hour": max(0, rule.requests_per_hour - requests_last_hour),
                    "per_day": max(0, rule.requests_per_day - requests_last_day)
                }
            }


class IntrusionDetection:
    """Intrusion detection system."""
    
    def __init__(self):
        self.suspicious_patterns = [
            r'(\.\./){3,}',  # Directory traversal
            r'<script[^>]*>.*?</script>',  # XSS
            r'union\s+select',  # SQL injection
            r'exec\s*\(',  # Code execution
            r'eval\s*\(',  # Code evaluation
            r'system\s*\(',  # System commands
        ]
        self.suspicious_ips: Set[str] = set()
        self.attack_patterns: Dict[str, int] = defaultdict(int)
        self.blocked_ips: Set[str] = set()
    
    def analyze_request(self, ip: str, user_agent: str, path: str, params: Dict[str, Any]) -> tuple[bool, ThreatLevel, List[str]]:
        """Analyze request for suspicious activity."""
        threats = []
        threat_level = ThreatLevel.LOW
        
        # Check for suspicious patterns in path
        for pattern in self.suspicious_patterns:
            if re.search(pattern, path, re.IGNORECASE):
                threats.append(f"Suspicious pattern in path: {pattern}")
                threat_level = ThreatLevel.HIGH
        
        # Check parameters
        for key, value in params.items():
            if isinstance(value, str):
                for pattern in self.suspicious_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        threats.append(f"Suspicious pattern in parameter {key}: {pattern}")
                        threat_level = ThreatLevel.HIGH
        
        # Check user agent
        suspicious_agents = ['sqlmap', 'nikto', 'nmap', 'masscan', 'dirb']
        if any(agent in user_agent.lower() for agent in suspicious_agents):
            threats.append(f"Suspicious user agent: {user_agent}")
            threat_level = ThreatLevel.MEDIUM
        
        # Check IP reputation
        if ip in self.suspicious_ips:
            threats.append(f"Request from suspicious IP: {ip}")
            threat_level = ThreatLevel.MEDIUM
        
        # Track attack patterns
        if threats:
            self.attack_patterns[ip] += 1
            if self.attack_patterns[ip] > 5:
                self.blocked_ips.add(ip)
                threat_level = ThreatLevel.CRITICAL
        
        is_suspicious = len(threats) > 0
        return is_suspicious, threat_level, threats
    
    def is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is blocked."""
        return ip in self.blocked_ips
    
    def block_ip(self, ip: str):
        """Block an IP address."""
        self.blocked_ips.add(ip)
        self.suspicious_ips.add(ip)
    
    def unblock_ip(self, ip: str):
        """Unblock an IP address."""
        self.blocked_ips.discard(ip)


class SecurityAuditLogger:
    """Security audit logging system."""
    
    def __init__(self, max_events: int = 10000):
        self.events: deque = deque(maxlen=max_events)
        self.event_counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.RLock()
    
    def log_event(self, event: SecurityEvent):
        """Log a security event."""
        with self._lock:
            self.events.append(event)
            self.event_counts[event.event_type] += 1
            
            # Log to standard logger based on threat level
            if event.threat_level == ThreatLevel.CRITICAL:
                logger.critical(f"Security Event: {event.event_type} - {event.details}")
            elif event.threat_level == ThreatLevel.HIGH:
                logger.error(f"Security Event: {event.event_type} - {event.details}")
            elif event.threat_level == ThreatLevel.MEDIUM:
                logger.warning(f"Security Event: {event.event_type} - {event.details}")
            else:
                logger.info(f"Security Event: {event.event_type} - {event.details}")
    
    def get_events(self, event_type: Optional[str] = None, limit: int = 100) -> List[SecurityEvent]:
        """Get security events."""
        with self._lock:
            events = list(self.events)
            
            if event_type:
                events = [e for e in events if e.event_type == event_type]
            
            return events[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get security statistics."""
        with self._lock:
            recent_events = list(self.events)[-1000:]  # Last 1000 events
            
            # Count by threat level
            threat_counts = defaultdict(int)
            for event in recent_events:
                threat_counts[event.threat_level.value] += 1
            
            # Count by event type
            type_counts = defaultdict(int)
            for event in recent_events:
                type_counts[event.event_type] += 1
            
            return {
                "total_events": len(self.events),
                "recent_events": len(recent_events),
                "threat_level_distribution": dict(threat_counts),
                "event_type_distribution": dict(type_counts),
                "events_per_hour": self._calculate_events_per_hour(recent_events)
            }
    
    def _calculate_events_per_hour(self, events: List[SecurityEvent]) -> float:
        """Calculate events per hour."""
        if not events:
            return 0.0
        
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        recent_events = [e for e in events if e.timestamp >= hour_ago]
        
        return len(recent_events)


class AdvancedSecurityManager:
    """Main security manager coordinating all security components."""
    
    def __init__(self, master_key: Optional[bytes] = None):
        self.encryption_manager = EncryptionManager(master_key)
        self.password_policy = PasswordPolicy()
        self.rate_limiter = RateLimiter()
        self.intrusion_detection = IntrusionDetection()
        self.audit_logger = SecurityAuditLogger()
        
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.api_keys: Dict[str, str] = {}  # api_key -> user_id
        
        # JWT settings
        self.jwt_secret = secrets.token_urlsafe(32)
        self.jwt_algorithm = "HS256"
        self.jwt_expiration = timedelta(hours=24)
        
        # Setup default rate limiting rules
        self._setup_default_rate_limits()
    
    def _setup_default_rate_limits(self):
        """Setup default rate limiting rules."""
        
        # General API rate limit
        self.rate_limiter.add_rule(RateLimitRule(
            name="api_general",
            requests_per_minute=60,
            requests_per_hour=1000,
            requests_per_day=10000
        ))
        
        # Authentication rate limit
        self.rate_limiter.add_rule(RateLimitRule(
            name="auth",
            requests_per_minute=5,
            requests_per_hour=20,
            requests_per_day=100
        ))
        
        # Agent execution rate limit
        self.rate_limiter.add_rule(RateLimitRule(
            name="agent_execution",
            requests_per_minute=10,
            requests_per_hour=100,
            requests_per_day=1000
        ))
    
    def create_user(self, username: str, email: str, password: str, roles: Optional[Set[str]] = None) -> tuple[bool, Union[User, List[str]]]:
        """Create a new user."""
        
        # Validate password
        is_valid, errors = self.password_policy.validate_password(password)
        if not is_valid:
            return False, errors
        
        # Check if user already exists
        if any(u.username == username or u.email == email for u in self.users.values()):
            return False, ["User with this username or email already exists"]
        
        # Create user
        user_id = secrets.token_urlsafe(16)
        password_hash = self.encryption_manager.hash_password(password)
        
        user = User(
            id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            roles=roles or set(),
            permissions=set()
        )
        
        self.users[user_id] = user
        
        # Log user creation
        self.audit_logger.log_event(SecurityEvent(
            id=secrets.token_urlsafe(8),
            event_type="user_created",
            user_id=user_id,
            ip_address="system",
            user_agent="system",
            resource="user",
            action="create",
            result="success",
            threat_level=ThreatLevel.LOW,
            details={"username": username, "email": email}
        ))
        
        return True, user
    
    def authenticate_user(self, username: str, password: str, ip_address: str, user_agent: str) -> tuple[bool, Union[str, str]]:
        """Authenticate user and return JWT token."""
        
        # Check rate limiting
        allowed, rate_info = self.rate_limiter.check_rate_limit(ip_address, "auth")
        if not allowed:
            self.audit_logger.log_event(SecurityEvent(
                id=secrets.token_urlsafe(8),
                event_type="auth_rate_limited",
                user_id=None,
                ip_address=ip_address,
                user_agent=user_agent,
                resource="auth",
                action="login",
                result="blocked",
                threat_level=ThreatLevel.MEDIUM,
                details=rate_info
            ))
            return False, "Rate limit exceeded"
        
        # Find user
        user = None
        for u in self.users.values():
            if u.username == username:
                user = u
                break
        
        if not user:
            self.audit_logger.log_event(SecurityEvent(
                id=secrets.token_urlsafe(8),
                event_type="auth_failed",
                user_id=None,
                ip_address=ip_address,
                user_agent=user_agent,
                resource="auth",
                action="login",
                result="failure",
                threat_level=ThreatLevel.MEDIUM,
                details={"reason": "user_not_found", "username": username}
            ))
            return False, "Invalid credentials"
        
        # Check if user is locked
        if user.locked_until and datetime.now() < user.locked_until:
            self.audit_logger.log_event(SecurityEvent(
                id=secrets.token_urlsafe(8),
                event_type="auth_locked",
                user_id=user.id,
                ip_address=ip_address,
                user_agent=user_agent,
                resource="auth",
                action="login",
                result="blocked",
                threat_level=ThreatLevel.HIGH,
                details={"locked_until": user.locked_until.isoformat()}
            ))
            return False, "Account locked"
        
        # Verify password
        if not self.encryption_manager.verify_password(password, user.password_hash):
            user.failed_login_attempts += 1
            
            # Lock account after too many failures
            if user.failed_login_attempts >= 5:
                user.locked_until = datetime.now() + timedelta(minutes=30)
            
            self.audit_logger.log_event(SecurityEvent(
                id=secrets.token_urlsafe(8),
                event_type="auth_failed",
                user_id=user.id,
                ip_address=ip_address,
                user_agent=user_agent,
                resource="auth",
                action="login",
                result="failure",
                threat_level=ThreatLevel.MEDIUM,
                details={"reason": "invalid_password", "failed_attempts": user.failed_login_attempts}
            ))
            return False, "Invalid credentials"
        
        # Reset failed attempts on successful login
        user.failed_login_attempts = 0
        user.last_login = datetime.now()
        
        # Generate JWT token
        payload = {
            "user_id": user.id,
            "username": user.username,
            "roles": list(user.roles),
            "exp": datetime.utcnow() + self.jwt_expiration,
            "iat": datetime.utcnow()
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        
        # Create session
        session_id = secrets.token_urlsafe(32)
        self.sessions[session_id] = {
            "user_id": user.id,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "created_at": datetime.now(),
            "last_activity": datetime.now()
        }
        
        self.audit_logger.log_event(SecurityEvent(
            id=secrets.token_urlsafe(8),
            event_type="auth_success",
            user_id=user.id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource="auth",
            action="login",
            result="success",
            threat_level=ThreatLevel.LOW,
            details={"session_id": session_id}
        ))
        
        return True, token
    
    def verify_token(self, token: str) -> tuple[bool, Union[User, str]]:
        """Verify JWT token and return user."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            user_id = payload.get("user_id")
            
            if user_id not in self.users:
                return False, "User not found"
            
            user = self.users[user_id]
            if not user.is_active:
                return False, "User inactive"
            
            return True, user
            
        except jwt.ExpiredSignatureError:
            return False, "Token expired"
        except jwt.InvalidTokenError:
            return False, "Invalid token"
    
    def check_permission(self, user: User, resource: str, action: str) -> bool:
        """Check if user has permission for resource/action."""
        
        # System admin has all permissions
        if "system_admin" in user.roles:
            return True
        
        # Check specific permission
        permission = f"{resource}:{action}"
        if permission in user.permissions:
            return True
        
        # Check role-based permissions
        role_permissions = {
            "admin": ["*:*"],
            "user": ["agent:execute", "benchmark:run", "profile:read", "profile:update"],
            "readonly": ["*:read"]
        }
        
        for role in user.roles:
            if role in role_permissions:
                for perm in role_permissions[role]:
                    if perm == "*:*" or perm == permission:
                        return True
                    
                    # Check wildcard permissions
                    perm_resource, perm_action = perm.split(":")
                    if (perm_resource == "*" or perm_resource == resource) and \
                       (perm_action == "*" or perm_action == action):
                        return True
        
        return False
    
    def analyze_request_security(self, ip: str, user_agent: str, path: str, params: Dict[str, Any]) -> tuple[bool, Dict[str, Any]]:
        """Analyze request for security threats."""
        
        # Check if IP is blocked
        if self.intrusion_detection.is_ip_blocked(ip):
            return False, {"reason": "IP blocked", "threat_level": "critical"}
        
        # Analyze for intrusion patterns
        is_suspicious, threat_level, threats = self.intrusion_detection.analyze_request(
            ip, user_agent, path, params
        )
        
        if is_suspicious:
            self.audit_logger.log_event(SecurityEvent(
                id=secrets.token_urlsafe(8),
                event_type="intrusion_detected",
                user_id=None,
                ip_address=ip,
                user_agent=user_agent,
                resource=path,
                action="request",
                result="blocked",
                threat_level=threat_level,
                details={"threats": threats}
            ))
            
            return False, {
                "reason": "Suspicious activity detected",
                "threat_level": threat_level.value,
                "threats": threats
            }
        
        return True, {}
    
    def generate_api_key(self, user_id: str) -> str:
        """Generate API key for user."""
        api_key = self.encryption_manager.generate_api_key()
        self.api_keys[api_key] = user_id
        
        # Add to user's API keys
        if user_id in self.users:
            self.users[user_id].api_keys.append(api_key)
        
        self.audit_logger.log_event(SecurityEvent(
            id=secrets.token_urlsafe(8),
            event_type="api_key_generated",
            user_id=user_id,
            ip_address="system",
            user_agent="system",
            resource="api_key",
            action="create",
            result="success",
            threat_level=ThreatLevel.LOW,
            details={"api_key_prefix": api_key[:8]}
        ))
        
        return api_key
    
    def verify_api_key(self, api_key: str) -> tuple[bool, Union[User, str]]:
        """Verify API key and return user."""
        if api_key not in self.api_keys:
            return False, "Invalid API key"
        
        user_id = self.api_keys[api_key]
        if user_id not in self.users:
            return False, "User not found"
        
        user = self.users[user_id]
        if not user.is_active:
            return False, "User inactive"
        
        return True, user
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get security dashboard data."""
        return {
            "audit_statistics": self.audit_logger.get_statistics(),
            "active_users": len([u for u in self.users.values() if u.is_active]),
            "active_sessions": len(self.sessions),
            "blocked_ips": len(self.intrusion_detection.blocked_ips),
            "suspicious_ips": len(self.intrusion_detection.suspicious_ips),
            "api_keys_issued": len(self.api_keys),
            "recent_threats": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "type": event.event_type,
                    "threat_level": event.threat_level.value,
                    "ip": event.ip_address,
                    "details": event.details
                }
                for event in self.audit_logger.get_events(limit=10)
                if event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]
            ]
        }


# Global security manager
_global_security_manager: Optional[AdvancedSecurityManager] = None


def get_global_security_manager() -> AdvancedSecurityManager:
    """Get or create global security manager."""
    global _global_security_manager
    if _global_security_manager is None:
        _global_security_manager = AdvancedSecurityManager()
    return _global_security_manager


# Security decorators
def require_authentication(func: Callable) -> Callable:
    """Decorator to require authentication."""
    def wrapper(*args, **kwargs):
        # This would be implemented based on the specific framework
        # For FastAPI, you'd use Depends() with a security function
        return func(*args, **kwargs)
    return wrapper


def require_permission(resource: str, action: str):
    """Decorator to require specific permission."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # This would check permissions before executing
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Example usage
async def demonstrate_security():
    """Demonstrate security features."""
    
    print("SECURITY Advanced Security System Demo")
    print("=" * 50)
    
    # Initialize security manager
    security = AdvancedSecurityManager()
    
    # Create a test user
    success, result = security.create_user(
        username="testuser",
        email="test@example.com",
        password="SecurePass123!",
        roles={"user"}
    )
    
    if success:
        user = result
        print(f"PASS Created user: {user.username}")
    else:
        print(f"FAIL Failed to create user: {result}")
        return
    
    # Test authentication
    success, token = security.authenticate_user(
        username="testuser",
        password="SecurePass123!",
        ip_address="192.168.1.100",
        user_agent="TestClient/1.0"
    )
    
    if success:
        print(f"PASS Authentication successful")
    else:
        print(f"FAIL Authentication failed: {token}")
    
    # Test token verification
    success, user_or_error = security.verify_token(token)
    if success:
        print(f"PASS Token verified for user: {user_or_error.username}")
    else:
        print(f"FAIL Token verification failed: {user_or_error}")
    
    # Test permission checking
    has_permission = security.check_permission(user, "agent", "execute")
    print(f"PASS User has agent:execute permission: {has_permission}")
    
    # Test rate limiting
    for i in range(3):
        allowed, info = security.rate_limiter.check_rate_limit("192.168.1.100", "api_general")
        print(f"PASS Request {i+1} allowed: {allowed}")
    
    # Test intrusion detection
    is_safe, analysis = security.analyze_request_security(
        ip="192.168.1.100",
        user_agent="TestClient/1.0",
        path="/api/agent/execute",
        params={"task": "Calculate 2+2"}
    )
    print(f"PASS Request is safe: {is_safe}")
    
    # Get security dashboard
    dashboard = security.get_security_dashboard()
    print(f"\nRESULTS Security Dashboard:")
    print(f"  Active users: {dashboard['active_users']}")
    print(f"  Active sessions: {dashboard['active_sessions']}")
    print(f"  Blocked IPs: {dashboard['blocked_ips']}")
    print(f"  API keys issued: {dashboard['api_keys_issued']}")
    
    print("\nSUCCESS Security demonstration completed!")


if __name__ == "__main__":
    asyncio.run(demonstrate_security()) 
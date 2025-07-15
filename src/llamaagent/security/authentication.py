"""
Authentication service for LlamaAgent security module.
"""

import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

try:
    import jwt
except ImportError:
    # Mock jwt for testing
    class jwt:
        @staticmethod
        def encode(
            payload: Dict[str, Any], secret: str, algorithm: str = "HS256"
        ) -> str:
            return f"mock_token_{payload.get('user_id', 'unknown')}"

        @staticmethod
        def decode(token: str, secret: str, algorithms: list) -> Dict[str, Any]:
            if token.startswith("mock_token_"):
                return {"user_id": token.replace("mock_token_", "")}
            raise Exception("Invalid token")

        class InvalidTokenError(Exception):
            pass


class AuthenticationService:
    """Handles authentication and JWT tokens."""

    def __init__(self, secret: Optional[str] = None) -> None:
        self.secret = secret or secrets.token_urlsafe(32)

    def generate_token(self, claims: Dict[str, Any]) -> str:
        """Generate JWT token."""
        payload = {
            **claims,
            "exp": datetime.now(timezone.utc) + timedelta(hours=24),
            "iat": datetime.now(timezone.utc),
        }
        return jwt.encode(payload, self.secret, algorithm="HS256")

    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token."""
        try:
            return jwt.decode(token, self.secret, algorithms=["HS256"])
        except (jwt.InvalidTokenError, Exception):
            return {}

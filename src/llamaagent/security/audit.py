"""
Audit logging service for LlamaAgent security module.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class AuditLogger:
    """Logs security-related events."""

    def __init__(self) -> None:
        self.logs: List[Dict[str, Any]] = []

    async def log_event(
        self,
        user_id: str,
        action: str,
        resource: str,
        result: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log security event."""
        self.logs.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_id": user_id,
                "action": action,
                "resource": resource,
                "result": result,
                "metadata": metadata or {},
            }
        )

    async def get_logs(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get audit logs."""
        logs = self.logs

        if user_id:
            logs = [l for l in logs if l["user_id"] == user_id]

        if action:
            logs = [l for l in logs if l["action"] == action]

        return logs[-limit:]

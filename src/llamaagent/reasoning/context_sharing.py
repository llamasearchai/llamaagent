"""
Context Sharing Protocol - Enterprise Implementation

This module implements advanced context sharing capabilities for multi-agent reasoning, including:
- Shared context management
- Context synchronization
- Context versioning
- Context conflict resolution
- Distributed context updates

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

# Optional imports with fallbacks
try:
    import redis

    redis_available = True
except ImportError:
    redis = None
    redis_available = False

try:
    from opentelemetry import trace

    tracer = trace.get_tracer(__name__)
    tracing_available = True
except ImportError:
    tracer = None
    tracing_available = False

logger = logging.getLogger(__name__)


class ContextScope(Enum):
    """Scope levels for context sharing."""

    GLOBAL = "global"
    AGENT = "agent"
    SESSION = "session"
    TASK = "task"
    WORKFLOW = "workflow"


class ConflictResolution(Enum):
    """Conflict resolution strategies."""

    LATEST_WINS = "latest_wins"
    MERGE = "merge"
    MANUAL = "manual"
    PRIORITY = "priority"


@dataclass
class ContextUpdate:
    """Record of a context update."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    context_id: str = ""
    agent_id: str = ""
    updates: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    update_type: str = "update"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SharedContext:
    """Shared context container."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    scope: ContextScope = ContextScope.GLOBAL
    data: Dict[str, Any] = field(default_factory=dict)
    version: int = 1
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    contributors: Set[str] = field(default_factory=set)
    access_permissions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_checksum(self) -> str:
        """Calculate checksum for data integrity."""
        data_str = json.dumps(self.data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "scope": self.scope.value,
            "data": self.data,
            "version": self.version,
            "last_updated": self.last_updated.isoformat(),
            "contributors": list(self.contributors),
            "access_permissions": self.access_permissions,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SharedContext':
        """Create instance from dictionary."""
        return cls(
            id=data["id"],
            scope=ContextScope(data["scope"]),
            data=data["data"],
            version=data["version"],
            last_updated=datetime.fromisoformat(data["last_updated"]),
            contributors=set(data["contributors"]),
            access_permissions=data["access_permissions"],
            metadata=data["metadata"],
        )


class ContextSharingProtocol:
    """Advanced context sharing system for multi-agent environments."""

    def __init__(
        self,
        redis_client: Any = None,
        conflict_resolution: ConflictResolution = ConflictResolution.MERGE,
        sync_interval: int = 30,  # seconds
    ):
        # Redis setup (optional)
        if redis_available and redis_client is not None:
            self.redis_client = redis_client
        elif redis_available:
            try:
                self.redis_client = redis.Redis(decode_responses=True)
            except Exception:
                self.redis_client = None
        else:
            self.redis_client = None

        self.conflict_resolution = conflict_resolution
        self.sync_interval = sync_interval

        # Local context cache
        self.local_contexts: Dict[str, SharedContext] = {}
        self.update_subscribers: Dict[str, List[Callable]] = {}
        self.conflict_resolvers: Dict[str, Callable] = {}

        self.logger = logger
        self._sync_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start context synchronization."""
        self._running = True
        self._sync_task = asyncio.create_task(self._sync_loop())
        self.logger.info("Context sharing protocol started")

    async def stop(self) -> None:
        """Stop context synchronization."""
        self._running = False
        if self._sync_task:
            self._sync_task.cancel()
        self.logger.info("Context sharing protocol stopped")

    async def create_context(
        self,
        context_id: str,
        scope: ContextScope,
        initial_data: Optional[Dict[str, Any]] = None,
        permissions: Optional[Dict[str, Any]] = None,
    ) -> SharedContext:
        """Create new shared context."""
        context = SharedContext(
            id=context_id,
            scope=scope,
            data=initial_data or {},
            access_permissions=permissions or {},
        )

        # Store locally and in Redis
        self.local_contexts[context_id] = context
        await self._persist_context(context)

        self.logger.info(f"Created shared context: {context_id} (scope: {scope.value})")
        return context

    async def get_context(
        self, context_id: str, agent_id: str = ""
    ) -> Optional[SharedContext]:
        """Retrieve shared context."""
        # Check local cache first
        if context_id in self.local_contexts:
            context = self.local_contexts[context_id]
            if self._check_permissions(context, agent_id, "read"):
                return context

        # Load from Redis
        context = await self._load_context(context_id)
        if context and self._check_permissions(context, agent_id, "read"):
            self.local_contexts[context_id] = context
            return context

        return None

    async def update_context(
        self,
        context_id: str,
        agent_id: str,
        updates: Dict[str, Any],
        merge: bool = True,
        conflict_resolution: Optional[ConflictResolution] = None,
    ) -> bool:
        """Update shared context."""
        if tracing_available and tracer:
            with tracer.start_as_current_span("update_context") as span:
                span.set_attribute("context.id", context_id)
                return await self._update_context_impl(
                    context_id, agent_id, updates, merge, conflict_resolution
                )
        else:
            return await self._update_context_impl(
                context_id, agent_id, updates, merge, conflict_resolution
            )

    async def _update_context_impl(
        self,
        context_id: str,
        agent_id: str,
        updates: Dict[str, Any],
        merge: bool,
        conflict_resolution: Optional[ConflictResolution],
    ) -> bool:
        """Internal implementation of context update."""
        context = await self.get_context(context_id, agent_id)
        if not context:
            return False

        # Check permissions
        if not self._check_permissions(context, agent_id, "write"):
            self.logger.warning(
                f"Agent {agent_id} lacks write permission for context {context_id}"
            )
            return False

        # Create update record
        update = ContextUpdate(
            context_id=context_id,
            agent_id=agent_id,
            updates=updates,
            update_type="merge" if merge else "replace",
        )

        # Apply update
        if merge:
            context.data.update(updates)
        else:
            context.data = updates

        # Update metadata
        context.version += 1
        context.last_updated = datetime.now(timezone.utc)
        context.contributors.add(agent_id)

        # Persist changes
        await self._persist_context(context)
        await self._record_update(update)

        # Notify subscribers
        await self._notify_subscribers(context_id)

        return True

    async def merge_contexts(
        self, target_context_id: str, source_context_id: str, agent_id: str = ""
    ) -> bool:
        """Merge two contexts."""
        target = await self.get_context(target_context_id, agent_id)
        source = await self.get_context(source_context_id, agent_id)

        if not target or not source:
            return False

        # Check permissions
        if not self._check_permissions(target, agent_id, "write"):
            return False

        # Resolve conflicts
        resolution = self.conflict_resolution
        merged_data = await self._resolve_conflicts(
            target.data, source.data, resolution
        )

        # Update target context
        return await self.update_context(
            target_context_id, agent_id, merged_data, merge=False
        )

    async def get_context_history(
        self, context_id: str, limit: int = 100
    ) -> List[ContextUpdate]:
        """Get context update history."""
        if not self.redis_client:
            return []

        try:
            key = f"context_history:{context_id}"
            history_data = self.redis_client.lrange(key, 0, limit - 1)

            updates = []
            for data in history_data:
                try:
                    update_dict = json.loads(data)
                    update = ContextUpdate(
                        id=update_dict["id"],
                        context_id=update_dict["context_id"],
                        agent_id=update_dict["agent_id"],
                        updates=update_dict["updates"],
                        timestamp=datetime.fromisoformat(update_dict["timestamp"]),
                        update_type=update_dict.get("update_type", "update"),
                        metadata=update_dict.get("metadata", {}),
                    )
                    updates.append(update)
                except Exception as e:
                    self.logger.error(f"Error parsing update: {e}")

            return updates
        except Exception as e:
            self.logger.error(f"Error retrieving context history: {e}")
            return []

    async def subscribe_to_updates(
        self, context_id: str, callback: Callable[[str, Dict[str, Any]], None]
    ):
        """Subscribe to context updates."""
        if context_id not in self.update_subscribers:
            self.update_subscribers[context_id] = []
        self.update_subscribers[context_id].append(callback)

    async def unsubscribe_from_updates(
        self, context_id: str, callback: Callable[[str, Dict[str, Any]], None]
    ):
        """Unsubscribe from context updates."""
        if context_id in self.update_subscribers:
            try:
                self.update_subscribers[context_id].remove(callback)
            except ValueError:
                pass

    async def get_sharing_stats(self) -> Dict[str, Any]:
        """Get context sharing statistics."""
        stats = {
            "local_contexts": len(self.local_contexts),
            "active_subscriptions": sum(
                len(subs) for subs in self.update_subscribers.values()
            ),
            "redis_available": self.redis_client is not None,
            "sync_interval": self.sync_interval,
            "conflict_resolution": self.conflict_resolution.value,
        }

        # Add Redis stats if available
        if self.redis_client:
            try:
                pattern = "context:*"
                context_keys = self.redis_client.keys(pattern)
                stats["redis_contexts"] = len(context_keys)
            except Exception:
                stats["redis_contexts"] = 0

        return stats

    # Private methods

    async def _sync_loop(self) -> None:
        """Main synchronization loop."""
        while self._running:
            try:
                await self._sync_contexts()
                await asyncio.sleep(self.sync_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Sync loop error: {e}")
                await asyncio.sleep(5)  # Short delay on error

    async def _sync_contexts(self) -> None:
        """Synchronize local contexts with Redis."""
        if not self.redis_client:
            return

        # Sync each local context
        for context_id, local_context in self.local_contexts.items():
            try:
                remote_context = await self._load_context(context_id)
                if remote_context and remote_context.version > local_context.version:
                    # Remote is newer, merge changes
                    merged_data = await self._resolve_conflicts(
                        local_context.data,
                        remote_context.data,
                        self.conflict_resolution,
                    )
                    local_context.data = merged_data
                    local_context.version = remote_context.version
                    local_context.last_updated = remote_context.last_updated
            except Exception as e:
                self.logger.error(f"Error syncing context {context_id}: {e}")

    async def _persist_context(self, context: SharedContext) -> None:
        """Persist context to Redis."""
        if not self.redis_client:
            return

        try:
            key = f"context:{context.id}"
            data = json.dumps(context.to_dict())

            # Set TTL based on scope
            ttl_map = {
                ContextScope.GLOBAL: 86400 * 30,  # 30 days
                ContextScope.AGENT: 3600 * 12,  # 12 hours
                ContextScope.SESSION: 3600 * 4,  # 4 hours
                ContextScope.TASK: 86400 * 1,  # 1 day
                ContextScope.WORKFLOW: 86400 * 7,  # 7 days
            }

            ttl = ttl_map.get(context.scope, 86400)
            self.redis_client.setex(key, ttl, data)

        except Exception as e:
            self.logger.error(f"Error persisting context: {e}")

    async def _load_context(self, context_id: str) -> Optional[SharedContext]:
        """Load context from Redis."""
        if not self.redis_client:
            return None

        try:
            key = f"context:{context_id}"
            data = self.redis_client.get(key)
            if data:
                context_dict = json.loads(data)
                return SharedContext.from_dict(context_dict)
        except Exception as e:
            self.logger.error(f"Error loading context: {e}")

        return None

    async def _record_update(self, update: ContextUpdate) -> None:
        """Record context update in history."""
        if not self.redis_client:
            return

        try:
            key = f"context_history:{update.context_id}"
            update_data = {
                "id": update.id,
                "context_id": update.context_id,
                "agent_id": update.agent_id,
                "updates": update.updates,
                "timestamp": update.timestamp.isoformat(),
                "update_type": update.update_type,
                "metadata": update.metadata,
            }

            # Add to history list (keep last 1000 updates)
            self.redis_client.lpush(key, json.dumps(update_data))
            self.redis_client.ltrim(key, 0, 999)

            # Set expiry for history
            self.redis_client.expire(key, 86400 * 7)  # Keep for 7 days

        except Exception as e:
            self.logger.error(f"Error recording update: {e}")

    async def _notify_subscribers(self, context_id: str) -> None:
        """Notify subscribers of context changes."""
        if context_id in self.update_subscribers:
            context = self.local_contexts.get(context_id)
            if context:
                for callback in self.update_subscribers[context_id]:
                    try:
                        await callback(context_id, context.data)
                    except Exception as e:
                        self.logger.error(f"Error notifying subscriber: {e}")

    def _check_permissions(
        self, context: SharedContext, agent_id: str, operation: str
    ) -> bool:
        """Check if agent has permission for operation."""
        if not context.access_permissions:
            return True  # No restrictions

        agent_perms = context.access_permissions.get(agent_id, {})
        if operation in agent_perms:
            return agent_perms[operation]

        # Check default permissions
        default_perms = context.access_permissions.get("default", {})
        return default_perms.get(operation, True)

    async def _resolve_conflicts(
        self,
        local_data: Dict[str, Any],
        remote_data: Dict[str, Any],
        resolution: ConflictResolution,
    ) -> Dict[str, Any]:
        """Resolve conflicts between local and remote data."""
        if resolution == ConflictResolution.LATEST_WINS:
            return remote_data

        elif resolution == ConflictResolution.MERGE:
            # Deep merge dictionaries
            return self._deep_merge(local_data, remote_data)

        elif resolution == ConflictResolution.PRIORITY:
            # Merge based on data priority
            priority_keys = ["priority", "importance", "weight"]
            result = local_data.copy()

            for key, value in remote_data.items():
                if key in priority_keys:
                    result[key] = value
                elif key not in result:
                    result[key] = value

            return result

        else:  # MANUAL or unknown
            # Return local data unchanged for manual resolution
            return local_data

    def _deep_merge(
        self, dict1: Dict[str, Any], dict2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = dict1.copy()

        for key, value in dict2.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

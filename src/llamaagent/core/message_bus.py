"""
Message Bus - High-performance message routing and communication system

Provides reliable message delivery, routing, filtering, and transformation
for agent-to-agent communication with Redis support for distributed systems.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import json
import logging
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

try:
    from opentelemetry import trace

    tracer = trace.get_tracer(__name__)
    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False
    tracer = None


class MessagePriority(Enum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


class MessageType(Enum):
    TASK_ASSIGNMENT = "task_assignment"
    TASK_RESULT = "task_result"
    CONTEXT_UPDATE = "context_update"
    COLLABORATION_REQUEST = "collaboration_request"
    COLLABORATION_RESPONSE = "collaboration_response"
    HEALTH_CHECK = "health_check"
    HEALTH_RESPONSE = "health_response"
    BROADCAST = "broadcast"
    SYSTEM_NOTIFICATION = "system_notification"


@dataclass
class Message:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    recipient_id: Optional[str] = None  # None for broadcast
    message_type: MessageType = MessageType.BROADCAST
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    priority: MessagePriority = MessagePriority.NORMAL
    requires_response: bool = False
    correlation_id: Optional[str] = None
    ttl: Optional[int] = None  # Time to live in seconds
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        if not self.ttl:
            return False
        age = (datetime.now(timezone.utc) - self.timestamp).total_seconds()
        return age > self.ttl

    def should_retry(self) -> bool:
        return self.retry_count < self.max_retries

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "message_type": self.message_type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.value,
            "requires_response": self.requires_response,
            "correlation_id": self.correlation_id,
            "ttl": self.ttl,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            sender_id=data.get("sender_id", ""),
            recipient_id=data.get("recipient_id"),
            message_type=MessageType(
                data.get("message_type", MessageType.BROADCAST.value)
            ),
            content=data.get("content", {}),
            timestamp=datetime.fromisoformat(
                data.get("timestamp", datetime.now(timezone.utc).isoformat())
            ),
            priority=MessagePriority(
                data.get("priority", MessagePriority.NORMAL.value)
            ),
            requires_response=data.get("requires_response", False),
            correlation_id=data.get("correlation_id"),
            ttl=data.get("ttl"),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            metadata=data.get("metadata", {}),
        )


class MessageFilter:
    def __init__(self, filter_func: Callable[[Message], bool]):
        self.filter_func = filter_func

    def matches(self, message: Message) -> bool:
        try:
            return self.filter_func(message)
        except Exception:
            return False


class MessageTransformer:
    def __init__(self, transform_func: Callable[[Message], Message]):
        self.transform_func = transform_func

    def transform(self, message: Message) -> Message:
        try:
            return self.transform_func(message)
        except Exception:
            return message


class MessageBus:
    def __init__(
        self,
        redis_client: Optional[Any] = None,
        max_queue_size: int = 10000,
        enable_persistence: bool = True,
        enable_metrics: bool = True,
        enable_tracing: bool = True,
    ):
        self.redis_client = redis_client
        self.max_queue_size = max_queue_size
        self.enable_persistence = enable_persistence
        self.enable_metrics = enable_metrics
        self.enable_tracing = enable_tracing and TRACING_AVAILABLE

        # Message queues by priority
        self.message_queues: Dict[MessagePriority, deque] = {
            priority: deque(maxlen=max_queue_size) for priority in MessagePriority
        }

        # Subscribers
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.pattern_subscribers: Dict[str, List[Callable]] = defaultdict(list)

        # Message filters and transformers
        self.filters: List[MessageFilter] = []
        self.transformers: List[MessageTransformer] = []

        # Routing tables
        self.routing_table: Dict[str, List[str]] = defaultdict(list)
        self.load_balancers: Dict[str, List[str]] = defaultdict(list)

        # Circuit breakers
        self.circuit_breakers: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # Metrics
        self.message_count = 0
        self.error_count = 0
        self.processed_count = 0

        # State
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None

        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("MessageBus")
        logger.setLevel(logging.INFO)
        return logger

    async def start(self) -> None:
        self._running = True
        self._processor_task = asyncio.create_task(self._process_messages())
        self.logger.info("Message bus started")

    async def stop(self) -> None:
        self._running = False
        if self._processor_task:
            self._processor_task.cancel()
        self.logger.info("Message bus stopped")

    async def publish(self, message: Message, load_balance: bool = False) -> None:
        if self.enable_tracing and tracer:
            with tracer.start_as_current_span("message_bus_publish"):
                await self._publish_internal(message, load_balance)
        else:
            await self._publish_internal(message, load_balance)

    async def _publish_internal(self, message: Message, load_balance: bool) -> None:
        self.message_count += 1

        # Apply filters
        for filter_obj in self.filters:
            if not filter_obj.matches(message):
                return

        # Apply transformers
        for transformer in self.transformers:
            message = transformer.transform(message)

        # Check expiration
        if message.is_expired():
            self.logger.warning(f"Message {message.id} expired, discarding")
            return

        # Add to queue
        self.message_queues[message.priority].appendleft(message)

        # Persist if enabled
        if self.enable_persistence and self.redis_client:
            await self._persist_message(message)

        # Route message
        await self._route_message(message, load_balance)

    async def _route_message(self, message: Message, load_balance: bool) -> None:
        target_agents = []

        # Direct recipient
        if message.recipient_id:
            target_agents.append(message.recipient_id)
        else:
            # Broadcast to all subscribers
            target_agents.extend(self.subscribers.keys())

        # Load balancing
        if load_balance and message.recipient_id:
            balanced_target = self._get_load_balanced_target(message.recipient_id)
            if balanced_target:
                target_agents = [balanced_target]

        # Deliver to targets
        for agent_id in target_agents:
            await self._deliver_to_agent(message, agent_id)

    async def _deliver_to_agent(self, message: Message, agent_id: str) -> None:
        # Try specific message type subscription
        type_key = f"{agent_id}:{message.message_type.value}"
        if type_key in self.subscribers:
            for callback in self.subscribers[type_key]:
                await self._safe_callback(callback, message)

        # Try general subscription
        elif agent_id in self.subscribers:
            for callback in self.subscribers[agent_id]:
                await self._safe_callback(callback, message)

        # Check pattern subscriptions
        for pattern, callbacks in self.pattern_subscribers.items():
            if self._matches_pattern(message, pattern):
                for callback in callbacks:
                    await self._safe_callback(callback, message)

    async def _safe_callback(self, callback: Callable, message: Message) -> None:
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(message)
            else:
                callback(message)
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Error in callback: {e}")

    def _matches_pattern(self, message: Message, pattern: str) -> bool:
        # Simple pattern matching - can be extended
        return (
            pattern in message.content.get("text", "")
            or pattern in message.message_type.value
        )

    def _get_load_balanced_target(self, lb_id: str) -> Optional[str]:
        if lb_id not in self.load_balancers:
            return None

        targets = self.load_balancers[lb_id]
        if not targets:
            return None

        # Simple round-robin
        target = targets.pop(0)
        targets.append(target)

        return target

    async def _persist_message(self, message: Message) -> None:
        if not self.redis_client:
            return

        try:
            key = f"message:{message.id}"
            data = json.dumps(message.to_dict())
            self.redis_client.set(key, data)

            # Set TTL
            if message.ttl:
                self.redis_client.expire(key, message.ttl)

        except Exception as e:
            self.logger.error(f"Failed to persist message: {e}")

    async def _process_messages(self) -> None:
        while self._running:
            try:
                # Process by priority
                for priority in MessagePriority:
                    queue = self.message_queues[priority]
                    if queue:
                        message = queue.pop()
                        await self._process_single_message(message)
                        self.processed_count += 1

                # Small delay to prevent busy waiting
                await asyncio.sleep(0.001)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error processing messages: {e}")
                await asyncio.sleep(0.1)

    async def _process_single_message(self, message: Message) -> None:
        try:
            # Check if message should be retried
            if message.should_retry():
                # Add back to queue for retry
                message.retry_count += 1
                self.message_queues[message.priority].appendleft(message)
            else:
                # Message processed successfully or max retries reached
                pass

        except Exception as e:
            self.logger.error(f"Error processing message {message.id}: {e}")

    async def subscribe(
        self,
        agent_id: str,
        callback: Callable,
        message_types: Optional[List[MessageType]] = None,
    ) -> None:
        if message_types:
            for msg_type in message_types:
                key = f"{agent_id}:{msg_type.value}"
                self.subscribers[key].append(callback)
        else:
            # Subscribe to all messages for this agent
            self.subscribers[agent_id].append(callback)

        self.logger.info(f"Agent {agent_id} subscribed to messages")

    async def unsubscribe(
        self,
        agent_id: str,
        callback: Callable,
        message_types: Optional[List[MessageType]] = None,
    ) -> None:
        if message_types:
            for msg_type in message_types:
                key = f"{agent_id}:{msg_type.value}"
                if callback in self.subscribers[key]:
                    self.subscribers[key].remove(callback)
        else:
            if callback in self.subscribers[agent_id]:
                self.subscribers[agent_id].remove(callback)

    async def subscribe_pattern(self, pattern: str, callback: Callable) -> None:
        self.pattern_subscribers[pattern].append(callback)

    def add_filter(self, filter_obj: MessageFilter) -> None:
        self.filters.append(filter_obj)

    def add_transformer(self, transformer: MessageTransformer) -> None:
        self.transformers.append(transformer)

    def add_routing_rule(
        self,
        from_agent: str,
        to_agent: str,
        condition: Optional[Callable[[Message], bool]] = None,
    ) -> None:
        rule = {"to_agent": to_agent, "condition": condition}
        self.routing_table[from_agent].append(to_agent)

    def add_load_balancer(self, lb_id: str, targets: List[str]) -> None:
        self.load_balancers[lb_id] = targets[:]

    async def get_message_stats(self) -> Dict[str, Any]:
        queue_sizes = {
            priority.name: len(queue) for priority, queue in self.message_queues.items()
        }

        return {
            "message_count": self.message_count,
            "error_count": self.error_count,
            "processed_count": self.processed_count,
            "queue_sizes": queue_sizes,
            "subscribers": len(self.subscribers),
            "pattern_subscribers": len(self.pattern_subscribers),
            "filters": len(self.filters),
            "transformers": len(self.transformers),
            "routing_rules": sum(
                len(targets) for targets in self.routing_table.values()
            ),
            "running": self._running,
        }


class MessageRouter:
    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus
        self.routing_policies: Dict[str, Callable] = {}

    def add_policy(self, name: str, policy: Callable[[Message], List[str]]) -> None:
        self.routing_policies[name] = policy

    async def route_with_policy(self, message: Message, policy_name: str) -> None:
        if policy_name in self.routing_policies:
            targets = self.routing_policies[policy_name](message)
            for target in targets:
                message.recipient_id = target
                await self.message_bus.publish(message)


# Default message bus instance
default_message_bus = MessageBus()


def get_message_bus() -> MessageBus:
    return default_message_bus

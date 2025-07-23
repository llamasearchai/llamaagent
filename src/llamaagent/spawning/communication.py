"""Inter-agent communication protocols and channels."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """Types of messages that can be sent between agents."""

    # Task-related
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    TASK_STATUS = "task_status"
    TASK_DELEGATION = "task_delegation"

    # Coordination
    COORDINATION = "coordination"
    SYNCHRONIZATION = "synchronization"
    NEGOTIATION = "negotiation"

    # Information sharing
    KNOWLEDGE_SHARE = "knowledge_share"
    CONTEXT_UPDATE = "context_update"
    CAPABILITY_ANNOUNCEMENT = "capability_announcement"

    # Control messages
    PAUSE = "pause"
    RESUME = "resume"
    TERMINATE = "terminate"
    HEALTH_CHECK = "health_check"

    # Results and feedback
    RESULT_SHARE = "result_share"
    FEEDBACK = "feedback"
    ERROR_REPORT = "error_report"


class MessagePriority(int, Enum):
    """Message priority levels."""

    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


@dataclass
class Message:
    """Message structure for inter-agent communication."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.TASK_REQUEST
    sender: str = ""
    recipient: str = ""  # Empty string means broadcast
    content: Any = None
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    expiry: Optional[float] = None  # Message expiry time
    requires_response: bool = False
    correlation_id: Optional[str] = None  # For request-response correlation
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if the message has expired."""
        if self.expiry is None:
            return False
        return time.time() > self.expiry

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "sender": self.sender,
            "recipient": self.recipient,
            "content": self.content,
            "priority": self.priority.value,
            "timestamp": self.timestamp,
            "expiry": self.expiry,
            "requires_response": self.requires_response,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Message:
        """Create message from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=MessageType(data.get("type", MessageType.TASK_REQUEST.value)),
            sender=data.get("sender", ""),
            recipient=data.get("recipient", ""),
            content=data.get("content"),
            priority=MessagePriority(
                data.get("priority", MessagePriority.NORMAL.value)
            ),
            timestamp=data.get("timestamp", time.time()),
            expiry=data.get("expiry"),
            requires_response=data.get("requires_response", False),
            correlation_id=data.get("correlation_id"),
            metadata=data.get("metadata", {}),
        )


class CommunicationProtocol(ABC):
    """Abstract base class for communication protocols."""

    @abstractmethod
    async def send(self, message: Message) -> bool:
        """Send a message."""

    @abstractmethod
    async def receive(self, timeout: Optional[float] = None) -> Optional[Message]:
        """Receive a message."""

    @abstractmethod
    async def subscribe(self, message_types: Set[MessageType]) -> None:
        """Subscribe to specific message types."""

    @abstractmethod
    async def unsubscribe(self, message_types: Set[MessageType]) -> None:
        """Unsubscribe from message types."""


class AgentChannel(CommunicationProtocol):
    """Communication channel for an individual agent."""

    def __init__(
        self,
        agent_id: str,
        bus: Optional[MessageBus] = None,
        max_queue_size: int = 1000,
    ) -> None:
        self.agent_id = agent_id
        self.bus = bus
        self.max_queue_size = max_queue_size

        # Message queue for this agent
        self.inbox: asyncio.Queue[Message] = asyncio.Queue(maxsize=max_queue_size)
        self.subscriptions: Set[MessageType] = set()

        # Response tracking
        self.pending_responses: Dict[str, asyncio.Future] = {}

        # Statistics
        self.messages_sent = 0
        self.messages_received = 0
        self.messages_dropped = 0

        # Register with message bus if provided
        if self.bus:
            self.bus.register_channel(self)

    async def send(self, message: Message) -> bool:
        """Send a message through the channel."""
        try:
            # Set sender if not already set
            if not message.sender:
                message.sender = self.agent_id

            self.messages_sent += 1

            # If connected to a bus, route through it
            if self.bus:
                return await self.bus.route_message(message)

            # Otherwise, direct delivery if possible
            logger.warning(f"Channel {self.agent_id} not connected to message bus")
            return False

        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False

    async def receive(self, timeout: Optional[float] = None) -> Optional[Message]:
        """Receive a message from the inbox."""
        try:
            if timeout:
                message = await asyncio.wait_for(self.inbox.get(), timeout)
            else:
                message = await self.inbox.get()

            # Check if message has expired
            if message.is_expired():
                logger.debug(f"Dropping expired message {message.id}")
                self.messages_dropped += 1
                return await self.receive(timeout)  # Try next message

            self.messages_received += 1
            return message

        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            return None

    async def deliver(self, message: Message) -> bool:
        """Deliver a message to this channel's inbox."""
        try:
            # Check if we should receive this message
            if message.recipient and message.recipient != self.agent_id:
                return False

            # Check subscriptions for broadcast messages
            if not message.recipient and message.type not in self.subscriptions:
                return False

            # Try to add to inbox
            try:
                self.inbox.put_nowait(message)
                return True
            except asyncio.QueueFull:
                logger.warning(
                    f"Inbox full for agent {self.agent_id}, dropping message"
                )
                self.messages_dropped += 1
                return False

        except Exception as e:
            logger.error(f"Error delivering message: {e}")
            return False

    async def request(
        self,
        recipient: str,
        content: Any,
        message_type: MessageType = MessageType.TASK_REQUEST,
        timeout: float = 30.0,
    ) -> Optional[Message]:
        """Send a request and wait for a response."""
        # Create request message
        request = Message(
            type=message_type,
            sender=self.agent_id,
            recipient=recipient,
            content=content,
            requires_response=True,
        )

        # Create future for response
        response_future: asyncio.Future[Message] = asyncio.Future()
        self.pending_responses[request.id] = response_future

        try:
            # Send request
            if not await self.send(request):
                raise Exception("Failed to send request")

            # Wait for response
            response = await asyncio.wait_for(response_future, timeout)
            return response

        except asyncio.TimeoutError:
            logger.warning(f"Request {request.id} timed out")
            return None
        finally:
            # Clean up
            self.pending_responses.pop(request.id, None)

    async def respond(self, original_message: Message, content: Any) -> bool:
        """Send a response to a message."""
        if not original_message.requires_response:
            return True

        response = Message(
            type=MessageType.TASK_RESPONSE,
            sender=self.agent_id,
            recipient=original_message.sender,
            content=content,
            correlation_id=original_message.id,
        )

        return await self.send(response)

    async def subscribe(self, message_types: Set[MessageType]) -> None:
        """Subscribe to message types."""
        self.subscriptions.update(message_types)
        if self.bus:
            await self.bus.update_subscriptions(self.agent_id, self.subscriptions)

    async def unsubscribe(self, message_types: Set[MessageType]) -> None:
        """Unsubscribe from message types."""
        self.subscriptions.difference_update(message_types)
        if self.bus:
            await self.bus.update_subscriptions(self.agent_id, self.subscriptions)

    def handle_response(self, message: Message) -> None:
        """Handle a response message."""
        if message.correlation_id and message.correlation_id in self.pending_responses:
            future = self.pending_responses[message.correlation_id]
            if not future.done():
                future.set_result(message)

    def get_stats(self) -> Dict[str, Any]:
        """Get channel statistics."""
        return {
            "agent_id": self.agent_id,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "messages_dropped": self.messages_dropped,
            "inbox_size": self.inbox.qsize(),
            "pending_responses": len(self.pending_responses),
            "subscriptions": len(self.subscriptions),
        }


class BroadcastChannel(AgentChannel):
    """Channel for broadcasting messages to multiple agents."""

    def __init__(self, channel_id: str, bus: Optional[MessageBus] = None) -> None:
        super().__init__(channel_id, bus)
        self.subscribers: Set[str] = set()

    async def broadcast(
        self,
        content: Any,
        message_type: MessageType = MessageType.COORDINATION,
        exclude: Optional[Set[str]] = None,
    ) -> int:
        """Broadcast a message to all subscribers."""
        exclude = exclude or set()
        message = Message(
            type=message_type,
            sender=self.agent_id,
            recipient="",  # Broadcast
            content=content,
        )

        sent_count = 0
        if self.bus:
            # Let the bus handle broadcasting
            await self.send(message)
            sent_count = len(self.subscribers) - len(exclude)

        return sent_count


class DirectChannel(CommunicationProtocol):
    """Direct peer-to-peer channel between two agents."""

    def __init__(
        self,
        agent1_id: str,
        agent2_id: str,
        max_queue_size: int = 100,
    ) -> None:
        self.agent1_id = agent1_id
        self.agent2_id = agent2_id

        # Bidirectional queues
        self.queue_1_to_2: asyncio.Queue[Message] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self.queue_2_to_1: asyncio.Queue[Message] = asyncio.Queue(
            maxsize=max_queue_size
        )

        # Statistics
        self.messages_exchanged = 0

    async def send(self, message: Message) -> bool:
        """Send a message through the direct channel."""
        try:
            # Determine direction
            if message.sender == self.agent1_id:
                queue = self.queue_1_to_2
            elif message.sender == self.agent2_id:
                queue = self.queue_2_to_1
            else:
                return False

            # Send message
            await queue.put(message)
            self.messages_exchanged += 1
            return True

        except asyncio.QueueFull:
            logger.warning("Direct channel queue full")
            return False
        except Exception as e:
            logger.error(f"Error in direct channel: {e}")
            return False

    async def receive(self, timeout: Optional[float] = None) -> Optional[Message]:
        """Receive is not implemented for direct channels."""
        logger.warning(
            "Direct channels require agent_id. Use receive_for_agent instead."
        )
        return None

    async def receive_for_agent(
        self,
        agent_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[Message]:
        """Receive a message for a specific agent."""
        try:
            # Determine queue
            if agent_id == self.agent1_id:
                queue = self.queue_2_to_1
            elif agent_id == self.agent2_id:
                queue = self.queue_1_to_2
            else:
                return None

            # Receive message
            if timeout:
                message = await asyncio.wait_for(queue.get(), timeout)
            else:
                message = await queue.get()

            return message

        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Error receiving in direct channel: {e}")
            return None

    async def subscribe(self, message_types: Set[MessageType]) -> None:
        """Not applicable for direct channels."""

    async def unsubscribe(self, message_types: Set[MessageType]) -> None:
        """Not applicable for direct channels."""


class MessageBus:
    """Central message routing system for all agents."""

    def __init__(self) -> None:
        self.channels: Dict[str, AgentChannel] = {}
        self.subscriptions: Dict[MessageType, Set[str]] = {}
        self.direct_channels: Dict[tuple[str, str], DirectChannel] = {}

        # Message history
        self.message_history: List[Message] = []
        self.max_history_size = 10000

        # Statistics
        self.total_messages_routed = 0
        self.failed_deliveries = 0

        # Locks
        self._lock = asyncio.Lock()

    def register_channel(self, channel: AgentChannel) -> None:
        """Register a channel with the message bus."""
        self.channels[channel.agent_id] = channel

    def unregister_channel(self, agent_id: str) -> None:
        """Unregister a channel."""
        self.channels.pop(agent_id, None)

        # Remove from all subscriptions
        for subscribers in self.subscriptions.values():
            subscribers.discard(agent_id)

    async def route_message(self, message: Message) -> bool:
        """Route a message to its destination(s)."""
        async with self._lock:
            self.total_messages_routed += 1

            # Add to history
            self.message_history.append(message)
            if len(self.message_history) > self.max_history_size:
                self.message_history.pop(0)

            # Direct message
            if message.recipient:
                if message.recipient in self.channels:
                    channel = self.channels[message.recipient]
                    success = await channel.deliver(message)

                    # Handle response correlation
                    if message.type == MessageType.TASK_RESPONSE:
                        channel.handle_response(message)

                    if not success:
                        self.failed_deliveries += 1
                    return success
                else:
                    logger.warning(f"Unknown recipient: {message.recipient}")
                    self.failed_deliveries += 1
                    return False

            # Broadcast message
            else:
                subscribers = self.subscriptions.get(message.type, set())
                delivered = 0

                for agent_id in subscribers:
                    if agent_id != message.sender and agent_id in self.channels:
                        channel = self.channels[agent_id]
                        if await channel.deliver(message):
                            delivered += 1

                return delivered > 0

    async def update_subscriptions(
        self,
        agent_id: str,
        message_types: Set[MessageType],
    ) -> None:
        """Update subscriptions for an agent."""
        async with self._lock:
            # Remove old subscriptions
            for msg_type, subscribers in self.subscriptions.items():
                subscribers.discard(agent_id)

            # Add new subscriptions
            for msg_type in message_types:
                if msg_type not in self.subscriptions:
                    self.subscriptions[msg_type] = set()
                self.subscriptions[msg_type].add(agent_id)

    def create_direct_channel(self, agent1_id: str, agent2_id: str) -> DirectChannel:
        """Create a direct channel between two agents."""
        key = tuple(sorted([agent1_id, agent2_id]))

        if key not in self.direct_channels:
            self.direct_channels[key] = DirectChannel(agent1_id, agent2_id)

        return self.direct_channels[key]

    def get_stats(self) -> Dict[str, Any]:
        """Get message bus statistics."""
        return {
            "total_channels": len(self.channels),
            "total_messages_routed": self.total_messages_routed,
            "failed_deliveries": self.failed_deliveries,
            "message_history_size": len(self.message_history),
            "subscription_types": len(self.subscriptions),
            "direct_channels": len(self.direct_channels),
        }

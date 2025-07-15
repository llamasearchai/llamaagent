"""Agent spawning and sub-agent management for LlamaAgent framework.

This module provides autonomous agent spawning capabilities, including:
- Dynamic agent creation based on task requirements
- Agent pool management for resource optimization
- Inter-agent communication protocols
- Hierarchical agent structures
- Load balancing and lifecycle management
"""

from .agent_pool import AgentPool, PoolConfig, PoolStats
from .agent_spawner import (
    AgentHierarchy,
    AgentRelationship,
    AgentSpawner,
    SpawnConfig,
    SpawnResult,
)
from .communication import (
    AgentChannel,
    BroadcastChannel,
    CommunicationProtocol,
    DirectChannel,
    Message,
    MessageType,
)

__all__ = [
    # Core spawning classes
    "AgentSpawner",
    "AgentPool",
    "AgentHierarchy",
    # Configuration
    "SpawnConfig",
    "PoolConfig",
    # Results and monitoring
    "SpawnResult",
    "PoolStats",
    "AgentRelationship",
    # Communication
    "AgentChannel",
    "Message",
    "MessageType",
    "CommunicationProtocol",
    "BroadcastChannel",
    "DirectChannel",
]

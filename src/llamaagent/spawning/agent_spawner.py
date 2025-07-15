"""Agent spawning and hierarchy management implementation."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Type

from ..agents.base import AgentConfig, AgentRole, BaseAgent
from ..agents.react import ReactAgent
from ..llm import create_provider
from ..memory.base import SimpleMemory
from ..tools import ToolRegistry

logger = logging.getLogger(__name__)


class AgentRelationship(str, Enum):
    """Types of relationships between agents."""

    PARENT_CHILD = "parent_child"
    SIBLING = "sibling"
    SUPERVISOR_WORKER = "supervisor_worker"
    PEER = "peer"
    COORDINATOR_SUBORDINATE = "coordinator_subordinate"


@dataclass
class SpawnConfig:
    """Configuration for spawning new agents."""

    # Agent configuration
    agent_type: Type[BaseAgent] = ReactAgent
    agent_config: Optional[AgentConfig] = None

    # Resource limits
    max_memory_mb: float = 512.0
    max_execution_time: float = 300.0
    max_api_calls: int = 100

    # Hierarchy settings
    parent_id: Optional[str] = None
    relationship: AgentRelationship = AgentRelationship.PARENT_CHILD
    inherit_tools: bool = True
    inherit_memory: bool = False
    share_context: bool = True

    # Lifecycle settings
    auto_cleanup: bool = True
    persistent: bool = False
    priority: int = 1  # 1-10, higher is more important

    # Communication settings
    communication_timeout: float = 30.0
    message_queue_size: int = 100


@dataclass
class SpawnResult:
    """Result of agent spawning operation."""

    success: bool
    agent_id: str
    agent: Optional[BaseAgent] = None
    error: Optional[str] = None
    spawn_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentNode:
    """Node in the agent hierarchy tree."""

    agent_id: str
    agent: BaseAgent
    parent_id: Optional[str] = None
    children: Set[str] = field(default_factory=set)
    relationship: AgentRelationship = AgentRelationship.PARENT_CHILD
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Resource tracking
    memory_usage_mb: float = 0.0
    api_calls_made: int = 0
    total_execution_time: float = 0.0

    # State management
    is_active: bool = True
    is_paused: bool = False
    last_activity: float = field(default_factory=time.time)


class AgentHierarchy:
    """Manages hierarchical relationships between agents."""

    def __init__(self) -> None:
        self.nodes: Dict[str, AgentNode] = {}
        self.root_agents: Set[str] = set()
        self._lock = asyncio.Lock()

    async def add_agent(
        self,
        agent_id: str,
        agent: BaseAgent,
        parent_id: Optional[str] = None,
        relationship: AgentRelationship = AgentRelationship.PARENT_CHILD,
    ) -> None:
        """Add an agent to the hierarchy."""
        async with self._lock:
            node = AgentNode(
                agent_id=agent_id,
                agent=agent,
                parent_id=parent_id,
                relationship=relationship,
            )

            self.nodes[agent_id] = node

            if parent_id and parent_id in self.nodes:
                self.nodes[parent_id].children.add(agent_id)
            else:
                self.root_agents.add(agent_id)

    async def remove_agent(self, agent_id: str, cascade: bool = True) -> None:
        """Remove an agent from the hierarchy."""
        async with self._lock:
            if agent_id not in self.nodes:
                return

            node = self.nodes[agent_id]

            # Handle cascading removal
            if cascade and node.children:
                for child_id in list(node.children):
                    await self.remove_agent(child_id, cascade=True)

            # Remove from parent's children
            if node.parent_id and node.parent_id in self.nodes:
                self.nodes[node.parent_id].children.discard(agent_id)

            # Remove from root agents if applicable
            self.root_agents.discard(agent_id)

            # Clean up the agent
            if hasattr(node.agent, "cleanup"):
                await node.agent.cleanup()

            del self.nodes[agent_id]

    def get_ancestors(self, agent_id: str) -> List[str]:
        """Get all ancestors of an agent."""
        ancestors = []
        current_id = agent_id

        while current_id in self.nodes:
            node = self.nodes[current_id]
            if node.parent_id:
                ancestors.append(node.parent_id)
                current_id = node.parent_id
            else:
                break

        return ancestors

    def get_descendants(self, agent_id: str) -> List[str]:
        """Get all descendants of an agent."""
        descendants = []

        def _collect_descendants(node_id: str) -> None:
            if node_id in self.nodes:
                for child_id in self.nodes[node_id].children:
                    descendants.append(child_id)
                    _collect_descendants(child_id)

        _collect_descendants(agent_id)
        return descendants

    def get_siblings(self, agent_id: str) -> List[str]:
        """Get all siblings of an agent."""
        if agent_id not in self.nodes:
            return []

        node = self.nodes[agent_id]
        if not node.parent_id or node.parent_id not in self.nodes:
            return []

        parent = self.nodes[node.parent_id]
        return [child_id for child_id in parent.children if child_id != agent_id]

    def get_hierarchy_stats(self) -> Dict[str, Any]:
        """Get statistics about the hierarchy."""
        total_agents = len(self.nodes)
        active_agents = sum(1 for n in self.nodes.values() if n.is_active)

        # Calculate depth
        max_depth = 0
        for agent_id in self.root_agents:
            depth = self._calculate_depth(agent_id)
            max_depth = max(max_depth, depth)

        # Resource usage
        total_memory = sum(n.memory_usage_mb for n in self.nodes.values())
        total_api_calls = sum(n.api_calls_made for n in self.nodes.values())

        return {
            "total_agents": total_agents,
            "active_agents": active_agents,
            "root_agents": len(self.root_agents),
            "max_depth": max_depth,
            "total_memory_mb": total_memory,
            "total_api_calls": total_api_calls,
            "relationships": self._count_relationships(),
        }

    def _calculate_depth(self, agent_id: str, current_depth: int = 0) -> int:
        """Calculate the depth of a subtree."""
        if agent_id not in self.nodes:
            return current_depth

        node = self.nodes[agent_id]
        if not node.children:
            return current_depth

        max_child_depth = current_depth
        for child_id in node.children:
            child_depth = self._calculate_depth(child_id, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)

        return max_child_depth

    def _count_relationships(self) -> Dict[str, int]:
        """Count different types of relationships."""
        counts = {rel.value: 0 for rel in AgentRelationship}

        for node in self.nodes.values():
            counts[node.relationship.value] += 1

        return counts


class AgentSpawner:
    """Manages dynamic agent creation and lifecycle."""

    def __init__(
        self,
        hierarchy: Optional[AgentHierarchy] = None,
        default_llm_provider: Optional[Any] = None,
        tool_registry: Optional[ToolRegistry] = None,
    ) -> None:
        self.hierarchy = hierarchy or AgentHierarchy()
        self.default_llm_provider = default_llm_provider
        self.tool_registry = tool_registry or ToolRegistry()
        self._spawn_counter = 0
        self._active_spawns: Dict[str, SpawnConfig] = {}
        self._resource_monitor = ResourceMonitor()

    async def spawn_agent(
        self,
        task: str,
        config: Optional[SpawnConfig] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> SpawnResult:
        """Spawn a new agent for a specific task."""
        start_time = time.time()
        config = config or SpawnConfig()

        # Generate unique agent ID
        agent_id = f"agent_{uuid.uuid4().hex[:8]}_{self._spawn_counter}"
        self._spawn_counter += 1

        try:
            # Check resource availability
            if not await self._check_resources(config):
                return SpawnResult(
                    success=False,
                    agent_id=agent_id,
                    error="Insufficient resources available",
                )

            # Create agent configuration
            agent_config = await self._create_agent_config(task, config, context)

            # Initialize tools
            tools = await self._initialize_tools(config, agent_config)

            # Initialize memory
            memory = await self._initialize_memory(config, agent_config)

            # Create LLM provider
            llm_provider = self._get_llm_provider(config)

            # Create the agent
            agent_class = config.agent_type
            agent = agent_class(
                config=agent_config,
                tools=tools,
                memory=memory,
                llm_provider=llm_provider,
            )

            # Add to hierarchy
            await self.hierarchy.add_agent(
                agent_id=agent_id,
                agent=agent,
                parent_id=config.parent_id,
                relationship=config.relationship,
            )

            # Track active spawn
            self._active_spawns[agent_id] = config

            # Initialize agent with context if needed
            if context and config.share_context:
                await self._share_context(agent, context)

            spawn_time = time.time() - start_time

            logger.info(
                f"Successfully spawned agent {agent_id} of type {agent_class.__name__} in {spawn_time:.2f}s"
            )

            return SpawnResult(
                success=True,
                agent_id=agent_id,
                agent=agent,
                spawn_time=spawn_time,
                metadata={
                    "task": task,
                    "config": config,
                    "context_shared": bool(context and config.share_context),
                },
            )

        except Exception as e:
            logger.error(f"Failed to spawn agent: {e}", exc_info=True)
            return SpawnResult(
                success=False,
                agent_id=agent_id,
                error=str(e),
                spawn_time=time.time() - start_time,
            )

    async def spawn_team(
        self,
        task: str,
        team_size: int,
        roles: Optional[List[AgentRole]] = None,
        coordinator_config: Optional[SpawnConfig] = None,
    ) -> Dict[str, SpawnResult]:
        """Spawn a team of agents with a coordinator."""
        results = {}

        # Spawn coordinator first
        coordinator_config = coordinator_config or SpawnConfig(
            agent_config=AgentConfig(
                role=AgentRole.COORDINATOR,
                name="TeamCoordinator",
            )
        )

        coordinator_result = await self.spawn_agent(
            task=f"Coordinate team for: {task}",
            config=coordinator_config,
        )
        results["coordinator"] = coordinator_result

        if not coordinator_result.success:
            return results

        # Spawn team members
        roles = roles or [AgentRole.GENERALIST] * team_size

        for i, role in enumerate(roles[:team_size]):
            member_config = SpawnConfig(
                agent_config=AgentConfig(
                    role=role,
                    name=f"TeamMember_{i}_{role.value}",
                ),
                parent_id=coordinator_result.agent_id,
                relationship=AgentRelationship.COORDINATOR_SUBORDINATE,
            )

            member_result = await self.spawn_agent(
                task=f"Team member for: {task}",
                config=member_config,
            )
            results[f"member_{i}"] = member_result

        return results

    async def terminate_agent(self, agent_id: str, cascade: bool = True) -> bool:
        """Terminate an agent and optionally its descendants."""
        try:
            # Remove from hierarchy (handles cleanup)
            await self.hierarchy.remove_agent(agent_id, cascade=cascade)

            # Remove from active spawns
            self._active_spawns.pop(agent_id, None)

            # Free resources
            await self._resource_monitor.free_resources(agent_id)

            logger.info(f"Terminated agent {agent_id} (cascade={cascade})")
            return True

        except Exception as e:
            logger.error(f"Failed to terminate agent {agent_id}: {e}")
            return False

    async def pause_agent(self, agent_id: str) -> bool:
        """Pause an agent's execution."""
        if agent_id not in self.hierarchy.nodes:
            return False

        node = self.hierarchy.nodes[agent_id]
        node.is_paused = True
        node.is_active = False

        logger.info(f"Paused agent {agent_id}")
        return True

    async def resume_agent(self, agent_id: str) -> bool:
        """Resume a paused agent."""
        if agent_id not in self.hierarchy.nodes:
            return False

        node = self.hierarchy.nodes[agent_id]
        node.is_paused = False
        node.is_active = True
        node.last_activity = time.time()

        logger.info(f"Resumed agent {agent_id}")
        return True

    async def _check_resources(self, config: SpawnConfig) -> bool:
        """Check if sufficient resources are available."""
        return await self._resource_monitor.check_availability(
            memory_mb=config.max_memory_mb,
            api_calls=config.max_api_calls,
        )

    async def _create_agent_config(
        self,
        task: str,
        spawn_config: SpawnConfig,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentConfig:
        """Create agent configuration based on task and spawn config."""
        if spawn_config.agent_config:
            config = spawn_config.agent_config
        else:
            # Auto-generate configuration based on task
            config = AgentConfig(
                name=f"Agent_{self._spawn_counter}",
                role=AgentRole.GENERALIST,
                description=f"Spawned agent for task: {task[:100]}",
                max_iterations=10,
                temperature=0.7,
            )

        # Apply resource limits
        config.timeout = spawn_config.max_execution_time
        config.metadata["spawn_config"] = spawn_config
        config.metadata["original_task"] = task

        return config

    async def _initialize_tools(
        self,
        spawn_config: SpawnConfig,
        agent_config: AgentConfig,
    ) -> ToolRegistry:
        """Initialize tools for the spawned agent."""
        if spawn_config.inherit_tools and spawn_config.parent_id:
            # Inherit tools from parent
            parent_node = self.hierarchy.nodes.get(spawn_config.parent_id)
            if parent_node and hasattr(parent_node.agent, "tools"):
                return parent_node.agent.tools

        # Create new tool registry
        tools = ToolRegistry()

        # Add requested tools
        for tool_name in agent_config.tools:
            tool = self.tool_registry.get(tool_name)
            if tool:
                tools.register(tool)

        return tools

    async def _initialize_memory(
        self,
        spawn_config: SpawnConfig,
        agent_config: AgentConfig,
    ) -> Optional[SimpleMemory]:
        """Initialize memory for the spawned agent."""
        if not agent_config.memory_enabled:
            return None

        if spawn_config.inherit_memory and spawn_config.parent_id:
            # Share memory with parent
            parent_node = self.hierarchy.nodes.get(spawn_config.parent_id)
            if parent_node and hasattr(parent_node.agent, "memory"):
                return parent_node.agent.memory

        # Create new memory
        return SimpleMemory()

    def _get_llm_provider(self, config: SpawnConfig) -> Any:
        """Get LLM provider for the agent."""
        if config.agent_config and config.agent_config.llm_provider:
            return config.agent_config.llm_provider

        return self.default_llm_provider or create_provider("mock")

    async def _share_context(self, agent: BaseAgent, context: Dict[str, Any]) -> None:
        """Share context with the newly spawned agent."""
        if hasattr(agent, "memory") and agent.memory:
            # Add context to agent's memory
            for key, value in context.items():
                await agent.memory.add(f"context_{key}", value)

    def get_stats(self) -> Dict[str, Any]:
        """Get spawner statistics."""
        hierarchy_stats = self.hierarchy.get_hierarchy_stats()

        return {
            "total_spawned": self._spawn_counter,
            "active_spawns": len(self._active_spawns),
            "hierarchy": hierarchy_stats,
            "resource_usage": self._resource_monitor.get_usage(),
        }


class ResourceMonitor:
    """Monitors and manages resource usage across spawned agents."""

    def __init__(
        self,
        max_total_memory_mb: float = 4096.0,
        max_total_api_calls: int = 10000,
    ) -> None:
        self.max_total_memory_mb = max_total_memory_mb
        self.max_total_api_calls = max_total_api_calls
        self.current_memory_mb = 0.0
        self.current_api_calls = 0
        self._agent_resources: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def check_availability(
        self,
        memory_mb: float,
        api_calls: int,
    ) -> bool:
        """Check if resources are available."""
        async with self._lock:
            return (
                self.current_memory_mb + memory_mb <= self.max_total_memory_mb
                and self.current_api_calls + api_calls <= self.max_total_api_calls
            )

    async def allocate_resources(
        self,
        agent_id: str,
        memory_mb: float,
        api_calls: int,
    ) -> bool:
        """Allocate resources for an agent."""
        async with self._lock:
            if not await self.check_availability(memory_mb, api_calls):
                return False

            self.current_memory_mb += memory_mb
            self.current_api_calls += api_calls

            self._agent_resources[agent_id] = {
                "memory_mb": memory_mb,
                "api_calls": api_calls,
                "allocated_at": time.time(),
            }

            return True

    async def free_resources(self, agent_id: str) -> None:
        """Free resources allocated to an agent."""
        async with self._lock:
            if agent_id in self._agent_resources:
                resources = self._agent_resources[agent_id]
                self.current_memory_mb -= resources["memory_mb"]
                self.current_api_calls -= resources["api_calls"]
                del self._agent_resources[agent_id]

    def get_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        return {
            "memory": {
                "used_mb": self.current_memory_mb,
                "total_mb": self.max_total_memory_mb,
                "percentage": (self.current_memory_mb / self.max_total_memory_mb) * 100,
            },
            "api_calls": {
                "used": self.current_api_calls,
                "total": self.max_total_api_calls,
                "percentage": (self.current_api_calls / self.max_total_api_calls) * 100,
            },
            "agent_count": len(self._agent_resources),
        }

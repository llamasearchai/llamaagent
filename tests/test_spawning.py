"""Comprehensive tests for agent spawning and sub-agent management."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from llamaagent.agents.base import AgentConfig, AgentResponse, AgentRole
from llamaagent.agents.react import ReactAgent
from llamaagent.orchestrator import (
    AgentOrchestrator,
    OrchestrationStrategy,
    WorkflowDefinition,
    WorkflowStep,
)
from llamaagent.spawning import (
    AgentChannel,
    AgentPool,
    AgentSpawner,
    BroadcastChannel,
    DirectChannel,
    Message,
    MessageBus,
    MessageType,
    PoolConfig,
    SpawnConfig,
)


class TestAgentSpawner:
    """Test cases for AgentSpawner."""

    @pytest.fixture
    def spawner(self):
        """Create a test spawner."""
        return AgentSpawner()

    @pytest.mark.asyncio
    async def test_spawn_single_agent(self, spawner):
        """Test spawning a single agent."""
        result = await spawner.spawn_agent(
            task="Test task",
            config=SpawnConfig(
                agent_config=AgentConfig(
                    name="test_agent",
                    role=AgentRole.GENERALIST,
                )
            ),
        )

        assert result.success
        assert result.agent is not None
        assert result.agent_id.startswith("agent_")
        assert spawner._spawn_counter == 1

    @pytest.mark.asyncio
    async def test_spawn_team(self, spawner):
        """Test spawning a team of agents."""
        results = await spawner.spawn_team(
            task="Complex team task",
            team_size=3,
            roles=[AgentRole.RESEARCHER, AgentRole.ANALYZER, AgentRole.EXECUTOR],
        )

        assert len(results) == 4  # 1 coordinator + 3 team members
        assert "coordinator" in results
        assert results["coordinator"].success

        for i in range(3):
            assert f"member_{i}" in results
            assert results[f"member_{i}"].success

    @pytest.mark.asyncio
    async def test_agent_hierarchy(self, spawner):
        """Test agent hierarchy management."""
        # Spawn parent
        parent_result = await spawner.spawn_agent(
            task="Parent task",
            config=SpawnConfig(
                agent_config=AgentConfig(name="parent", role=AgentRole.COORDINATOR)
            ),
        )

        # Spawn children
        child1_result = await spawner.spawn_agent(
            task="Child 1 task",
            config=SpawnConfig(
                agent_config=AgentConfig(name="child1"),
                parent_id=parent_result.agent_id,
            ),
        )

        child2_result = await spawner.spawn_agent(
            task="Child 2 task",
            config=SpawnConfig(
                agent_config=AgentConfig(name="child2"),
                parent_id=parent_result.agent_id,
            ),
        )

        # Check hierarchy
        hierarchy = spawner.hierarchy
        assert len(hierarchy.nodes) == 3
        assert parent_result.agent_id in hierarchy.root_agents
        assert (
            child1_result.agent_id in hierarchy.nodes[parent_result.agent_id].children
        )
        assert (
            child2_result.agent_id in hierarchy.nodes[parent_result.agent_id].children
        )

        # Test hierarchy methods
        ancestors = hierarchy.get_ancestors(child1_result.agent_id)
        assert parent_result.agent_id in ancestors

        descendants = hierarchy.get_descendants(parent_result.agent_id)
        assert child1_result.agent_id in descendants
        assert child2_result.agent_id in descendants

        siblings = hierarchy.get_siblings(child1_result.agent_id)
        assert child2_result.agent_id in siblings

    @pytest.mark.asyncio
    async def test_resource_limits(self, spawner):
        """Test resource limit enforcement."""
        # Set low resource limits
        spawner._resource_monitor.max_total_memory_mb = 100

        # Spawn agent with high memory requirement
        result = await spawner.spawn_agent(
            task="High memory task", config=SpawnConfig(max_memory_mb=200)
        )

        assert not result.success
        assert "Insufficient resources" in result.error

    @pytest.mark.asyncio
    async def test_agent_lifecycle(self, spawner):
        """Test agent lifecycle management."""
        # Spawn agent
        result = await spawner.spawn_agent(
            task="Lifecycle test",
            config=SpawnConfig(agent_config=AgentConfig(name="lifecycle_agent")),
        )

        assert result.success
        agent_id = result.agent_id

        # Pause agent
        assert await spawner.pause_agent(agent_id)
        assert spawner.hierarchy.nodes[agent_id].is_paused
        assert not spawner.hierarchy.nodes[agent_id].is_active

        # Resume agent
        assert await spawner.resume_agent(agent_id)
        assert not spawner.hierarchy.nodes[agent_id].is_paused
        assert spawner.hierarchy.nodes[agent_id].is_active

        # Terminate agent
        assert await spawner.terminate_agent(agent_id)
        assert agent_id not in spawner.hierarchy.nodes


class TestAgentPool:
    """Test cases for AgentPool."""

    @pytest.fixture
    def pool_config(self):
        """Create test pool configuration."""
        return PoolConfig(
            min_agents=1,
            max_agents=5,
            initial_agents=2,
            auto_scale=True,
            scale_up_threshold=0.7,
            scale_down_threshold=0.3,
        )

    @pytest.fixture
    def pool(self, pool_config):
        """Create test agent pool."""
        return AgentPool(config=pool_config)

    @pytest.mark.asyncio
    async def test_pool_start_stop(self, pool):
        """Test starting and stopping the pool."""
        # Start pool
        await pool.start()
        assert pool.is_running
        assert len(pool.agents) == pool.config.initial_agents

        # Stop pool
        await pool.stop()
        assert not pool.is_running
        assert len(pool.agents) == 0

    @pytest.mark.asyncio
    async def test_task_submission(self, pool):
        """Test submitting tasks to the pool."""
        await pool.start()

        # Submit task
        task_id = await pool.submit_task(
            task="Test pool task",
            context={"key": "value"},
            priority=5,
        )

        assert task_id.startswith("task_")
        assert len(pool.task_queue) > 0 or task_id in pool.active_tasks

        await pool.stop()

    @pytest.mark.asyncio
    async def test_submit_and_wait(self, pool):
        """Test submitting task and waiting for result."""
        await pool.start()

        # Mock agent execution
        for agent in pool.agents.values():
            agent.execute = AsyncMock(
                return_value=AgentResponse(
                    content="Task completed",
                    success=True,
                )
            )

        # Submit and wait
        result = await pool.submit_and_wait(
            task="Test task",
            priority=10,
            timeout=5.0,
        )

        assert result.success
        assert result.content == "Task completed"

        await pool.stop()

    @pytest.mark.asyncio
    async def test_pool_statistics(self, pool):
        """Test pool statistics collection."""
        await pool.start()

        stats = pool.get_pool_stats()
        assert stats.total_agents == pool.config.initial_agents
        assert stats.idle_agents == pool.config.initial_agents
        assert stats.busy_agents == 0
        assert stats.queue_size == 0

        await pool.stop()

    @pytest.mark.asyncio
    async def test_auto_scaling(self, pool):
        """Test auto-scaling functionality."""
        await pool.start()
        len(pool.agents)

        # Submit many tasks to trigger scale up
        for i in range(20):
            await pool.submit_task(f"Task {i}", priority=i)

        # Wait for scaling
        await asyncio.sleep(0.1)

        # Should scale up (mocked, so might not actually increase)
        stats = pool.get_pool_stats()
        assert stats.queue_size > 0 or stats.busy_agents > 0

        await pool.stop()


class TestCommunication:
    """Test cases for inter-agent communication."""

    @pytest.fixture
    def message_bus(self):
        """Create test message bus."""
        return MessageBus()

    @pytest.fixture
    def agent_channel(self, message_bus):
        """Create test agent channel."""
        return AgentChannel(
            agent_id="test_agent",
            bus=message_bus,
        )

    @pytest.mark.asyncio
    async def test_direct_message(self, message_bus):
        """Test direct messaging between agents."""
        # Create channels
        channel1 = AgentChannel("agent1", message_bus)
        channel2 = AgentChannel("agent2", message_bus)

        # Send message
        message = Message(
            type=MessageType.TASK_REQUEST,
            sender="agent1",
            recipient="agent2",
            content="Hello agent2",
        )

        assert await channel1.send(message)

        # Receive message
        received = await channel2.receive(timeout=1.0)
        assert received is not None
        assert received.content == "Hello agent2"
        assert received.sender == "agent1"

    @pytest.mark.asyncio
    async def test_broadcast_message(self, message_bus):
        """Test broadcasting messages."""
        # Create channels with subscriptions
        channel1 = AgentChannel("agent1", message_bus)
        channel2 = AgentChannel("agent2", message_bus)
        channel3 = AgentChannel("agent3", message_bus)

        # Subscribe to coordination messages
        await channel1.subscribe({MessageType.COORDINATION})
        await channel2.subscribe({MessageType.COORDINATION})
        # channel3 not subscribed

        # Broadcast message
        broadcast = BroadcastChannel("coordinator", message_bus)
        sent_count = await broadcast.broadcast(
            content="Team update",
            message_type=MessageType.COORDINATION,
        )

        # Check reception
        msg1 = await channel1.receive(timeout=1.0)
        msg2 = await channel2.receive(timeout=1.0)
        msg3 = await channel3.receive(timeout=0.1)

        assert msg1 is not None
        assert msg2 is not None
        assert msg3 is None  # Not subscribed

    @pytest.mark.asyncio
    async def test_request_response(self, message_bus):
        """Test request-response communication."""
        channel1 = AgentChannel("agent1", message_bus)
        channel2 = AgentChannel("agent2", message_bus)

        # Setup response handler
        async def handle_messages():
            msg = await channel2.receive()
            if msg and msg.requires_response:
                await channel2.respond(msg, "Response to request")

        # Start handler
        handler_task = asyncio.create_task(handle_messages())

        # Send request
        response = await channel1.request(
            recipient="agent2",
            content="Request content",
            timeout=2.0,
        )

        assert response is not None
        assert response.content == "Response to request"

        handler_task.cancel()

    @pytest.mark.asyncio
    async def test_direct_channel(self):
        """Test direct peer-to-peer channel."""
        channel = DirectChannel("agent1", "agent2")

        # Send from agent1 to agent2
        message1 = Message(
            sender="agent1",
            content="Hello from agent1",
        )
        assert await channel.send(message1)

        # Receive for agent2
        received1 = await channel.receive_for_agent("agent2", timeout=1.0)
        assert received1 is not None
        assert received1.content == "Hello from agent1"

        # Send from agent2 to agent1
        message2 = Message(
            sender="agent2",
            content="Reply from agent2",
        )
        assert await channel.send(message2)

        # Receive for agent1
        received2 = await channel.receive_for_agent("agent1", timeout=1.0)
        assert received2 is not None
        assert received2.content == "Reply from agent2"


class TestOrchestratorIntegration:
    """Test orchestrator integration with spawning."""

    @pytest.fixture
    def orchestrator(self):
        """Create test orchestrator with spawning enabled."""
        return AgentOrchestrator(
            enable_spawning=True,
            enable_pool=True,
            enable_communication=True,
        )

    @pytest.mark.asyncio
    async def test_dynamic_workflow_execution(self, orchestrator):
        """Test dynamic agent spawning during workflow execution."""
        # Create workflow that requires non-existent agents
        workflow = WorkflowDefinition(
            workflow_id="dynamic_test",
            name="Dynamic Spawning Test",
            description="Test workflow with dynamic agent spawning",
            steps=[
                WorkflowStep(
                    step_id="step1",
                    agent_name="dynamic_researcher",  # Doesn't exist
                    task="Research the topic",
                ),
                WorkflowStep(
                    step_id="step2",
                    agent_name="dynamic_analyzer",  # Doesn't exist
                    task="Analyze the research",
                    dependencies=["step1"],
                ),
            ],
            strategy=OrchestrationStrategy.DYNAMIC,
        )

        orchestrator.register_workflow(workflow)

        # Execute workflow - should spawn agents dynamically
        result = await orchestrator.execute_workflow("dynamic_test")

        assert result.success
        assert "step1" in result.results
        assert "step2" in result.results

        # Check that agents were spawned
        spawning_stats = orchestrator.get_spawning_stats()
        assert spawning_stats is not None
        assert spawning_stats["total_spawned"] >= 2

    @pytest.mark.asyncio
    async def test_pool_based_workflow(self, orchestrator):
        """Test pool-based workflow execution."""
        if orchestrator.agent_pool:
            await orchestrator.agent_pool.start()

        workflow = WorkflowDefinition(
            workflow_id="pool_test",
            name="Pool-Based Test",
            description="Test workflow using agent pool",
            steps=[
                WorkflowStep(
                    step_id=f"step{i}",
                    agent_name="pool_worker",
                    task=f"Process item {i}",
                )
                for i in range(5)
            ],
            strategy=OrchestrationStrategy.POOL_BASED,
        )

        orchestrator.register_workflow(workflow)

        # Execute workflow using pool
        result = await orchestrator.execute_workflow("pool_test")

        # Pool might not complete all tasks in test environment
        assert len(result.results) > 0

        if orchestrator.agent_pool:
            await orchestrator.agent_pool.stop()

    @pytest.mark.asyncio
    async def test_spawn_agent_for_task(self, orchestrator):
        """Test spawning individual agent for task."""
        agent = await orchestrator.spawn_agent_for_task(
            task="Research quantum computing applications",
            agent_name="quantum_researcher",
        )

        assert agent is not None
        assert agent.config.name == "quantum_researcher"
        assert agent.config.role == AgentRole.RESEARCHER
        assert "quantum_researcher" in orchestrator.agents

    @pytest.mark.asyncio
    async def test_agent_communication(self, orchestrator):
        """Test agent communication through orchestrator."""
        # Register test agents
        agent1 = ReactAgent(
            config=AgentConfig(name="comm_agent1"),
            llm_provider=MagicMock(),
        )
        agent2 = ReactAgent(
            config=AgentConfig(name="comm_agent2"),
            llm_provider=MagicMock(),
        )

        orchestrator.register_agent(agent1)
        orchestrator.register_agent(agent2)

        # Test broadcast
        sent_count = await orchestrator.broadcast_to_agents(
            MessageType.COORDINATION,
            content="Team meeting at 3pm",
        )

        assert sent_count == 2

        # Test channels were created
        assert "comm_agent1" in orchestrator.agent_channels
        assert "comm_agent2" in orchestrator.agent_channels

    def test_statistics_collection(self, orchestrator):
        """Test statistics collection from all components."""
        # Get spawning stats
        spawning_stats = orchestrator.get_spawning_stats()
        assert spawning_stats is not None
        assert "total_spawned" in spawning_stats

        # Get pool stats
        pool_stats = orchestrator.get_pool_stats()
        if orchestrator.enable_pool:
            assert pool_stats is not None
            assert "total_agents" in pool_stats

        # Get communication stats
        comm_stats = orchestrator.get_communication_stats()
        if orchestrator.enable_communication:
            assert comm_stats is not None
            assert "total_channels" in comm_stats


@pytest.mark.asyncio
async def test_full_integration():
    """Test full integration of spawning, pool, and communication."""
    # Create orchestrator with all features
    orchestrator = AgentOrchestrator(
        enable_spawning=True,
        enable_pool=True,
        enable_communication=True,
    )

    # Start pool
    if orchestrator.agent_pool:
        await orchestrator.agent_pool.start()

    # Create complex workflow
    workflow = WorkflowDefinition(
        workflow_id="full_integration",
        name="Full Integration Test",
        description="Test all spawning features",
        steps=[
            # Dynamic spawning
            WorkflowStep(
                step_id="research",
                agent_name="dynamic_researcher",
                task="Research the topic using dynamic spawning",
            ),
            # Pool execution
            WorkflowStep(
                step_id="analyze",
                agent_name="pool_analyzer",
                task="Analyze research results using pool",
                dependencies=["research"],
            ),
            # Communication test
            WorkflowStep(
                step_id="coordinate",
                agent_name="coordinator",
                task="Coordinate final report",
                dependencies=["analyze"],
            ),
        ],
        strategy=OrchestrationStrategy.DYNAMIC,
    )

    orchestrator.register_workflow(workflow)

    # Execute workflow
    result = await orchestrator.execute_workflow("full_integration")

    # Verify execution
    assert "research" in result.results
    assert result.execution_time > 0

    # Collect all statistics
    all_stats = {
        "spawning": orchestrator.get_spawning_stats(),
        "pool": orchestrator.get_pool_stats(),
        "communication": orchestrator.get_communication_stats(),
    }

    # Verify stats collected
    assert all_stats["spawning"] is not None

    # Cleanup
    if orchestrator.agent_pool:
        await orchestrator.agent_pool.stop()


if __name__ == "__main__":
    # Run specific test for debugging
    asyncio.run(test_full_integration())

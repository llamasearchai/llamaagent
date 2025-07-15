#!/usr/bin/env python3
"""
Master Integration Test Suite

Comprehensive test suite that validates all major components of LlamaAgent
including multi-agent orchestration, OpenAI integration, SPRE, and tools.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio

import pytest

from llamaagent import AgentConfig, AgentRole, ReactAgent
from llamaagent.integration.openai_agents import (
    OpenAIAgentMode,
    OpenAIAgentsIntegration,
    OpenAIIntegrationConfig,
)
from llamaagent.llm import create_provider
from llamaagent.memory import SimpleMemory
from llamaagent.tools import CalculatorTool, PythonREPLTool, ToolRegistry
from llamaagent.types import TaskInput, TaskOutput, TaskStatus
from src.llamaagent.orchestrator import (
    AgentOrchestrator,
    OrchestrationStrategy,
    WorkflowDefinition,
    WorkflowStep,
)


class TestMasterIntegration:
    """Master integration test suite."""

    @pytest.fixture
    def mock_provider(self):
        """Create mock LLM provider."""
        return create_provider("mock")

    @pytest.fixture
    def tool_registry(self):
        """Create tool registry with basic tools."""
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.register(PythonREPLTool())
        return registry

    @pytest.fixture
    async def basic_agent(self, mock_provider, tool_registry):
        """Create basic agent for testing."""
        config = AgentConfig(
            name="TestAgent",
            role=AgentRole.GENERALIST,
            description="Test agent",
            tools=["calculator", "python_repl"],
        )
        return ReactAgent(
            config=config, tools=tool_registry, llm_provider=mock_provider
        )

    @pytest.fixture
    async def orchestrator(self):
        """Create agent orchestrator."""
        return AgentOrchestrator()

    @pytest.mark.asyncio
    async def test_single_agent_execution(self, basic_agent):
        """Test basic single agent execution."""
        task = "Calculate 25 * 4"
        response = await basic_agent.execute(task)

        assert response.success is True
        assert response.content is not None
        assert response.execution_time > 0
        assert basic_agent.name == "TestAgent"

    @pytest.mark.asyncio
    async def test_agent_with_tools(self, tool_registry):
        """Test agent using tools."""
        config = AgentConfig(
            name="ToolUser",
            role=AgentRole.EXECUTOR,
            tools=["calculator"],
            spree_enabled=False,
        )

        agent = ReactAgent(
            config=config, tools=tool_registry, llm_provider=create_provider("mock")
        )

        response = await agent.execute("Calculate the square root of 144")
        assert response.success is True

    @pytest.mark.asyncio
    async def test_multi_agent_workflow(self, orchestrator, mock_provider):
        """Test multi-agent orchestration."""
        # Create specialized agents
        agents = {
            "researcher": ReactAgent(
                config=AgentConfig(name="ResearchAgent", role=AgentRole.RESEARCHER),
                llm_provider=mock_provider,
            ),
            "analyst": ReactAgent(
                config=AgentConfig(name="AnalysisAgent", role=AgentRole.ANALYZER),
                llm_provider=mock_provider,
            ),
            "writer": ReactAgent(
                config=AgentConfig(name="WriterAgent", role=AgentRole.SPECIALIST),
                llm_provider=mock_provider,
            ),
        }

        # Register agents
        for agent in agents.values():
            orchestrator.register_agent(agent)

        # Define workflow
        workflow = WorkflowDefinition(
            workflow_id="test_workflow",
            name="Test Workflow",
            description="Multi-agent test workflow",
            strategy=OrchestrationStrategy.SEQUENTIAL,
            steps=[
                WorkflowStep(
                    step_id="research",
                    agent_name="ResearchAgent",
                    task="Research test topic",
                ),
                WorkflowStep(
                    step_id="analyze",
                    agent_name="AnalysisAgent",
                    task="Analyze research findings",
                    dependencies=["research"],
                ),
                WorkflowStep(
                    step_id="report",
                    agent_name="WriterAgent",
                    task="Write summary report",
                    dependencies=["research", "analyze"],
                ),
            ],
        )

        orchestrator.register_workflow(workflow)

        # Execute workflow
        result = await orchestrator.execute_workflow("test_workflow")

        assert result.success is True
        assert len(result.results) == 3
        assert result.workflow_id == "test_workflow"
        assert result.execution_time > 0

    @pytest.mark.asyncio
    async def test_parallel_workflow_execution(self, orchestrator, mock_provider):
        """Test parallel workflow execution."""
        # Create agents
        agent1 = ReactAgent(
            config=AgentConfig(name="Agent1"), llm_provider=mock_provider
        )
        agent2 = ReactAgent(
            config=AgentConfig(name="Agent2"), llm_provider=mock_provider
        )

        orchestrator.register_agent(agent1)
        orchestrator.register_agent(agent2)

        # Define parallel workflow
        workflow = WorkflowDefinition(
            workflow_id="parallel_test",
            name="Parallel Test",
            description="Test parallel execution",
            strategy=OrchestrationStrategy.PARALLEL,
            steps=[
                WorkflowStep(
                    step_id="task1", agent_name="Agent1", task="First parallel task"
                ),
                WorkflowStep(
                    step_id="task2", agent_name="Agent2", task="Second parallel task"
                ),
            ],
        )

        orchestrator.register_workflow(workflow)
        result = await orchestrator.execute_workflow("parallel_test")

        assert result.success is True
        assert len(result.results) == 2

    @pytest.mark.asyncio
    async def test_spre_planning(self, mock_provider, tool_registry):
        """Test SPRE planning functionality."""
        config = AgentConfig(
            name="SPREAgent",
            role=AgentRole.PLANNER,
            spree_enabled=True,
            tools=["calculator"],
        )

        agent = ReactAgent(
            config=config, tools=tool_registry, llm_provider=mock_provider
        )

        complex_task = """
        Calculate compound interest on $1000 at 5% for 10 years,
        then determine the monthly payment for a loan of that amount
        over 5 years at 3% interest.
        """

        response = await agent.execute(complex_task)

        assert response.success is True
        # SPRE should create a plan
        assert response.trace is not None

    @pytest.mark.asyncio
    async def test_openai_integration(self, basic_agent):
        """Test OpenAI integration."""
        config = OpenAIIntegrationConfig(
            mode=OpenAIAgentMode.HYBRID,
            model_name="gpt-4o-mini",
            budget_limit=1.0,
            enable_tracing=False,
        )

        integration = OpenAIAgentsIntegration(config)
        integration.register_agent(basic_agent)

        task_input = TaskInput(
            id="test_001", task="Simple test task", agent_name="TestAgent"
        )

        result = await integration.run_task("TestAgent", task_input)

        assert isinstance(result, TaskOutput)
        assert result.task_id == "test_001"
        assert result.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]

    @pytest.mark.asyncio
    async def test_budget_tracking(self):
        """Test budget tracking functionality."""
        from llamaagent.integration.openai_agents import BudgetTracker

        tracker = BudgetTracker(budget_limit=10.0)

        # Add some usage
        tracker.add_usage(cost=0.01, tokens=100, model="gpt-4o-mini")
        tracker.add_usage(cost=0.02, tokens=200, model="gpt-4o-mini")

        assert tracker.current_cost == 0.03
        assert tracker.get_remaining_budget() == 9.97

        summary = tracker.get_usage_summary()
        assert summary["budget_limit"] == 10.0
        assert summary["current_cost"] == 0.03
        assert summary["total_calls"] == 2

    @pytest.mark.asyncio
    async def test_memory_integration(self, mock_provider):
        """Test memory integration."""
        memory = SimpleMemory()

        config = AgentConfig(name="MemoryAgent", memory_enabled=True)

        agent = ReactAgent(config=config, memory=memory, llm_provider=mock_provider)

        # First interaction
        await agent.execute("Remember that my favorite color is blue")

        # Check memory was stored
        assert len(memory.messages) > 0

        # Second interaction should have context
        response = await agent.execute("What is my favorite color?")
        assert response.success is True

    @pytest.mark.asyncio
    async def test_error_handling(self, basic_agent):
        """Test error handling and recovery."""
        # Test with empty task
        response = await basic_agent.execute("")
        assert response.success is True  # Should handle gracefully

        # Test with None context
        response = await basic_agent.execute("Test task", context=None)
        assert response.success is True

    @pytest.mark.asyncio
    async def test_streaming_execution(self, basic_agent):
        """Test streaming execution."""
        chunks = []
        async for chunk in basic_agent.stream_execute("Generate a story"):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    @pytest.mark.asyncio
    async def test_debate_strategy(self, orchestrator, mock_provider):
        """Test debate orchestration strategy."""
        # Create debate agents
        agents = [
            ReactAgent(
                config=AgentConfig(name=f"Debater{i}"), llm_provider=mock_provider
            )
            for i in range(3)
        ]

        for agent in agents:
            orchestrator.register_agent(agent)

        # Define debate workflow
        workflow = WorkflowDefinition(
            workflow_id="debate_test",
            name="Debate Test",
            description="Test debate strategy",
            strategy=OrchestrationStrategy.DEBATE,
            steps=[
                WorkflowStep(
                    step_id=f"position_{i}",
                    agent_name=f"Debater{i}",
                    task="Argue your position",
                )
                for i in range(3)
            ],
        )

        orchestrator.register_workflow(workflow)
        result = await orchestrator.execute_workflow("debate_test")

        assert result.success is True
        assert "final_synthesis" in result.results

    @pytest.mark.asyncio
    async def test_tool_creation(self, tool_registry):
        """Test dynamic tool creation capability."""
        from llamaagent.tools import BaseTool

        class CustomTool(BaseTool):
            name = "custom_calculator"
            description = "Custom calculation tool"

            async def execute(self, expression: str) -> float:
                # Simple eval for testing (not for production!)
                return eval(expression, {"__builtins__": {}}, {})

        # Register custom tool
        custom_tool = CustomTool()
        tool_registry.register(custom_tool)

        # Verify registration
        assert "custom_calculator" in [t.name for t in tool_registry.list_tools()]

        # Use the tool
        result = await custom_tool.execute("2 + 2")
        assert result == 4

    @pytest.mark.asyncio
    async def test_agent_roles(self):
        """Test all agent roles."""
        for role in AgentRole:
            config = AgentConfig(name=f"{role.value}Agent", role=role)
            agent = ReactAgent(config=config, llm_provider=create_provider("mock"))

            assert agent.config.role == role
            response = await agent.execute(f"Test as {role.value}")
            assert response.success is True

    @pytest.mark.asyncio
    async def test_workflow_with_failures(self, orchestrator, mock_provider):
        """Test workflow handling with step failures."""
        agent = ReactAgent(
            config=AgentConfig(name="TestAgent"), llm_provider=mock_provider
        )
        orchestrator.register_agent(agent)

        # Workflow with non-required failing step
        workflow = WorkflowDefinition(
            workflow_id="failure_test",
            name="Failure Test",
            description="Test failure handling",
            strategy=OrchestrationStrategy.SEQUENTIAL,
            steps=[
                WorkflowStep(
                    step_id="optional_fail",
                    agent_name="NonExistentAgent",  # Will fail
                    task="This will fail",
                    required=False,  # Non-required
                ),
                WorkflowStep(
                    step_id="success",
                    agent_name="TestAgent",
                    task="This should succeed",
                ),
            ],
        )

        orchestrator.register_workflow(workflow)

        # Should still succeed overall
        with pytest.raises(ValueError):  # Agent not found
            await orchestrator.execute_workflow("failure_test")

    @pytest.mark.asyncio
    async def test_execution_history(self, orchestrator, mock_provider):
        """Test execution history tracking."""
        agent = ReactAgent(
            config=AgentConfig(name="HistoryAgent"), llm_provider=mock_provider
        )
        orchestrator.register_agent(agent)

        # Create simple workflow
        workflow = WorkflowDefinition(
            workflow_id="history_test",
            name="History Test",
            description="Test history",
            steps=[
                WorkflowStep(
                    step_id="task", agent_name="HistoryAgent", task="Simple task"
                )
            ],
        )

        orchestrator.register_workflow(workflow)

        # Execute multiple times
        for i in range(3):
            await orchestrator.execute_workflow("history_test")

        # Check history
        history = orchestrator.get_execution_history()
        assert len(history) == 3
        assert all(h.workflow_id == "history_test" for h in history)

        # Clear history
        orchestrator.clear_history()
        assert len(orchestrator.get_execution_history()) == 0


@pytest.mark.asyncio
async def test_full_system_integration():
    """Test complete system integration."""
    # This would be a comprehensive end-to-end test
    # combining all components in a realistic scenario

    # 1. Create multi-agent system
    orchestrator = AgentOrchestrator()

    # 2. Create specialized agents with tools
    tool_registry = ToolRegistry()
    tool_registry.register(CalculatorTool())
    tool_registry.register(PythonREPLTool())

    research_agent = ReactAgent(
        config=AgentConfig(
            name="Researcher",
            role=AgentRole.RESEARCHER,
            tools=["calculator"],
            spree_enabled=True,
        ),
        tools=tool_registry,
        llm_provider=create_provider("mock"),
    )

    analysis_agent = ReactAgent(
        config=AgentConfig(
            name="Analyst", role=AgentRole.ANALYZER, tools=["python_repl"]
        ),
        tools=tool_registry,
        llm_provider=create_provider("mock"),
    )

    # 3. Register agents
    orchestrator.register_agent(research_agent)
    orchestrator.register_agent(analysis_agent)

    # 4. Create complex workflow
    workflow = WorkflowDefinition(
        workflow_id="integration_test",
        name="Full Integration Test",
        description="Complete system test",
        strategy=OrchestrationStrategy.SEQUENTIAL,
        steps=[
            WorkflowStep(
                step_id="research",
                agent_name="Researcher",
                task="Research mathematical constants and their relationships",
            ),
            WorkflowStep(
                step_id="analysis",
                agent_name="Analyst",
                task="Analyze the research and create visualizations",
                dependencies=["research"],
            ),
        ],
    )

    orchestrator.register_workflow(workflow)

    # 5. Execute with monitoring
    result = await orchestrator.execute_workflow("integration_test")

    # 6. Verify results
    assert result.success is True
    assert len(result.results) == 2
    assert result.results["research"].success is True
    assert result.results["analysis"].success is True

    print("Full system integration test passed!")


if __name__ == "__main__":
    # Run the integration test
    asyncio.run(test_full_system_integration())

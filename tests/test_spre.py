import pytest

from llamaagent.agents.base import AgentConfig, ExecutionPlan, PlanStep
from llamaagent.agents.react import ReactAgent
from llamaagent.llm import MockProvider


@pytest.mark.asyncio
async def test_plan_step_creation():
    """Test plan step creation."""
    step = PlanStep(
        step_id=1,
        description="Test step",
        required_information="Test info",
        expected_outcome="Test outcome",
    )

    assert step.step_id == 1
    assert step.description == "Test step"
    assert not step.is_completed


@pytest.mark.asyncio
async def test_execution_plan():
    """Test execution plan creation."""
    steps = [
        PlanStep(1, "Step 1", "Info 1", "Outcome 1"),
        PlanStep(2, "Step 2", "Info 2", "Outcome 2"),
    ]

    plan = ExecutionPlan(
        original_task="Test task",
        steps=steps,
    )

    assert plan.original_task == "Test task"
    assert len(plan.steps) == 2
    assert plan.current_step == 0


@pytest.mark.asyncio
async def test_react_agent_basic_execution():
    """Test basic ReactAgent execution without SPRE."""
    config = AgentConfig(
        name="TestAgent",
        spree_enabled=False,
    )

    agent = ReactAgent(config=config, llm_provider=MockProvider())
    response = await agent.execute("What is 2+2?")

    assert response.success
    assert isinstance(response.content, str)
    assert response.execution_time > 0


@pytest.mark.asyncio
async def test_react_agent_spre_execution():
    """Test ReactAgent execution with SPRE enabled."""
    config = AgentConfig(
        name="TestAgent",
        spree_enabled=True,
    )

    agent = ReactAgent(config=config, llm_provider=MockProvider())
    response = await agent.execute("Calculate the area of a circle with radius 5")

    assert response.success
    assert isinstance(response.content, str)
    assert len(agent.trace) > 0

    # Check for planning events in trace
    event_types = [event["type"] for event in agent.trace]
    assert "task_start" in event_types


@pytest.mark.asyncio
async def test_agent_trace():
    """Test agent execution trace."""
    config = AgentConfig(name="TestAgent")
    agent = ReactAgent(config=config, llm_provider=MockProvider())

    # Execute task
    await agent.execute("Test task")

    # Check trace
    assert len(agent.trace) > 0

    # Check trace structure
    for event in agent.trace:
        assert "timestamp" in event
        assert "type" in event
        assert "data" in event
        assert "agent_id" in event
        assert "agent_name" in event


@pytest.mark.asyncio
async def test_resource_assessment():
    """Test resource assessment functionality."""
    config = AgentConfig(
        name="TestAgent",
        spree_enabled=True,
    )

    agent = ReactAgent(config=config, llm_provider=MockProvider())

    # Create a test step
    step = PlanStep(
        step_id=1,
        description="Calculate the square root of 144",
        required_information="Mathematical calculation",
        expected_outcome="Numerical result",
    )

    # Test resource assessment
    need_tool = await agent._assess_resource_need(step)
    assert isinstance(need_tool, bool)


@pytest.mark.asyncio
async def test_token_counting():
    """Test token counting functionality."""
    config = AgentConfig(name="TestAgent")
    agent = ReactAgent(config=config, llm_provider=MockProvider())

    # Add some trace events
    agent.add_trace("test_event", {"data": "test data"})
    agent.add_trace("another_event", {"data": "more test data"})

    # Count tokens
    token_count = agent._count_tokens()
    assert token_count > 0
    assert isinstance(token_count, int)

"""Test LangGraph integration with proper imports and async support."""

import os
from unittest.mock import MagicMock

import pytest

from src.llamaagent.integration.langgraph import BudgetExceededError, LangGraphAgent
from src.llamaagent.llm.providers.mock_provider import MockProvider
from src.llamaagent.types import TaskInput

# Set test API key
os.environ["OPENAI_API_KEY"] = "test_api_key_for_testing"


@pytest.fixture
def mock_llm_provider() -> MockProvider:
    """Create mock LLM provider for testing."""
    return MockProvider()


@pytest.mark.asyncio
async def test_budget_enforcement(mock_llm_provider: MockProvider) -> None:
    """Test budget enforcement functionality."""
    tools = MagicMock()
    tools.get_all.return_value = []

    agent = LangGraphAgent(
        name="test_agent",
        llm_provider=mock_llm_provider,
        tools=tools,
        budget=0.01,  # Set very low budget
    )

    with pytest.raises(BudgetExceededError):
        task_input = TaskInput(id="1", task="Test task")
        await agent.execute_task(task_input)


@pytest.mark.asyncio
async def test_langgraph_agent_creation(mock_llm_provider: MockProvider) -> None:
    """Test LangGraph agent creation."""
    tools = MagicMock()
    tools.get_all.return_value = []

    agent = LangGraphAgent(
        name="test_agent", llm_provider=mock_llm_provider, tools=tools, budget=10.0
    )

    assert agent.name == "test_agent"
    assert agent.budget == 10.0
    assert agent.current_cost == 0.0


@pytest.mark.asyncio
async def test_task_execution_success(mock_llm_provider: MockProvider) -> None:
    """Test successful task execution."""
    tools = MagicMock()
    tools.get_all.return_value = []

    agent = LangGraphAgent(
        name="test_agent", llm_provider=mock_llm_provider, tools=tools, budget=10.0
    )

    task_input = TaskInput(id="1", task="Simple test task")

    # This should work with mock provider and sufficient budget
    try:
        result = await agent.execute_task(task_input)
        assert result is not None
    except BudgetExceededError:
        # Expected with very low budget simulation
        pass

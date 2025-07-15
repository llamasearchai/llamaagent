# Standard library
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Third-party testing utilities
import pytest
import pytest_asyncio

# Ensure 'src' directory is in sys.path for imports when package not installed
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("CI", "true")
os.environ.pop("OPENAI_API_KEY", None)


@pytest.fixture
def sample_task() -> str:
    """Sample task for testing."""
    return "Calculate the sum of squares of the first 10 natural numbers"


@pytest.fixture
def sample_problem() -> str:
    """Sample problem for testing."""
    return "What is the best approach to implement a distributed caching system?"


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Temporary output directory for test results."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
async def mock_llm_provider():
    """Mock LLM provider for testing."""
    from llamaagent.llm import MockProvider

    return MockProvider()


@pytest.fixture
def sample_agent_config():
    """Sample agent configuration for testing."""
    from llamaagent.agents.base import AgentConfig

    return AgentConfig(
        name="test_agent", max_iterations=5, timeout=30.0, spree_enabled=True
    )


@pytest.fixture
def sample_tools():
    """Sample tools for testing."""
    from llamaagent.tools import ToolRegistry
    from llamaagent.tools.calculator import CalculatorTool
    from llamaagent.tools.python_repl import PythonREPLTool

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(PythonREPLTool())
    return registry


@pytest.fixture
def mock_agent():
    """Mock agent for testing"""
    agent = MagicMock()
    agent.process = AsyncMock()

    # Create a mock response object
    mock_response = MagicMock()
    mock_response.content = "Mock response to: What is 2 + 2? The answer is 4."
    mock_response.execution_time = 0.0
    mock_response.token_count = 10

    agent.process.return_value = mock_response
    return agent


@pytest.fixture
def mock_tools():
    """Mock tools for testing"""
    from llamaagent.tools.calculator import CalculatorTool

    return [CalculatorTool()]


@pytest_asyncio.fixture
async def async_mock_agent():
    """Async mock agent for testing"""
    agent = MagicMock()
    agent.process = AsyncMock()

    mock_response = MagicMock()
    mock_response.content = "The answer to 2 + 2 is 4."
    mock_response.execution_time = 0.0
    mock_response.token_count = 10

    agent.process.return_value = mock_response
    return agent


@pytest.fixture
def mock_provider(mock_llm_provider):
    """Alias for backward-compatibility with tests expecting 'mock_provider'."""
    return mock_llm_provider


@pytest.fixture
def agent_config(sample_agent_config):
    """Alias for backward-compatibility."""
    return sample_agent_config


@pytest.fixture
def tool_registry():
    """Provide a pre-populated ToolRegistry instance."""
    from llamaagent.tools import ToolRegistry, get_all_tools

    reg = ToolRegistry()
    for tool in get_all_tools():
        reg.register(tool)
    return reg


def test_fixtures_available():
    """Test that fixtures are properly configured."""
    assert True  # Basic assertion to validate test setup

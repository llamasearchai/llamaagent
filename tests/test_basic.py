"""Basic tests for LlamaAgent functionality."""

from typing import Any, Dict

import pytest

from llamaagent.agents import ReactAgent
from llamaagent.agents.base import AgentConfig
from llamaagent.llm import MockProvider
from llamaagent.memory import MemoryEntry, SimpleMemory
from llamaagent.tools import ToolRegistry
from llamaagent.tools.base import BaseTool, Tool
from llamaagent.tools.calculator import CalculatorTool
from llamaagent.tools.python_repl import PythonREPLTool


@pytest.mark.asyncio
async def test_agent_basic_execution():
    """Test basic agent execution."""
    config = AgentConfig(name="TestAgent")
    agent = ReactAgent(config=config)

    response = await agent.execute("Hello, how are you?")

    assert response.success
    assert isinstance(response.content, str)
    assert len(response.content) > 0
    assert response.execution_time >= 0


@pytest.mark.asyncio
async def test_agent_with_tools():
    """Test agent execution with tools."""
    tools = ToolRegistry()
    tools.register(CalculatorTool())

    config = AgentConfig(name="TestAgent")
    agent = ReactAgent(config=config, tools=tools)

    response = await agent.execute("What is 2 + 2?")

    assert response.success
    assert isinstance(response.content, str)


@pytest.mark.asyncio
async def test_agent_with_memory():
    """Test agent with memory functionality."""
    memory = SimpleMemory()
    config = AgentConfig(name="TestAgent", enable_memory=True)
    agent = ReactAgent(config=config, memory=memory)

    # First interaction
    response1 = await agent.execute("Remember that my name is Alice")
    assert response1.success

    # Second interaction
    response2 = await agent.execute("What is my name?")
    assert response2.success


def test_calculator_tool():
    """Test calculator tool functionality."""
    tool = CalculatorTool()

    # Test basic arithmetic
    result = tool.execute("2 + 2")
    assert "4" in result

    result = tool.execute("10 * 5")
    assert "50" in result

    # Test invalid expression
    result = tool.execute("invalid expression")
    assert "error" in result.lower() or "invalid" in result.lower()


def test_python_repl_tool():
    """Test Python REPL tool functionality."""
    tool = PythonREPLTool()

    # Test basic Python code
    result = tool.execute("print('Hello, World!')")
    assert "Hello, World!" in result

    # Test calculation
    result = tool.execute("result = 5 * 3\nprint(result)")
    assert "15" in result

    # Test error handling
    result = tool.execute("invalid_syntax(")
    assert "error" in result.lower() or "syntax" in result.lower()


@pytest.mark.asyncio
async def test_memory_operations():
    """Test memory storage and retrieval."""
    memory = SimpleMemory()

    # Add memories
    entry_id1 = await memory.add("Alice likes programming")
    entry_id2 = await memory.add("Bob enjoys cooking")
    entry_id3 = await memory.add("Alice works at Google")

    assert entry_id1
    assert entry_id2
    assert entry_id3

    # Search memories
    results = await memory.search("Alice")
    assert len(results) == 2
    assert all("Alice" in entry["content"] for entry in results)

    # Search with limit
    results = await memory.search("Alice", limit=1)
    assert len(results) == 1

    # Test memory count
    count = memory.count()
    assert count == 3


def test_tool_registry():
    """Test tool registry functionality."""

    # Create a concrete tool for testing
    class TestTool(BaseTool):
        @property
        def name(self) -> str:
            return "test_tool"

        @property
        def description(self) -> str:
            return "A test tool for coverage"

        def execute(self, *args, **kwargs) -> Dict[str, Any]:
            return {"result": "test executed", "args": args, "kwargs": kwargs}

    # Test registry operations
    registry = ToolRegistry()
    tool = TestTool()

    # Test registration
    registry.register(tool)
    assert "test_tool" in registry.list_names()
    assert registry.get("test_tool") == tool

    # Test deregistration
    registry.deregister("test_tool")
    assert "test_tool" not in registry.list_names()
    assert registry.get("test_tool") is None

    # Test missing tool
    assert registry.get("nonexistent") is None

    # Test deregistering non-existent tool (should not raise)
    registry.deregister("nonexistent")


def test_base_tool_abstract_methods():
    """Test that BaseTool abstract methods are properly defined."""
    # Test that we cannot instantiate BaseTool directly
    with pytest.raises(TypeError):
        BaseTool()  # type: ignore[abstract] # Should raise TypeError due to abstract methods

    # Test concrete implementation
    class ConcreteTool(BaseTool):
        @property
        def name(self) -> str:
            return "concrete_tool"

        @property
        def description(self) -> str:
            return "A concrete tool implementation"

        def execute(self, *args, **kwargs) -> Dict[str, Any]:
            return {"status": "success", "input": {"args": args, "kwargs": kwargs}}

    # Test that concrete implementation works
    tool = ConcreteTool()
    assert tool.name == "concrete_tool"
    assert tool.description == "A concrete tool implementation"

    result = tool.execute("arg1", "arg2", key="value")
    assert result["status"] == "success"
    assert result["input"]["args"] == ("arg1", "arg2")
    assert result["input"]["kwargs"] == {"key": "value"}

    # Test that abstract methods exist and are properly defined
    assert hasattr(BaseTool, "name")
    assert hasattr(BaseTool, "description")
    assert hasattr(BaseTool, "execute")

    # Verify they are abstract
    assert getattr(BaseTool.name, "__isabstractmethod__", False)
    assert getattr(BaseTool.description, "__isabstractmethod__", False)
    assert getattr(BaseTool.execute, "__isabstractmethod__", False)


def test_tool_compatibility_alias():
    """Test that Tool alias works for backward compatibility."""
    # Test that Tool is an alias for BaseTool
    assert Tool is BaseTool

    # Test that we can subclass using Tool
    class CompatibilityTool(Tool):
        @property
        def name(self) -> str:
            return "compat_tool"

        @property
        def description(self) -> str:
            return "Tool using compatibility alias"

        def execute(self, *args, **kwargs) -> Dict[str, Any]:
            return {"compatibility": True}

    tool = CompatibilityTool()
    assert tool.name == "compat_tool"
    assert tool.description == "Tool using compatibility alias"
    assert tool.execute()["compatibility"] is True


def test_tool_registry_comprehensive():
    """Test comprehensive tool registry functionality."""

    class Tool1(BaseTool):
        @property
        def name(self) -> str:
            return "tool1"

        @property
        def description(self) -> str:
            return "First tool"

        def execute(self, *args, **kwargs) -> Dict[str, Any]:
            return {"tool": "1"}

    class Tool2(BaseTool):
        @property
        def name(self) -> str:
            return "tool2"

        @property
        def description(self) -> str:
            return "Second tool"

        def execute(self, *args, **kwargs) -> Dict[str, Any]:
            return {"tool": "2"}

    registry = ToolRegistry()
    tool1 = Tool1()
    tool2 = Tool2()

    # Test multiple registrations
    registry.register(tool1)
    registry.register(tool2)

    # Test list_names maintains order
    names = registry.list_names()
    assert names == ["tool1", "tool2"]

    # Test list_tools
    tools = registry.list_tools()
    assert len(tools) == 2
    assert tools[0] == tool1
    assert tools[1] == tool2

    # Test get functionality
    assert registry.get("tool1") == tool1
    assert registry.get("tool2") == tool2

    # Test deregistration
    registry.deregister("tool1")
    assert registry.get("tool1") is None
    assert registry.get("tool2") == tool2
    assert registry.list_names() == ["tool2"]


def test_llm_provider():
    """Test LLM provider functionality."""
    provider = MockProvider()

    # This is synchronous since we're testing the mock
    import asyncio

    async def test_provider():
        from llamaagent.llm import LLMMessage

        messages = [
            LLMMessage(role="system", content="You are a helpful assistant"),
            LLMMessage(role="user", content="Hello!"),
        ]

        response = await provider.complete(messages)

        assert response.content
        assert response.tokens_used >= 0
        assert response.model

    asyncio.run(test_provider())


def test_agent_config():
    """Test agent configuration."""
    config = AgentConfig(
        name="TestAgent",
        max_iterations=5,
        spree_enabled=True,
        enable_tools=True,
    )

    assert config.name == "TestAgent"
    assert config.max_iterations == 5
    assert config.spree_enabled is True
    assert config.enable_tools is True


@pytest.mark.asyncio
async def test_agent_trace():
    """Test agent execution tracing."""
    config = AgentConfig(name="TestAgent")
    agent = ReactAgent(config=config)

    response = await agent.execute("Test task")

    assert response.success
    assert response.content is not None
    # Check basic response attributes
    assert hasattr(response, 'execution_time')
    assert hasattr(response, 'tokens_used')


def test_memory_entry():
    """Test memory entry functionality."""
    entry = MemoryEntry(
        content="Test content", tags=["test", "example"], metadata={"source": "test"}
    )

    assert entry.content == "Test content"
    assert "test" in entry.tags
    assert "example" in entry.tags
    assert entry.metadata["source"] == "test"
    assert entry.id  # Should have an ID
    assert entry.timestamp > 0  # Should have a timestamp


@pytest.mark.asyncio
async def test_agent_error_handling():
    """Test agent error handling."""
    config = AgentConfig(name="TestAgent")
    agent = ReactAgent(config=config)

    # This should not crash even with empty input
    response = await agent.execute("")

    # Response should still be valid
    assert isinstance(response.success, bool)
    assert isinstance(response.content, str)
    assert response.execution_time >= 0

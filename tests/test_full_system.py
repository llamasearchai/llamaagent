"""Full system tests"""

import time
import uuid
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, Mock

import pytest

# Import from llamaagent
from src.llamaagent.agents.base import AgentResponse, BaseAgent
from src.llamaagent.llm.providers.base_provider import BaseLLMProvider
from src.llamaagent.tools.base import BaseTool
from src.llamaagent.types import TaskInput, TaskOutput, TaskResult, TaskStatus


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM Provider for testing"""

    def __init__(self, api_key: str = "test-key", **kwargs: Any):
        self.api_key = api_key
        self.responses = ["Test response"]
        self.response_index = 0

    async def generate_response(
        self,
        messages: List[Any],
        model: str = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Any:  # type: ignore[override]
        """Implement abstract method"""
        response = Mock()
        response.content = self.responses[self.response_index % len(self.responses)]
        self.response_index += 1
        return response

    async def acomplete(self, messages: List[Any], **kwargs: Any) -> Any:  # type: ignore[override]
        """Implement abstract method"""
        return await self.generate_response(messages, **kwargs)

    async def generate_streaming_response(
        self,
        messages: List[Any],
        model: str = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs: Any,
    ):  # type: ignore[override]
        """Implement abstract method for streaming"""
        for chunk in self.responses[self.response_index % len(self.responses)]:
            yield chunk

    def complete(self, messages: List[Any], **kwargs: Any) -> Any:  # type: ignore[override]
        """Sync completion - matches base class signature"""
        response = Mock()
        response.content = self.responses[self.response_index % len(self.responses)]
        self.response_index += 1
        return response

    async def agenerate_streaming_response(self, messages: List[Any], **kwargs: Any):  # type: ignore[override]
        """Implement abstract method for streaming"""
        async for chunk in self.generate_streaming_response(messages, **kwargs):
            yield chunk


class MockTool(BaseTool):
    """Mock tool for testing"""

    def __init__(self, name: str = "mock_tool") -> None:
        self._name = name
        self.call_count = 0

    # ------------------------------------------------------------------
    # BaseTool interface
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:  # type: ignore[override]
        return self._name

    @property
    def description(self) -> str:  # type: ignore[override]
        return "A mock tool"

    def execute(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # type: ignore[override]
        self.call_count += 1
        return {
            "result": f"Mock tool result {self.call_count}",
            "args": args,
            "kwargs": kwargs,
        }


class MockAgent(BaseAgent):
    """Mock agent for testing"""

    def __init__(self, agent_id: str = "mock_agent", **kwargs: Any):
        # Import AgentConfig here to avoid circular imports
        from src.llamaagent.agents.base import AgentConfig

        # Extract name to avoid duplicate argument
        name = kwargs.pop("name", "MockAgent")

        # Create config for base agent
        config = AgentConfig(name=name, description="A mock agent", **kwargs)

        # Initialize base agent with config
        super().__init__(config=config)

        # Store additional attributes
        self._name = name  # Use the name we extracted above
        self._llm_provider = kwargs.get("llm_provider", MockLLMProvider())

    async def execute_task(self, task_input: TaskInput) -> TaskOutput:
        """Execute a mock task"""
        # Use the stored llm_provider
        response = await self._llm_provider.acomplete(
            messages=[{"role": "user", "content": task_input.task}]
        )

        # Create TaskResult first
        task_result = TaskResult(
            success=True,
            data={"response": response.content},
            metadata={"agent": self.name},
        )

        return TaskOutput(
            task_id=task_input.id, status=TaskStatus.COMPLETED, result=task_result
        )

    async def execute(
        self, task: str, context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Execute a task and return response - implements abstract method."""
        # Convert to TaskInput format
        task_input = TaskInput(id=str(uuid.uuid4()), task=task, context=context or {})
        output = await self.execute_task(task_input)
        
        # Convert to AgentResponse
        return AgentResponse(
            content=output.result.data.get("response", "") if output.result else "",
            metadata=output.result.metadata if output.result else {},
            agent_name=self.name,
            timestamp=time.time()
        )

    async def arun(self, task_input: TaskInput) -> TaskOutput:
        """Async run method"""
        return await self.execute_task(task_input)

    def run(self, task_input: TaskInput) -> TaskOutput:
        """Sync run method"""
        # Use sync complete method
        response = self._llm_provider.complete(
            messages=[{"role": "user", "content": task_input.task}]
        )

        # Create TaskResult first
        task_result = TaskResult(
            success=True,
            data={"response": response.content},
            metadata={"agent": self.name},
        )

        return TaskOutput(
            task_id=task_input.id, status=TaskStatus.COMPLETED, result=task_result
        )


@pytest.fixture
def mock_llm_provider() -> MockLLMProvider:
    """Create a mock LLM provider"""
    return MockLLMProvider(api_key="test-key")


@pytest.fixture
def mock_agent(mock_llm_provider: MockLLMProvider) -> MockAgent:
    """Create a mock agent"""
    return MockAgent(
        agent_id="test_agent", name="TestAgent", llm_provider=mock_llm_provider
    )


@pytest.fixture
def mock_tool() -> MockTool:
    """Create a mock tool"""
    return MockTool(name="test_tool")


class TestFullSystem:
    """Test the full system integration"""

    @pytest.mark.asyncio
    async def test_agent_execution(self, mock_agent: MockAgent):
        """Test basic agent execution"""
        task_input = TaskInput(id="1", task="Test task")
        result = await mock_agent.execute_task(task_input)

        assert result.result and result.result.success is True  # type: ignore[bool-expr]
        assert result.result and result.result.data["response"] == "Test response"  # type: ignore[index]
        assert result.result and result.result.metadata["agent"] == "TestAgent"  # type: ignore[index]

    @pytest.mark.asyncio
    async def test_tool_integration(self, mock_agent: MockAgent, mock_tool: MockTool):
        """Test tool integration"""
        # Register tool with the agent's tool registry
        if hasattr(mock_agent, "tools"):
            mock_agent.tools.register(mock_tool)

        # Execute task
        task_input = TaskInput(id="2", task="Use the tool")
        result = await mock_agent.execute_task(task_input)

        assert result.success is True

    def test_sync_execution(self, mock_agent: MockAgent):
        """Test synchronous execution"""
        task_input = TaskInput(id="3", task="Sync test")
        result = mock_agent.run(task_input)

        # The result attribute is guaranteed in this test scenario.
        assert result.result and result.result.success is True  # type: ignore[bool-expr]
        assert result.result and result.result.data["response"] == "Test response"  # type: ignore[index]
        assert result.result and result.result.metadata["agent"] == "TestAgent"  # type: ignore[index]

    @pytest.mark.asyncio
    async def test_error_handling(self, mock_llm_provider: MockLLMProvider):
        """Test error handling"""
        # Make the provider raise an error
        mock_llm_provider.acomplete = AsyncMock(side_effect=Exception("Test error"))

        agent = MockAgent(agent_id="error_agent", llm_provider=mock_llm_provider)

        task_input = TaskInput(id="4", task="This should fail")

        # The agent should handle the error gracefully
        with pytest.raises(Exception):
            await agent.execute_task(task_input)

    def test_agent_registry(self):
        """Test agent registry functionality"""
        agents: List[MockAgent] = []

        # Create multiple agents
        for i in range(3):
            agent = MockAgent(agent_id=f"agent_{i}")
            agents.append(agent)

        # Verify all agents were created
        assert len(agents) == 3

        # Get first agent's result
        result = agents[0].run(TaskInput(id="5", task="Test"))
        assert result.success is True
        assert result.result and result.result.data["response"] == "Test response"  # type: ignore[index]

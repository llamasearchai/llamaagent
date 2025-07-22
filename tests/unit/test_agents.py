"""
Unit tests for agents.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import uuid

from src.llamaagent.agents.base import AgentConfig
from src.llamaagent.agents.react import ReactAgent
from src.llamaagent.llm.providers.mock_provider import MockProvider
from src.llamaagent.types import TaskInput


class TestAgents:
    """Test suite for agents."""

    def test_react_agent_initialization(self):
        """Test ReactAgent can be initialized."""
        config = AgentConfig(name="TestAgent", metadata={"spree_enabled": False})
        provider = MockProvider(model_name="test-model")
        agent = ReactAgent(config=config, llm_provider=provider)
        assert agent is not None
        assert agent.config.name == "TestAgent"

    def test_react_agent_processes_task(self):
        """Test ReactAgent can process tasks."""
        config = AgentConfig(name="TestAgent", metadata={"spree_enabled": False})
        provider = MockProvider(model_name="test-model")
        agent = ReactAgent(config=config, llm_provider=provider)

        task_input = TaskInput(id=str(uuid.uuid4()), task="Test task")
        response = agent.run(task_input)
        assert response is not None
        assert response.success is not None
        assert response.task_id == task_input.id

    def test_agent_error_handling(self):
        """Test agent error handling."""
        config = AgentConfig(name="TestAgent", metadata={"spree_enabled": False})
        provider = MockProvider(model_name="test-model")
        agent = ReactAgent(config=config, llm_provider=provider)

        # Test with invalid input
        task_input = TaskInput(id=str(uuid.uuid4()), task="")
        response = agent.run(task_input)
        assert response is not None
        # Should handle empty input gracefully

    def test_agent_with_spree_mode(self):
        """Test agent with SPREE mode enabled."""
        config = AgentConfig(name="TestAgent", metadata={"spree_enabled": True})
        provider = MockProvider(model_name="test-model")
        agent = ReactAgent(config=config, llm_provider=provider)

        task_input = TaskInput(
            id=str(uuid.uuid4()), task="Complex task requiring planning"
        )
        response = agent.run(task_input)
        assert response is not None
        assert response.success is not None

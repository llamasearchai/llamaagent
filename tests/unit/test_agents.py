"""
Unit tests for agents module.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import pytest
from unittest.mock import Mock

from llamaagent.agents.base import AgentConfig, AgentRole
from llamaagent.agents.react import ReactAgent
from llamaagent.llm.providers.mock_provider import MockProvider
from llamaagent.types import TaskInput


class TestAgents:
    """Test suite for agents."""

    def test_react_agent_initialization(self):
        """Test ReactAgent can be initialized."""
        config = AgentConfig(name="TestAgent", spree_enabled=False)
        provider = MockProvider(model_name="test-model")
        agent = ReactAgent(config=config, llm_provider=provider)
        assert agent is not None
        assert agent.config.name == "TestAgent"

    def test_react_agent_basic_execution(self):
        """Test ReactAgent can execute basic tasks."""
        config = AgentConfig(name="TestAgent")
        provider = MockProvider(model_name="test-model")
        agent = ReactAgent(config=config, llm_provider=provider)
        
        task_input = TaskInput(task="Hello, world!")
        # Use execute method instead of run
        result = agent.execute("Hello, world!")
        assert result is not None

    def test_agent_config_validation(self):
        """Test AgentConfig validation."""
        config = AgentConfig(name="TestAgent", max_iterations=5)
        assert config.name == "TestAgent"
        assert config.max_iterations == 5
        assert config.role == AgentRole.GENERALIST

    def test_react_agent_with_tools(self):
        """Test ReactAgent with tools enabled."""
        config = AgentConfig(name="TestAgent", enable_tools=True)
        provider = MockProvider(model_name="test-model")
        agent = ReactAgent(config=config, llm_provider=provider)
        
        task_input = TaskInput(task="Calculate 2+2")
        # Use execute method
        result = agent.execute("Calculate 2+2")
        assert result is not None

    def test_react_agent_error_handling(self):
        """Test ReactAgent error handling."""
        config = AgentConfig(name="TestAgent")
        provider = MockProvider(model_name="test-model")
        agent = ReactAgent(config=config, llm_provider=provider)
        
        # Test with empty task
        result = agent.execute("")
        assert result is not None

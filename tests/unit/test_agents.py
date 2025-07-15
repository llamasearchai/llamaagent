"""
Unit tests for agents.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import pytest
from unittest.mock import Mock
from src.llamaagent.agents.react import ReactAgent
from src.llamaagent.types import AgentConfig
from src.llamaagent.llm.providers.mock_provider import MockProvider


class TestAgents:
    """Test suite for agents."""
    
    def test_react_agent_initialization(self):
        """Test ReactAgent can be initialized."""
        config = AgentConfig(
            agent_name="TestAgent",
            metadata={"spree_enabled": False}
        )
        provider = MockProvider(model_name="test-model")
        agent = ReactAgent(config=config, llm_provider=provider)
        assert agent is not None
        assert agent.config.agent_name == "TestAgent"
    
    def test_react_agent_processes_task(self):
        """Test ReactAgent can process tasks."""
        config = AgentConfig(
            agent_name="TestAgent",
            metadata={"spree_enabled": False}
        )
        provider = MockProvider(model_name="test-model")
        agent = ReactAgent(config=config, llm_provider=provider)
        
        response = agent.process_task("Test task")
        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0
    
    def test_agent_error_handling(self):
        """Test agent error handling."""
        config = AgentConfig(
            agent_name="TestAgent",
            metadata={"spree_enabled": False}
        )
        provider = MockProvider(model_name="test-model")
        agent = ReactAgent(config=config, llm_provider=provider)
        
        # Test with invalid input
        response = agent.process_task("")
        assert response is not None
        # Should handle empty input gracefully
    
    def test_agent_with_spree_mode(self):
        """Test agent with SPREE mode enabled."""
        config = AgentConfig(
            agent_name="TestAgent",
            metadata={"spree_enabled": True}
        )
        provider = MockProvider(model_name="test-model")
        agent = ReactAgent(config=config, llm_provider=provider)
        
        response = agent.process_task("Complex task requiring planning")
        assert response is not None
        assert response.content is not None

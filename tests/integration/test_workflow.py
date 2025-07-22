"""
Integration tests for complete workflows.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import pytest

from src.llamaagent.agents.react import ReactAgent
from src.llamaagent.llm.factory import LLMFactory
from src.llamaagent.llm.providers.mock_provider import MockProvider
from src.llamaagent.types import AgentConfig


class TestWorkflows:
    """Test suite for complete workflows."""

    def test_simple_workflow(self):
        """Test simple agent workflow."""
        # Create agent
        config = AgentConfig(
            agent_name="WorkflowAgent", metadata={"spree_enabled": False}
        )
        provider = MockProvider(model_name="test-model")
        agent = ReactAgent(config=config, llm_provider=provider)

        # Process multiple tasks
        tasks = [
            "Calculate 2 + 2",
            "What is the capital of France?",
            "Explain photosynthesis",
        ]

        responses = []
        for task in tasks:
            response = agent.process_task(task)
            responses.append(response)
            assert response is not None
            assert response.content is not None

        assert len(responses) == 3

    def test_spree_workflow(self):
        """Test SPREE workflow."""
        config = AgentConfig(agent_name="SpreeAgent", metadata={"spree_enabled": True})
        provider = MockProvider(model_name="test-model")
        agent = ReactAgent(config=config, llm_provider=provider)

        # Process complex task
        complex_task = "Plan a marketing campaign for a new product"
        response = agent.process_task(complex_task)

        assert response is not None
        assert response.content is not None
        # SPREE mode should provide more detailed response
        assert len(response.content) > 50

    def test_error_recovery_workflow(self):
        """Test workflow error recovery."""
        config = AgentConfig(
            agent_name="ErrorTestAgent", metadata={"spree_enabled": False}
        )
        provider = MockProvider(model_name="test-model")
        agent = ReactAgent(config=config, llm_provider=provider)

        # Test with various edge cases
        edge_cases = ["", None, "A" * 10000]  # Empty, None, very long

        for case in edge_cases:
            try:
                response = agent.process_task(case)
                # Should handle gracefully
                assert response is not None
            except Exception as e:
                # Should not crash completely
                assert "Error" in str(e)

"""
Performance tests for system benchmarking.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.llamaagent.agents.react import ReactAgent
from src.llamaagent.llm.providers.mock_provider import MockProvider
from src.llamaagent.types import AgentConfig


class TestPerformance:
    """Performance test suite."""

    def test_single_agent_performance(self):
        """Test single agent performance."""
        config = AgentConfig(agent_name="PerfAgent", metadata={"spree_enabled": False})
        provider = MockProvider(model_name="test-model")
        agent = ReactAgent(config=config, llm_provider=provider)

        # Measure response time
        start_time = time.time()
        response = agent.process_task("Simple task")
        end_time = time.time()

        assert response is not None
        assert (end_time - start_time) < 1.0  # Should complete within 1 second

    def test_concurrent_agents(self):
        """Test concurrent agent performance."""

        def create_and_run_agent(agent_id):
            config = AgentConfig(
                agent_name=f"ConcurrentAgent{agent_id}",
                metadata={"spree_enabled": False},
            )
            provider = MockProvider(model_name="test-model")
            agent = ReactAgent(config=config, llm_provider=provider)

            return agent.process_task(f"Task {agent_id}")

        # Run multiple agents concurrently
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_and_run_agent, i) for i in range(10)]
            results = [future.result() for future in futures]
        end_time = time.time()

        # All should complete successfully
        assert len(results) == 10
        assert all(result is not None for result in results)

        # Should complete within reasonable time
        assert (end_time - start_time) < 10.0

    def test_memory_usage(self):
        """Test memory usage doesn't grow excessively."""
        config = AgentConfig(
            agent_name="MemoryAgent", metadata={"spree_enabled": False}
        )
        provider = MockProvider(model_name="test-model")
        agent = ReactAgent(config=config, llm_provider=provider)

        # Process many tasks
        for i in range(100):
            response = agent.process_task(f"Task {i}")
            assert response is not None

        # Memory should be reasonable (this is a basic check)
        import psutil

        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        assert memory_mb < 500  # Should use less than 500MB

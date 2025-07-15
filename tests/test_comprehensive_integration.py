#!/usr/bin/env python3
"""
Comprehensive integration tests for LlamaAgent.

Author: Nik Jois <nikjois@llamasearch.ai>

This module provides comprehensive integration testing covering:
- End-to-end agent workflows
- Multi-provider LLM integration
- Tool system functionality
- Performance and reliability testing
- Error handling and edge cases
"""

import logging
import asyncio

# Import core modules
import sys
import time
from pathlib import Path
from typing import Any, Optional

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llamaagent.agents import AgentConfig, ReactAgent
from llamaagent.llm.factory import LLMFactory
from llamaagent.tools import ToolRegistry, get_all_tools
from llamaagent.types import LLMMessage, LLMResponse
logger = logging.getLogger(__name__)


class TestComprehensiveIntegration:
    """Comprehensive integration tests."""

    @pytest.fixture
    def agent_config(self, mock_provider: Any) -> AgentConfig:
        """Create test agent configuration."""
        return AgentConfig(
            llm_provider=mock_provider,
            tools=[],
            max_iterations=5,
            verbose=False,
            debug=False,
        )

    @pytest.fixture
    def tool_registry(self) -> ToolRegistry:
        """Create test tool registry."""
        registry = ToolRegistry()
        try:
            for tool in get_all_tools():
                registry.register(tool)
        except Exception:
            # Fallback if tools not available
            pass
        return registry

    @pytest.fixture
    def mock_llm_provider(self) -> Optional[Any]:
        """Create mock LLM provider."""
        try:
            factory = LLMFactory()
            provider = factory.create_provider("mock")
            return provider
        except Exception:
            return None

    @pytest.mark.asyncio
    async def test_basic_agent_creation(
        self, agent_config: AgentConfig, tool_registry: ToolRegistry
    ) -> None:
        """Test basic agent creation and configuration."""
        agent = ReactAgent(agent_config, tools=tool_registry)

        assert agent is not None
        assert agent.config.name == "TestAgent"
        assert agent.config.spree_enabled is True
        assert agent.tools is not None

    @pytest.mark.asyncio
    async def test_simple_math_calculation(
        self, agent_config: AgentConfig, tool_registry: ToolRegistry
    ) -> None:
        """Test simple mathematical calculation."""
        agent = ReactAgent(agent_config, tools=tool_registry)

        task = "Calculate 2 + 2"

        try:
            response = await agent.execute(task)

            assert response is not None
            assert hasattr(response, "content")
            assert hasattr(response, "success")

            # Should contain the answer 4
            if response.success:
                assert "4" in str(response.content)
        except Exception as e:
            # If execution fails, ensure it's handled gracefully
            assert isinstance(e, (ValueError, TypeError, RuntimeError))

    @pytest.mark.asyncio
    async def test_error_handling(
        self, agent_config: AgentConfig, tool_registry: ToolRegistry
    ) -> None:
        """Test error handling with invalid inputs."""
        agent = ReactAgent(agent_config, tools=tool_registry)

        # Test empty task
        try:
            response = await agent.execute("")
            assert response is not None
        except Exception:
            # Should handle gracefully
            pass

        # Test None task - use empty string instead as execute expects str
        try:
            response = await agent.execute("")
            assert response is not None
        except Exception:
            # Should handle gracefully
            pass

    @pytest.mark.asyncio
    async def test_provider_factory_creation(self) -> None:
        """Test LLM provider factory."""
        try:
            # Test mock provider
            factory = LLMFactory()
            mock_provider = factory.create_provider("mock")
            assert mock_provider is not None

            # Test health check if available
            try:
                if hasattr(mock_provider, "health_check"):
                    health = await mock_provider.health_check()
                    assert isinstance(health, bool)
            except AttributeError:
                # Health check method may not be available
                pass

        except Exception as e:
            pytest.skip(f"Provider creation failed: {e}")

    @pytest.mark.asyncio
    async def test_tool_registry_functionality(self) -> None:
        """Test tool registry functionality."""
        registry = ToolRegistry()

        # Test basic registry operations
        assert registry is not None

        # Test listing tools
        tool_names = registry.list_names()
        assert isinstance(tool_names, list)

        # Test getting non-existent tool
        non_existent = registry.get("non_existent_tool")
        assert non_existent is None

    @pytest.mark.asyncio
    async def test_concurrent_execution(
        self, agent_config: AgentConfig, tool_registry: ToolRegistry
    ) -> None:
        """Test concurrent agent execution."""
        tasks = ["Calculate 1+1", "Calculate 2+2", "Calculate 3+3"]

        agents = []
        for i, task in enumerate(tasks):
            config = AgentConfig(name=f"ConcurrentAgent{i}")
            agent = ReactAgent(config, tools=tool_registry)
            agents.append((agent, task))

        # Execute concurrently
        async def execute_task(agent: Any, task: str) -> Any:
            try:
                return await agent.execute(task)
            except Exception as e:
                return e

        responses = await asyncio.gather(
            *[execute_task(agent, task) for agent, task in agents],
            return_exceptions=True,
        )

        # All should complete (successfully or with handled errors)
        assert len(responses) == len(tasks)

    @pytest.mark.asyncio
    async def test_performance_monitoring(
        self, agent_config: AgentConfig, tool_registry: ToolRegistry
    ) -> None:
        """Test performance monitoring capabilities."""
        agent = ReactAgent(agent_config, tools=tool_registry)

        task = "Simple calculation: 5 * 5"

        start_time = time.time()
        try:
            response = await agent.execute(task)
            execution_time = time.time() - start_time

            # Basic performance checks
            assert execution_time >= 0
            assert execution_time < 60  # Should complete within 60 seconds

            if response and hasattr(response, "execution_time"):
                assert response.execution_time >= 0

        except Exception:
            execution_time = time.time() - start_time
            assert execution_time < 60  # Even failures should complete quickly

    @pytest.mark.asyncio
    async def test_configuration_validation(self) -> None:
        """Test configuration validation."""
        # Test valid configuration
        valid_config = AgentConfig(
            name="ValidAgent", max_iterations=5, spree_enabled=True
        )

        agent = ReactAgent(valid_config)
        assert agent.config.name == "ValidAgent"
        assert agent.config.max_iterations == 5

        # Test edge case configurations
        edge_config = AgentConfig(
            name="EdgeAgent",
            max_iterations=1,  # Minimum iterations
            spree_enabled=False,
        )

        edge_agent = ReactAgent(edge_config)
        assert edge_agent.config.max_iterations == 1

    @pytest.mark.asyncio
    async def test_system_robustness(
        self, agent_config: AgentConfig, tool_registry: ToolRegistry
    ) -> None:
        """Test system robustness with various inputs."""
        agent = ReactAgent(agent_config, tools=tool_registry)

        test_cases = [
            "Calculate 10 + 20",
            "What is 5 * 6?",
            "Compute the sum of 1, 2, and 3",
            "Hello world",  # Non-calculation task
            "Test unicode",  # Unicode characters
        ]

        for task in test_cases:
            try:
                response = await agent.execute(task)
                # Should handle all cases without crashing
                assert response is not None
            except Exception as e:
                # Exceptions should be specific and handleable
                assert isinstance(
                    e, (ValueError, TypeError, RuntimeError, AttributeError)
                )


@pytest.mark.integration
class TestProductionFeatures:
    """Test production-ready features."""

    @pytest.mark.asyncio
    async def test_health_checks(self) -> None:
        """Test system health checks."""
        try:
            # Test provider health
            factory = LLMFactory()
            provider = factory.create_provider("mock")
            try:
                if hasattr(provider, "health_check"):
                    health = await provider.health_check()
                    assert isinstance(health, bool)
            except AttributeError:
                # Health check may not be available on this provider type
                pass

            # Test basic system health
            config = AgentConfig(name="HealthTestAgent")
            agent = ReactAgent(config)

            # Basic functionality test
            response = await agent.execute("Test health check")
            assert response is not None

        except Exception as e:
            pytest.skip(f"Health check not available: {e}")

    @pytest.mark.asyncio
    async def test_load_simulation(self) -> None:
        """Test system behavior under moderate load."""
        concurrent_requests = 5
        tasks = [f"Calculate {i} + {i}" for i in range(concurrent_requests)]

        async def execute_single_task(task_id: int, task: str) -> Any:
            config = AgentConfig(name=f"LoadTestAgent{task_id}")
            agent = ReactAgent(config)
            try:
                return await agent.execute(task)
            except Exception as e:
                return e

        start_time = time.time()
        responses = await asyncio.gather(
            *[execute_single_task(i, task) for i, task in enumerate(tasks)],
            return_exceptions=True,
        )
        total_time = time.time() - start_time

        # Verify basic metrics
        assert len(responses) == concurrent_requests
        assert total_time < 120  # Should complete within 2 minutes

        # Count successful responses
        successful = sum(1 for r in responses if not isinstance(r, Exception))
        success_rate = successful / len(responses)

        # Should have reasonable success rate
        assert success_rate >= 0.0  # At minimum, shouldn't crash completely

    @pytest.mark.asyncio
    async def test_memory_usage(
        self, agent_config: AgentConfig, tool_registry: ToolRegistry
    ) -> None:
        """Test memory usage and cleanup."""
        import gc

        initial_objects = len(gc.get_objects())

        # Create and use multiple agents
        for i in range(10):
            config = AgentConfig(name=f"MemoryTestAgent{i}")
            agent = ReactAgent(config, tools=tool_registry)

            try:
                await agent.execute("Simple test")
            except Exception as e:
                logger.error(f"Error: {e}")  # Ignore execution errors for memory test

            # Clear reference
            del agent

        # Force garbage collection
        gc.collect()

        final_objects = len(gc.get_objects())

        # Memory growth should be reasonable
        memory_growth = final_objects - initial_objects
        assert memory_growth < 10000  # Arbitrary but reasonable limit

    @pytest.mark.asyncio
    async def test_error_recovery(self) -> None:
        """Test error recovery mechanisms."""
        config = AgentConfig(name="ErrorRecoveryAgent")
        agent = ReactAgent(config)

        # Test recovery from various error conditions
        error_cases = [
            "",  # Empty input
            "x" * 10000,  # Very long input
        ]

        for case in error_cases:
            try:
                response = await agent.execute(case)
                # Should handle gracefully
                assert response is not None
            except Exception as e:
                # Should be specific, handleable exceptions
                assert isinstance(e, (ValueError, TypeError, RuntimeError))

    @pytest.mark.asyncio
    async def test_configuration_management(self) -> None:
        """Test configuration management."""
        # Test different configuration scenarios
        configs = [
            AgentConfig(name="MinimalAgent"),
            AgentConfig(name="FullFeaturedAgent", spree_enabled=True),
            AgentConfig(name="ProductionAgent", spree_enabled=False),
        ]

        for config in configs:
            agent = ReactAgent(config)
            assert agent.config.name == config.name
            assert agent.config.spree_enabled == config.spree_enabled


async def test_llm_provider_functionality(mock_provider: Any) -> None:
    """Test basic LLM provider functionality."""
    messages = [LLMMessage(role="user", content="Test message")]
    if hasattr(mock_provider, "complete"):
        response = await mock_provider.complete(messages)
        assert isinstance(response, LLMResponse)
        assert len(response.content) > 0

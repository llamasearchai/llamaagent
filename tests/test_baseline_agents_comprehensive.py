#!/usr/bin/env python3
"""Comprehensive tests for llamaagent.benchmarks.baseline_agents module."""

from unittest.mock import Mock, patch

import pytest

from llamaagent.agents.base import AgentConfig, AgentRole
from llamaagent.benchmarks.baseline_agents import (
    BaselineAgentFactory,
    BaselineType,
    PreActOnlyAgent,
    SEMOnlyAgent,
    VanillaReactAgent,
)


class TestBaselineAgentFactory:
    """Test BaselineAgentFactory functionality."""

    def test_get_baseline_description_unknown_type(self):
        """Test get_baseline_description with unknown baseline type."""
        result = BaselineAgentFactory.get_baseline_description("unknown_baseline")
        assert result == "Unknown baseline type"

    def test_get_baseline_description_known_types(self):
        """Test get_baseline_description with all known types."""
        descriptions = {
            BaselineType.VANILLA_REACT: "Standard ReAct agent without planning or resource assessment",
            BaselineType.PREACT_ONLY: "Agent with planning but executes tools for every step",
            BaselineType.SEM_ONLY: "Reactive agent with resource assessment but no strategic planning",
            BaselineType.SPRE_FULL: "Full SPRE implementation with planning and resource assessment",
        }

        for baseline_type, expected in descriptions.items():
            result = BaselineAgentFactory.get_baseline_description(baseline_type)
            assert result == expected

    def test_create_agent_unknown_type(self):
        """Test create_agent raises ValueError for unknown baseline type."""
        with pytest.raises(ValueError, match="Unknown baseline type: invalid_type"):
            BaselineAgentFactory.create_agent("invalid_type")

    def test_create_vanilla_react_agent(self):
        """Test creating vanilla ReAct agent."""
        with patch(
            "llamaagent.benchmarks.baseline_agents.ToolRegistry"
        ) as mock_registry:
            with patch(
                "llamaagent.benchmarks.baseline_agents.get_all_tools", return_value=[]
            ):
                mock_registry_instance = Mock()
                mock_registry.return_value = mock_registry_instance

                agent = BaselineAgentFactory.create_agent(BaselineType.VANILLA_REACT)

                assert isinstance(agent, VanillaReactAgent)
                assert "Vanilla" in agent.config.name
                assert not agent.config.spree_enabled

    def test_create_preact_only_agent(self):
        """Test creating PreAct only agent."""
        with patch(
            "llamaagent.benchmarks.baseline_agents.ToolRegistry"
        ) as mock_registry:
            with patch(
                "llamaagent.benchmarks.baseline_agents.get_all_tools", return_value=[]
            ):
                mock_registry_instance = Mock()
                mock_registry.return_value = mock_registry_instance

                agent = BaselineAgentFactory.create_agent(BaselineType.PREACT_ONLY)

                assert isinstance(agent, PreActOnlyAgent)
                assert "PreAct" in agent.config.name
                assert agent.config.spree_enabled

    def test_create_sem_only_agent(self):
        """Test creating SEM only agent."""
        with patch(
            "llamaagent.benchmarks.baseline_agents.ToolRegistry"
        ) as mock_registry:
            with patch(
                "llamaagent.benchmarks.baseline_agents.get_all_tools", return_value=[]
            ):
                mock_registry_instance = Mock()
                mock_registry.return_value = mock_registry_instance

                agent = BaselineAgentFactory.create_agent(BaselineType.SEM_ONLY)

                assert isinstance(agent, SEMOnlyAgent)
                assert "SEM" in agent.config.name
                assert agent.config.spree_enabled

    def test_create_spre_full_agent(self):
        """Test creating full SPRE agent."""
        with patch(
            "llamaagent.benchmarks.baseline_agents.ToolRegistry"
        ) as mock_registry:
            with patch(
                "llamaagent.benchmarks.baseline_agents.get_all_tools", return_value=[]
            ):
                mock_registry_instance = Mock()
                mock_registry.return_value = mock_registry_instance

                agent = BaselineAgentFactory.create_agent(BaselineType.SPRE_FULL)

                # Should be a regular ReactAgent for SPRE_FULL
                assert agent.__class__.__name__ == "ReactAgent"
                assert "SPRE" in agent.config.name
                assert agent.config.spree_enabled


class TestVanillaReactAgent:
    """Test VanillaReactAgent specific functionality."""

    @pytest.mark.asyncio
    async def test_execute_overrides_spree_setting(self):
        """Test that execute method overrides SPREE setting."""
        config = AgentConfig(name="Test", role=AgentRole.GENERALIST, spree_enabled=True)

        with patch("llamaagent.benchmarks.baseline_agents.ToolRegistry"):
            with patch(
                "llamaagent.benchmarks.baseline_agents.get_all_tools", return_value=[]
            ):
                agent = VanillaReactAgent(config)

                # Mock the parent execute method
                with patch.object(
                    VanillaReactAgent.__bases__[0], "execute"
                ) as mock_super_execute:
                    with patch.object(agent, "add_trace") as mock_add_trace:
                        from llamaagent.agents.base import AgentResponse

                        mock_response = AgentResponse(
                            content="test", success=True, tokens_used=0, trace=[]
                        )
                        mock_super_execute.return_value = mock_response

                        await agent.execute("test task")

                        # Should have temporarily disabled SPREE
                        mock_super_execute.assert_called_once()
                        mock_add_trace.assert_called_once_with(
                            "baseline_type", {"type": BaselineType.VANILLA_REACT}
                        )

                        # SPREE setting should be restored
                        assert agent.config.spree_enabled


class TestPreActOnlyAgent:
    """Test PreActOnlyAgent specific functionality."""

    @pytest.mark.asyncio
    async def test_assess_resource_need_always_true(self):
        """Test that _assess_resource_need always returns True."""
        config = AgentConfig(name="Test", role=AgentRole.PLANNER, spree_enabled=True)

        with patch("llamaagent.benchmarks.baseline_agents.ToolRegistry"):
            with patch(
                "llamaagent.benchmarks.baseline_agents.get_all_tools", return_value=[]
            ):
                agent = PreActOnlyAgent(config)

                from llamaagent.agents.base import PlanStep

                step = PlanStep(
                    step_id=1,
                    description="test",
                    required_information="info",
                    expected_outcome="result",
                )

                with patch.object(agent, "add_trace") as mock_add_trace:
                    result = await agent._assess_resource_need(step)

                    assert result
                    mock_add_trace.assert_called_once_with(
                        "resource_assessment_override",
                        {
                            "step_id": step.step_id,
                            "forced_tool_usage": True,
                            "baseline_type": BaselineType.PREACT_ONLY,
                        },
                    )


class TestSEMOnlyAgent:
    """Test SEMOnlyAgent specific functionality."""

    @pytest.mark.asyncio
    async def test_execute_spre_pipeline_override(self):
        """Test that _execute_spre_pipeline creates single-step plan."""
        config = AgentConfig(name="Test", role=AgentRole.GENERALIST, spree_enabled=True)

        with patch("llamaagent.benchmarks.baseline_agents.ToolRegistry"):
            with patch(
                "llamaagent.benchmarks.baseline_agents.get_all_tools", return_value=[]
            ):
                agent = SEMOnlyAgent(config)

                with patch.object(agent, "add_trace") as mock_add_trace:
                    with patch.object(
                        agent, "_execute_plan_with_resource_assessment"
                    ) as mock_execute_plan:
                        mock_execute_plan.return_value = [{"result": "Task completed"}]

                        result = await agent._execute_spre_pipeline("test task")

                        assert result == "Task completed"
                        mock_add_trace.assert_called_once_with(
                            "sem_only_execution",
                            {"baseline_type": BaselineType.SEM_ONLY},
                        )
                        mock_execute_plan.assert_called_once()

                        # Check that a single-step plan was created
                        plan_arg = mock_execute_plan.call_args[0][0]
                        assert plan_arg.original_task == "test task"
                        assert len(plan_arg.steps) == 1
                        assert plan_arg.steps[0].description == "test task"

    @pytest.mark.asyncio
    async def test_execute_spre_pipeline_empty_results(self):
        """Test _execute_spre_pipeline with empty step results."""
        config = AgentConfig(name="Test", role=AgentRole.GENERALIST, spree_enabled=True)

        with patch("llamaagent.benchmarks.baseline_agents.ToolRegistry"):
            with patch(
                "llamaagent.benchmarks.baseline_agents.get_all_tools", return_value=[]
            ):
                agent = SEMOnlyAgent(config)

                with patch.object(agent, "add_trace"):
                    with patch.object(
                        agent, "_execute_plan_with_resource_assessment"
                    ) as mock_execute_plan:
                        mock_execute_plan.return_value = []  # Empty results

                        result = await agent._execute_spre_pipeline("test task")

                        assert result == "Task 'test task' completed"

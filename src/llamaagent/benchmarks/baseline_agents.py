from __future__ import annotations

"""Baseline agent implementations for scientific comparison.

This module implements the four baseline configurations specified in the
research document for rigorous SPRE evaluation:
1. Vanilla ReAct
2. Pre-Act Only
3. SEM Only
4. SPRE Agent (full implementation)
"""

from typing import Any, Dict

from ..agents.base import AgentConfig, AgentRole
from ..agents.react import ReactAgent
from ..llm import LLMProvider
from ..tools import ToolRegistry, get_all_tools

__all__ = ["BaselineAgentFactory", "BaselineType"]


class BaselineType:
    """Baseline agent configuration types."""

    VANILLA_REACT = "vanilla_react"
    PREACT_ONLY = "preact_only"
    SEM_ONLY = "sem_only"
    SPRE_FULL = "spre_full"


class VanillaReactAgent(ReactAgent):
    """Vanilla ReAct agent - standard implementation without SPRE."""

    async def execute(self, task: str, context: Dict[str, Any] | None = None):
        """Execute using simple mode regardless of config."""
        # Override to force simple execution
        original_spree = self.config.spree_enabled
        self.config.spree_enabled = False
        try:
            result = await super().execute(task, context)
            # Add baseline identifier to trace
            self.add_trace("baseline_type", {"type": BaselineType.VANILLA_REACT})
            return result
        finally:
            self.config.spree_enabled = original_spree


class PreActOnlyAgent(ReactAgent):
    """Pre-Act Only agent - plans but executes tools for every step."""

    async def _assess_resource_need(self, step) -> bool:
        """Always return True to force tool usage for every step."""
        self.add_trace(
            "resource_assessment_override",
            {"step_id": step.step_id, "forced_tool_usage": True, "baseline_type": BaselineType.PREACT_ONLY},
        )
        return True  # Always use tools


class SEMOnlyAgent(ReactAgent):
    """SEM Only agent - reactive with resource assessment but no planning."""

    async def _execute_spre_pipeline(self, task: str, context: Dict[str, Any] | None = None) -> str:
        """Override to skip planning phase."""
        self.add_trace("sem_only_execution", {"baseline_type": BaselineType.SEM_ONLY})

        # Create single-step plan
        from ..agents.base import ExecutionPlan, PlanStep

        plan = ExecutionPlan(
            original_task=task,
            steps=[
                PlanStep(
                    step_id=1,
                    description=task,
                    required_information="Complete task solution",
                    expected_outcome="Task completion",
                )
            ],
        )

        # Execute with resource assessment
        step_results = await self._execute_plan_with_resource_assessment(plan, context)

        # Simple synthesis (just return the single result)
        return step_results[0]["result"] if step_results else f"Task '{task}' completed"


class BaselineAgentFactory:
    """Factory for creating baseline agents for scientific comparison."""

    @staticmethod
    def create_agent(baseline_type: str, llm_provider: LLMProvider | None = None, name_suffix: str = "") -> ReactAgent:
        """Create agent of specified baseline type."""

        # Common tool setup
        tools = ToolRegistry()
        for tool in get_all_tools():
            tools.register(tool)

        base_name = f"{baseline_type.replace('_', '-').title()}{name_suffix}"

        if baseline_type == BaselineType.VANILLA_REACT:
            config = AgentConfig(
                name=f"{base_name}-Vanilla", role=AgentRole.GENERALIST, spree_enabled=False, max_iterations=10
            )
            return VanillaReactAgent(config, llm_provider=llm_provider, tools=tools)

        elif baseline_type == BaselineType.PREACT_ONLY:
            config = AgentConfig(
                name=f"{base_name}-PreAct",
                role=AgentRole.PLANNER,
                spree_enabled=True,  # Enable planning
                max_iterations=10,
            )
            return PreActOnlyAgent(config, llm_provider=llm_provider, tools=tools)

        elif baseline_type == BaselineType.SEM_ONLY:
            config = AgentConfig(
                name=f"{base_name}-SEM",
                role=AgentRole.GENERALIST,
                spree_enabled=True,  # Enable resource assessment
                max_iterations=10,
            )
            return SEMOnlyAgent(config, llm_provider=llm_provider, tools=tools)

        elif baseline_type == BaselineType.SPRE_FULL:
            config = AgentConfig(
                name=f"{base_name}-SPRE", role=AgentRole.PLANNER, spree_enabled=True, max_iterations=10
            )
            return ReactAgent(config, llm_provider=llm_provider, tools=tools)

        else:
            raise ValueError(f"Unknown baseline type: {baseline_type}")

    @staticmethod
    def get_all_baseline_types() -> list[str]:
        """Get list of all available baseline types."""
        return [BaselineType.VANILLA_REACT, BaselineType.PREACT_ONLY, BaselineType.SEM_ONLY, BaselineType.SPRE_FULL]

    @staticmethod
    def get_baseline_description(baseline_type: str) -> str:
        """Get description of baseline type."""
        descriptions = {
            BaselineType.VANILLA_REACT: "Standard ReAct agent without planning or resource assessment",
            BaselineType.PREACT_ONLY: "Agent with planning but executes tools for every step",
            BaselineType.SEM_ONLY: "Reactive agent with resource assessment but no strategic planning",
            BaselineType.SPRE_FULL: "Full SPRE implementation with planning and resource assessment",
        }
        return descriptions.get(baseline_type, "Unknown baseline type")

"""
Baseline agents for benchmarking and scientific comparison.

This module provides different baseline agent implementations to compare against
the full SPRE (Strategic Planning & Resourceful Execution) framework.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List

from ..agents.base import AgentRole, ExecutionPlan, PlanStep
from ..agents.react import ReactAgent
from ..llm import LLMProvider
from ..tools import ToolRegistry, get_all_tools
from ..types import AgentConfig


class BaselineType(str, Enum):
    """Baseline agent configuration types."""

    VANILLA_REACT = "vanilla_react"
    PREACT_ONLY = "preact_only"
    SEM_ONLY = "sem_only"
    SPRE_FULL = "spre_full"


class VanillaReactAgent(ReactAgent):
    """Vanilla ReAct agent - standard implementation without SPRE.

    This agent executes tasks without strategic planning or resource assessment,
    following the traditional ReAct pattern of thought-action-observation.
    """

    async def execute(self, task: str, context: Dict[str, Any] | None = None) -> str:
        """Execute using simple mode regardless of config."""
        # Override to force simple execution
        original_spre = self.config.spree_enabled
        self.config.spree_enabled = False

        try:
            # Add baseline identifier to trace
            self.add_trace("baseline_type", {"type": BaselineType.VANILLA_REACT})

            result = await super().execute(task, context)
            return result
        finally:
            self.config.spree_enabled = original_spre


class PreActOnlyAgent(ReactAgent):
    """Pre-Act Only agent - plans but executes tools for every step.

    This agent performs strategic planning but doesn't use resource assessment,
    executing tools for every step regardless of whether they're needed.
    """

    async def _assess_resource_need(self, step: PlanStep) -> bool:
        """Always return True to force tool usage for every step."""
        self.add_trace(
            "resource_assessment_override",
            {
                "step_id": step.step_id,
                "forced_tool_usage": True,
                "reason": "PreAct baseline always uses tools",
            },
        )
        return True  # Always use tools


class SEMOnlyAgent(ReactAgent):
    """SEM Only agent - reactive with resource assessment but no planning.

    This agent uses Selective Execution Mode (resource assessment) but doesn't
    perform upfront strategic planning, making decisions reactively.
    """

    async def _execute_spre_pipeline(
        self, task: str, context: Dict[str, Any] | None = None
    ) -> str:
        """Override to skip planning phase."""
        self.add_trace("sem_only_execution", {"baseline_type": BaselineType.SEM_ONLY})

        # Create single-step plan
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
    def create_agent(
        baseline_type: str,
        llm_provider: LLMProvider | None = None,
        name_suffix: str = "",
        **kwargs,
    ) -> ReactAgent:
        """Create agent of specified baseline type.

        Args:
            baseline_type: Type of baseline agent to create
            llm_provider: Optional LLM provider to use
            name_suffix: Optional suffix for agent name
            **kwargs: Additional configuration options

        Returns:
            Configured baseline agent instance
        """
        # Common tool setup
        tools_registry = ToolRegistry()
        for tool in get_all_tools():
            tools_registry.register(tool)

        # Get list of tool names for the config
        tool_names = tools_registry.list_names()

        base_name = f"{baseline_type.replace('_', '-').title()}{name_suffix}"

        if baseline_type == BaselineType.VANILLA_REACT:
            config = AgentConfig(
                agent_name=f"{base_name}-Vanilla",
                metadata={
                    "role": AgentRole.GENERALIST,
                    "spree_enabled": False,
                    "max_iterations": 10,
                },
                tools=tool_names,  # Pass list of tool names, not registry
            )
            return VanillaReactAgent(
                config, llm_provider=llm_provider, tools=tools_registry
            )

        elif baseline_type == BaselineType.PREACT_ONLY:
            config = AgentConfig(
                name=f"{base_name}-PreAct",
                role=AgentRole.PLANNER,
                spree_enabled=True,  # Enable planning
                max_iterations=10,
                tools=tool_names,  # Pass list of tool names, not registry
            )
            return PreActOnlyAgent(
                config, llm_provider=llm_provider, tools=tools_registry
            )

        elif baseline_type == BaselineType.SEM_ONLY:
            config = AgentConfig(
                name=f"{base_name}-SEM",
                role=AgentRole.GENERALIST,
                spree_enabled=True,  # Enable resource assessment
                max_iterations=10,
                tools=tool_names,  # Pass list of tool names, not registry
            )
            return SEMOnlyAgent(config, llm_provider=llm_provider, tools=tools_registry)

        elif baseline_type == BaselineType.SPRE_FULL:
            config = AgentConfig(
                name=f"{base_name}-SPRE",
                role=AgentRole.PLANNER,
                spree_enabled=True,  # Full SPRE
                max_iterations=10,
                tools=tool_names,  # Pass list of tool names, not registry
            )
            return ReactAgent(config, llm_provider=llm_provider, tools=tools_registry)

        else:
            raise ValueError(f"Unknown baseline type: {baseline_type}")

    @staticmethod
    def get_all_baseline_types() -> List[str]:
        """Get list of all available baseline types."""
        return [
            BaselineType.VANILLA_REACT,
            BaselineType.PREACT_ONLY,
            BaselineType.SEM_ONLY,
            BaselineType.SPRE_FULL,
        ]

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


__all__ = [
    "BaselineType",
    "VanillaReactAgent",
    "PreActOnlyAgent",
    "SEMOnlyAgent",
    "BaselineAgentFactory",
]

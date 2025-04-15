"""
Planner component for breaking tasks into executable steps.

The Planner is responsible for:
1. Analyzing a task to understand what needs to be done
2. Breaking it down into a sequence of executable steps
3. Revising the plan as new information becomes available
"""

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .config import AgentConfig
from .llm import get_llm_client
from .memory import Memory
from .types import AgentState, Plan, PlanStep

logger = logging.getLogger(__name__)


class Planner:
    """
    Planner component that creates and revises plans for tasks.

    The Planner uses an LLM to:
    - Analyze the task and break it into steps
    - Determine which tools to use for each step
    - Update the plan based on new observations

    Attributes:
        config (AgentConfig): Configuration for the planner
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize the planner.

        Args:
            config: Configuration settings
        """
        self.config = config or AgentConfig()
        self.llm = get_llm_client(self.config.llm_config)
        logger.debug("Planner initialized")

    def create_plan(
        self, task: str, state: AgentState, memory: Optional[Memory] = None
    ) -> Plan:
        """
        Create a plan for executing a task.

        Args:
            task: The task to plan for
            state: The current state of the agent
            memory: The agent's memory for context

        Returns:
            A Plan object with steps to execute
        """
        logger.info(f"Creating plan for task: {task}")

        # Retrieve relevant memories if available
        context = ""
        if memory:
            relevant_memories = memory.retrieve(task, limit=5)
            if relevant_memories.items:
                context = "Relevant context:\n" + "\n".join(
                    [f"- {item.content}" for item in relevant_memories.items]
                )

        # Create the prompt for the LLM
        tools_descriptions = "\n".join(
            [
                f"- {tool.name}: {tool.description}"
                for tool in self.config.available_tools
            ]
        )

        prompt = f"""
        You are an AI assistant tasked with breaking down a complex task into a series of steps.
        
        TASK: {task}
        
        {context}
        
        Available tools:
        {tools_descriptions}
        
        Create a step-by-step plan to complete this task. For each step:
        1. Provide a clear description of what needs to be done
        2. Specify which tool should be used (if any)
        3. Explain what information is needed for this step
        
        Provide your reasoning for this plan and estimate the number of steps required.
        
        Output format:
        REASONING: <your reasoning here>
        STEPS: <number of steps>
        PLAN:
        1. <step description> | TOOL: <tool name> | ARGS: <key1>=<value1>, <key2>=<value2>
        2. ...
        """

        # Send the prompt to the LLM
        response = self.llm.generate(prompt)

        # Parse the response to extract the plan
        plan_steps, reasoning, estimated_steps = self._parse_plan_response(response)

        # Create the Plan object
        plan = Plan(
            steps=plan_steps, reasoning=reasoning, estimated_steps=estimated_steps
        )

        logger.info(f"Created plan with {len(plan_steps)} steps")
        return plan

    def revise_plan(self, state: AgentState, memory: Optional[Memory] = None) -> Plan:
        """
        Revise a plan based on new observations.

        Args:
            state: The current state of the agent
            memory: The agent's memory for context

        Returns:
            A revised Plan object
        """
        logger.info("Revising plan based on new observations")

        # Extract current plan and observations
        current_plan = state.plan
        if not current_plan:
            # If there's no current plan, create a new one
            return self.create_plan(state.task, state, memory)

        # Get the observations so far
        observations = []
        for i, result in enumerate(state.results):
            step_desc = (
                current_plan.steps[i].description
                if i < len(current_plan.steps)
                else "Unknown step"
            )
            observations.append(f"Step {i+1}: {step_desc} -> {result.observation}")

        observations_text = "\n".join(observations)

        # Create the prompt for the LLM
        tools_descriptions = "\n".join(
            [
                f"- {tool.name}: {tool.description}"
                for tool in self.config.available_tools
            ]
        )

        # Get remaining steps
        remaining_steps = current_plan.steps[state.current_step + 1 :]
        remaining_steps_text = "\n".join(
            [
                f"{i+state.current_step+2}. {step.description}"
                for i, step in enumerate(remaining_steps)
            ]
        )

        prompt = f"""
        You are an AI assistant revising a plan based on new observations.
        
        TASK: {state.task}
        
        Current plan progress:
        {observations_text}
        
        Remaining steps in current plan:
        {remaining_steps_text}
        
        Available tools:
        {tools_descriptions}
        
        Based on the observations so far, revise the plan to complete the task.
        Consider if the remaining steps are still appropriate or if new steps are needed.
        
        Output format:
        REASONING: <your reasoning here>
        STEPS: <number of steps>
        PLAN:
        1. <step description> | TOOL: <tool name> | ARGS: <key1>=<value1>, <key2>=<value2>
        2. ...
        """

        # Send the prompt to the LLM
        response = self.llm.generate(prompt)

        # Parse the response to extract the plan
        plan_steps, reasoning, estimated_steps = self._parse_plan_response(response)

        # Mark completed steps
        for i in range(state.current_step + 1):
            if i < len(current_plan.steps):
                current_plan.steps[i].is_completed = True

        # Create the revised Plan object
        revised_plan = Plan(
            steps=plan_steps, reasoning=reasoning, estimated_steps=estimated_steps
        )

        logger.info(f"Revised plan with {len(plan_steps)} steps")
        return revised_plan

    def _parse_plan_response(self, response: str) -> Tuple[List[PlanStep], str, int]:
        """
        Parse the LLM response to extract plan steps, reasoning, and estimated steps.

        Args:
            response: The LLM's response

        Returns:
            A tuple of (plan_steps, reasoning, estimated_steps)
        """
        # Default values
        reasoning = ""
        estimated_steps = 0
        plan_steps = []

        # Extract reasoning
        if "REASONING:" in response:
            reasoning_parts = response.split("REASONING:")[1].split("STEPS:")[0].strip()
            reasoning = reasoning_parts

        # Extract estimated steps
        if "STEPS:" in response:
            try:
                steps_part = response.split("STEPS:")[1].split("PLAN:")[0].strip()
                estimated_steps = int(steps_part)
            except (ValueError, IndexError):
                estimated_steps = 0

        # Extract plan
        if "PLAN:" in response:
            plan_text = response.split("PLAN:")[1].strip()
            lines = plan_text.split("\n")

            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Extract step description and tool details
                step_id = str(uuid.uuid4())
                tool_name = None
                tool_args = {}

                # Parse the line to extract step components
                try:
                    # Handle numbered steps (e.g., "1. Do something | TOOL: xyz")
                    if line[0].isdigit() and ". " in line:
                        line = line.split(". ", 1)[1]

                    if "|" in line:
                        parts = [part.strip() for part in line.split("|")]
                        description = parts[0]

                        # Extract tool name and args if specified
                        for part in parts[1:]:
                            if part.startswith("TOOL:"):
                                tool_name = part[5:].strip()
                            elif part.startswith("ARGS:"):
                                args_str = part[5:].strip()
                                for arg in args_str.split(","):
                                    if "=" in arg:
                                        key, value = arg.split("=", 1)
                                        tool_args[key.strip()] = value.strip()
                    else:
                        description = line

                    # Create the step
                    step = PlanStep(
                        id=step_id,
                        description=description,
                        tool_name=tool_name,
                        tool_args=tool_args,
                    )

                    plan_steps.append(step)

                except Exception as e:
                    logger.warning(f"Error parsing plan step '{line}': {e}")
                    # Add a simple step with the line as description
                    step = PlanStep(id=str(uuid.uuid4()), description=line)
                    plan_steps.append(step)

        return plan_steps, reasoning, estimated_steps

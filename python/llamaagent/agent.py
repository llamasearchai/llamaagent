"""
Agent implementation that orchestrates models, tools, and memory.
"""

import logging
import uuid
from typing import Any, Callable, Dict, List, Optional, Union

from .config import AgentConfig
from .executor import Executor
from .memory import Memory
from .planner import Planner
from .reflector import Reflector
from .types import AgentResponse, AgentState, Tool, ToolResult

logger = logging.getLogger(__name__)


class Agent:
    """
    A flexible agent that can decompose tasks, use tools, and maintain a memory.

    The Agent uses a modular architecture with four key components:
    1. Planner: Breaks down tasks into steps
    2. Executor: Carries out the steps, using tools as needed
    3. Memory: Stores information for retrieval
    4. Reflector: Analyzes performance and suggests improvements

    Examples:
        >>> from llamaagent import Agent
        >>> from llamaagent.tools import WebSearch, Calculator
        >>> from llamadb import LlamaDB
        >>>
        >>> # Initialize agent with tools and memory
        >>> db = LlamaDB()
        >>> agent = Agent(
        ...     tools=[WebSearch(), Calculator()],
        ...     memory=db.create_collection("agent_memory")
        ... )
        >>>
        >>> # Run the agent on a task
        >>> result = agent.run("Research quantum computing advancements")
        >>> print(result.summary)
    """

    def __init__(
        self,
        tools: Optional[List[Tool]] = None,
        memory: Optional[Memory] = None,
        planner: Optional[Planner] = None,
        executor: Optional[Executor] = None,
        reflector: Optional[Reflector] = None,
        config: Optional[AgentConfig] = None,
    ):
        """
        Initialize an agent with tools and components.

        Args:
            tools: List of tools the agent can use
            memory: Memory system for storing information
            planner: Component for breaking tasks into steps
            executor: Component for executing actions using tools
            reflector: Component for analyzing and improving performance
            config: Configuration settings for the agent
        """
        self.id = str(uuid.uuid4())
        self.tools = tools or []
        self.memory = memory or Memory()
        self.config = config or AgentConfig()

        # Initialize components
        self.planner = planner or Planner(config=self.config)
        self.executor = executor or Executor(tools=self.tools, config=self.config)
        self.reflector = reflector or Reflector(config=self.config)

        # Initialize state
        self.state = AgentState(
            id=self.id,
            task=None,
            plan=None,
            current_step=0,
            observations=[],
            results=[],
        )

        logger.info(f"Agent initialized with {len(self.tools)} tools")

    def add_tool(self, tool: Tool) -> None:
        """
        Add a new tool to the agent.

        Args:
            tool: The tool to add
        """
        self.tools.append(tool)
        self.executor.tools = self.tools
        logger.debug(f"Tool {tool.name} added to agent")

    def run(self, task: str, **kwargs) -> AgentResponse:
        """
        Run the agent on a task.

        This is the main entry point for the agent. It will:
        1. Create a plan for the task
        2. Execute the plan step by step
        3. Reflect on the execution to improve
        4. Return the results

        Args:
            task: The task to perform
            **kwargs: Additional arguments to pass to the components

        Returns:
            An AgentResponse with the results and metadata
        """
        logger.info(f"Starting agent run for task: {task}")

        # Reset state for new task
        self.state = AgentState(
            id=self.id,
            task=task,
            plan=None,
            current_step=0,
            observations=[],
            results=[],
        )

        # Store task in memory
        if self.memory:
            self.memory.add(
                {"type": "task", "content": task, "timestamp": self.state.timestamp}
            )

        # Create a plan
        logger.debug("Creating plan")
        self.state.plan = self.planner.create_plan(task, self.state, self.memory)

        # Execute each step in the plan
        for i, step in enumerate(self.state.plan.steps):
            logger.debug(f"Executing step {i+1}: {step.description}")
            self.state.current_step = i

            # Execute the step
            result = self.executor.execute_step(step, self.state, self.memory)

            # Record the result
            self.state.observations.append(result.observation)
            self.state.results.append(result)

            # Store in memory
            if self.memory:
                self.memory.add(
                    {
                        "type": "observation",
                        "step": i,
                        "content": result.observation,
                        "timestamp": result.timestamp,
                    }
                )

            # Check if we need to revise the plan
            if result.requires_planning:
                logger.debug("Revising plan based on new information")
                self.state.plan = self.planner.revise_plan(self.state, self.memory)

        # Reflect on the execution
        reflection = self.reflector.reflect(self.state, self.memory)

        # Store reflection in memory
        if self.memory and reflection:
            self.memory.add(
                {
                    "type": "reflection",
                    "content": reflection,
                    "timestamp": self.state.timestamp,
                }
            )

        # Prepare the response
        summary = self._generate_summary(self.state)

        logger.info(f"Agent run completed for task: {task}")

        return AgentResponse(
            task=task, summary=summary, steps=self.state.results, reflection=reflection
        )

    def _generate_summary(self, state: AgentState) -> str:
        """
        Generate a summary of the results.

        Args:
            state: The current agent state

        Returns:
            A string summarizing the results
        """
        # For now, use a simple summary approach
        # In a real implementation, this would use an LLM to generate a coherent summary

        if not state.results:
            return "No steps were executed."

        # Collect all observations
        observations = [
            result.observation for result in state.results if result.observation
        ]

        # Join them with separators
        summary = "\n\n".join(observations)

        # If it's too long, truncate and add ellipsis
        if len(summary) > 1000:
            summary = summary[:997] + "..."

        return summary

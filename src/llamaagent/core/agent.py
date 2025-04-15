"""
Core Agent implementation for LlamaAgent
"""

import json
import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator

from llamaagent.core.memory import Memory
from llamaagent.core.tool import Tool
from llamaagent.core.workflow import Workflow

logger = logging.getLogger(__name__)


class AgentResult(BaseModel):
    """
    Result of an agent execution
    """

    agent_id: str
    task_id: str
    input: str
    output: str
    tools_used: List[str] = Field(default_factory=list)
    thinking: Optional[str] = None
    execution_time: float
    successful: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Agent(BaseModel):
    """
    Agent that can execute tasks using tools and memory
    """

    name: str
    description: str
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 1000
    tools: List[Tool] = Field(default_factory=list)
    memory: Optional[Memory] = None
    workflows: List[Workflow] = Field(default_factory=list)
    system_prompt: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize default system prompt if not provided
        if not self.system_prompt:
            self.system_prompt = self._get_default_system_prompt()

    def _get_default_system_prompt(self) -> str:
        """
        Get the default system prompt for the agent
        """
        prompt = f"""You are {self.name}, {self.description}.
You can use tools to help you complete tasks.
Always think step-by-step about how to solve the task.
If you need information, use the appropriate tools to get it.
Respond directly to the user with your final answer."""

        # Add tool descriptions if available
        if self.tools:
            prompt += "\n\nAvailable tools:"
            for tool in self.tools:
                prompt += f"\n- {tool.name}: {tool.description}"

        return prompt

    def add_tool(self, tool: Tool) -> None:
        """
        Add a tool to the agent
        """
        self.tools.append(tool)
        # Update system prompt to include new tool
        self.system_prompt = self._get_default_system_prompt()

    def add_workflow(self, workflow: Workflow) -> None:
        """
        Add a workflow to the agent
        """
        self.workflows.append(workflow)

    def run(self, task: str, **kwargs) -> AgentResult:
        """
        Run the agent on a task

        Args:
            task: The task to execute
            **kwargs: Additional parameters for execution

        Returns:
            AgentResult: The result of the execution
        """
        task_id = str(uuid.uuid4())
        start_time = time.time()
        tools_used = []

        logger.info(f"Agent {self.name} ({self.id}) starting task: {task}")

        try:
            # Check if task matches a workflow
            workflow = self._find_matching_workflow(task)

            if workflow:
                logger.info(f"Using workflow: {workflow.name}")
                output, thinking, tools_used = self._execute_workflow(
                    workflow, task, **kwargs
                )
            else:
                # Execute task directly
                output, thinking, tools_used = self._execute_task(task, **kwargs)

            # Save to memory if available
            if self.memory:
                self.memory.add(task=task, response=output)

            execution_time = time.time() - start_time

            logger.info(f"Agent {self.name} completed task in {execution_time:.2f}s")

            return AgentResult(
                agent_id=self.id,
                task_id=task_id,
                input=task,
                output=output,
                thinking=thinking,
                tools_used=tools_used,
                execution_time=execution_time,
                successful=True,
                metadata=kwargs.get("metadata", {}),
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            logger.error(f"Agent {self.name} failed: {error_msg}")

            return AgentResult(
                agent_id=self.id,
                task_id=task_id,
                input=task,
                output="",
                tools_used=tools_used,
                execution_time=execution_time,
                successful=False,
                error=error_msg,
                metadata=kwargs.get("metadata", {}),
            )

    def _find_matching_workflow(self, task: str) -> Optional[Workflow]:
        """
        Find a workflow that matches the task
        """
        # Simple matching for now - can be improved with embeddings/LLM matching
        for workflow in self.workflows:
            if any(keyword in task.lower() for keyword in workflow.keywords):
                return workflow
        return None

    def _execute_workflow(self, workflow: Workflow, task: str, **kwargs) -> tuple:
        """
        Execute a workflow on a task
        """
        output = ""
        thinking = ""
        tools_used = []

        # Execute workflow steps
        current_input = task
        for step in workflow.steps:
            step_thinking = f"Step: {step.name}\nInput: {current_input}\n"
            thinking += step_thinking

            # Find the tool for this step
            tool = next((t for t in self.tools if t.name == step.tool), None)
            if not tool:
                thinking += f"Error: Tool '{step.tool}' not found\n"
                continue

            # Execute the tool
            try:
                tools_used.append(step.tool)
                tool_result = tool.run(current_input)
                current_input = (
                    tool_result  # Output of one step becomes input to the next
                )

                step_thinking += f"Output: {tool_result}\n\n"
                thinking += step_thinking
            except Exception as e:
                step_thinking += f"Error: {str(e)}\n\n"
                thinking += step_thinking
                if not step.optional:
                    raise

        output = current_input  # Final step output is the workflow output
        return output, thinking, tools_used

    def _execute_task(self, task: str, **kwargs) -> tuple:
        """
        Execute a task directly using LLM and tools
        """
        # This is a simplified implementation
        # In a real implementation, this would use the LLM to:
        # 1. Parse the task
        # 2. Decide which tools to use
        # 3. Execute the tools
        # 4. Synthesize the results

        # Placeholder implementation
        output = f"I've processed your task: {task}"
        thinking = f"Thinking about how to solve: {task}\n"
        tools_used = []

        # Simple keyword matching to decide which tools to use
        for tool in self.tools:
            if any(keyword in task.lower() for keyword in tool.keywords):
                thinking += f"Using tool: {tool.name}\n"
                try:
                    tools_used.append(tool.name)
                    tool_result = tool.run(task)
                    thinking += f"Tool result: {tool_result}\n"
                    output = f"Using {tool.name}, I found: {tool_result}"
                except Exception as e:
                    thinking += f"Tool error: {str(e)}\n"

        return output, thinking, tools_used

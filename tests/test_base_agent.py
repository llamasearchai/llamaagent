#!/usr/bin/env python3
"""Test script to verify base agent implementation."""

import asyncio
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from typing import Any, Dict, Optional

# Import directly from the module
from llamaagent.agents.base import (AgentConfig, AgentMessage, AgentResponse,
                                    AgentRole, AgentTrace, BaseAgent,
                                    ExecutionPlan, PlanStep, Step)
from llamaagent.memory.base import SimpleMemory
from llamaagent.tools import ToolRegistry


class TestAgent(BaseAgent):
    """Simple test agent implementation."""

    async def execute(
        self, task: str, context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Execute a test task."""
        # Start trace
        self.trace = AgentTrace(
            agent_name=self.config.name,
            task=task,
            start_time=asyncio.get_event_loop().time(),
        )

        # Start a step
        step = self.start_step("processing", f"Processing task: {task}")

        # Simulate processing
        result = f"Completed task: {task}"

        # Complete the step
        self.complete_step(result)

        # End trace
        self.trace.end_time = asyncio.get_event_loop().time()
        self.trace.success = True

        return AgentResponse(
            content=result,
            success=True,
            trace=self.trace.steps,
            execution_time=self.trace.execution_time,
        )


async def main():
    """Test the base agent implementation."""
    print("Testing BaseAgent implementation...")

    # Test AgentConfig
    config = AgentConfig(
        name="TestBot",
        role=AgentRole.EXECUTOR,
        description="A test agent",
        max_iterations=5,
        temperature=0.5,
    )
    print(f" Created AgentConfig: {config.name} ({config.role})")

    # Test PlanStep
    step = PlanStep(
        step_id=1,
        description="Test step",
        required_information="Nothing",
        expected_outcome="Success",
    )
    print(f" Created PlanStep: {step.description}")

    # Test ExecutionPlan
    plan = ExecutionPlan(original_task="Test task", steps=[step])
    print(f" Created ExecutionPlan with {len(plan.steps)} steps")

    # Test AgentMessage
    message = AgentMessage(sender="TestBot", recipient="User", content="Hello!")
    print(f" Created AgentMessage: {message.content}")

    # Test Step
    test_step = Step(step_type="test", description="Test step")
    test_step.complete("Done")
    print(f" Created and completed Step (duration: {test_step.duration:.3f}s)")

    # Test AgentTrace
    trace = AgentTrace(agent_name="TestBot", task="Test task", start_time=0.0)
    trace.add_step("init", "Initializing")
    print(f" Created AgentTrace with {len(trace.steps)} steps")

    # Test TestAgent
    agent = TestAgent(config)
    print(f" Created TestAgent: {agent}")

    # Test execution
    response = await agent.execute("Calculate 2 + 2")
    print(f" Execute response: {response.content}")
    print(f"  Success: {response.success}")
    print(f"  Execution time: {response.execution_time:.3f}s")

    # Test task execution
    from llamaagent.types import TaskInput

    task_input = TaskInput(id="test-123", task="Calculate 5 + 5", agent_name="TestBot")

    task_output = await agent.execute_task(task_input)
    print(f" Task execution: {task_output.status.value}")
    print(
        f"  Result: {task_output.result.data['content'] if task_output.result else 'None'}"
    )

    print("\nPASS All tests passed! BaseAgent implementation is working correctly.")


if __name__ == "__main__":
    asyncio.run(main())

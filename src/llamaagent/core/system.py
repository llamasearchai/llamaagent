"""
Agent System implementation for multi-agent coordination
"""

import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator

from llamaagent.core.agent import Agent, AgentResult

logger = logging.getLogger(__name__)


class AgentSystemResult(BaseModel):
    """
    Result of an agent system execution
    """

    system_id: str
    task_id: str
    input: str
    output: str
    agent_results: List[AgentResult] = Field(default_factory=list)
    execution_time: float
    successful: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentSystem(BaseModel):
    """
    System of multiple agents that can work together
    """

    name: str
    description: Optional[str] = None
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agents: List[Agent] = Field(default_factory=list)
    coordinator: str = "round_robin"  # round_robin, sequential, parallel, hierarchical
    shared_memory: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    def add_agent(self, agent: Agent) -> None:
        """
        Add an agent to the system
        """
        self.agents.append(agent)

    def run(self, task: str, **kwargs) -> AgentSystemResult:
        """
        Run the agent system on a task

        Args:
            task: The task to execute
            **kwargs: Additional parameters for execution

        Returns:
            AgentSystemResult: The result of the execution
        """
        task_id = str(uuid.uuid4())
        start_time = time.time()
        agent_results = []

        logger.info(f"Agent system {self.name} ({self.id}) starting task: {task}")

        try:
            # Execute based on coordination strategy
            if self.coordinator == "round_robin":
                output = self._run_round_robin(task, agent_results, **kwargs)
            elif self.coordinator == "sequential":
                output = self._run_sequential(task, agent_results, **kwargs)
            elif self.coordinator == "parallel":
                output = self._run_parallel(task, agent_results, **kwargs)
            elif self.coordinator == "hierarchical":
                output = self._run_hierarchical(task, agent_results, **kwargs)
            else:
                raise ValueError(f"Unknown coordinator type: {self.coordinator}")

            execution_time = time.time() - start_time

            logger.info(
                f"Agent system {self.name} completed task in {execution_time:.2f}s"
            )

            return AgentSystemResult(
                system_id=self.id,
                task_id=task_id,
                input=task,
                output=output,
                agent_results=agent_results,
                execution_time=execution_time,
                successful=True,
                metadata=kwargs.get("metadata", {}),
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            logger.error(f"Agent system {self.name} failed: {error_msg}")

            return AgentSystemResult(
                system_id=self.id,
                task_id=task_id,
                input=task,
                output="",
                agent_results=agent_results,
                execution_time=execution_time,
                successful=False,
                error=error_msg,
                metadata=kwargs.get("metadata", {}),
            )

    def _run_round_robin(
        self,
        task: str,
        agent_results: List[AgentResult],
        max_iterations: int = 5,
        **kwargs,
    ) -> str:
        """
        Run the task using round-robin coordination between agents

        Each agent takes a turn processing the result of the previous agent.
        """
        current_task = task

        for i in range(max_iterations):
            logger.info(f"Round-robin iteration {i+1}/{max_iterations}")

            for agent in self.agents:
                logger.info(f"Agent {agent.name} processing: {current_task[:50]}...")

                result = agent.run(current_task, **kwargs)
                agent_results.append(result)

                if not result.successful:
                    logger.warning(f"Agent {agent.name} failed: {result.error}")
                    continue

                current_task = result.output

                # If the task seems complete, stop iterating
                if self._check_task_completion(current_task, task):
                    logger.info(f"Task appears complete after agent {agent.name}")
                    return current_task

        return current_task

    def _run_sequential(
        self, task: str, agent_results: List[AgentResult], **kwargs
    ) -> str:
        """
        Run the task sequentially through agents

        Each agent processes the result of the previous agent, but only once.
        """
        current_task = task

        for agent in self.agents:
            logger.info(f"Agent {agent.name} processing: {current_task[:50]}...")

            result = agent.run(current_task, **kwargs)
            agent_results.append(result)

            if not result.successful:
                logger.warning(f"Agent {agent.name} failed: {result.error}")
                continue

            current_task = result.output

        return current_task

    def _run_parallel(
        self, task: str, agent_results: List[AgentResult], **kwargs
    ) -> str:
        """
        Run the task in parallel across all agents

        All agents process the original task, then results are combined.
        """
        import concurrent.futures

        def run_agent(agent):
            logger.info(f"Agent {agent.name} processing task")
            return agent.run(task, **kwargs)

        # Run agents in parallel
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.agents)
        ) as executor:
            future_to_agent = {
                executor.submit(run_agent, agent): agent for agent in self.agents
            }

            for future in concurrent.futures.as_completed(future_to_agent):
                agent = future_to_agent[future]
                try:
                    result = future.result()
                    agent_results.append(result)
                    logger.info(f"Agent {agent.name} completed")
                except Exception as e:
                    logger.error(f"Agent {agent.name} generated an exception: {e}")

        # Combine results
        successful_results = [r for r in agent_results if r.successful]

        if not successful_results:
            raise RuntimeError("All agents failed to process the task")

        # Simple combination strategy - join with newlines
        combined_output = "\n\n".join(
            [
                f"== Agent {r.agent_id} ({self.get_agent_name(r.agent_id)}) ==\n{r.output}"
                for r in successful_results
            ]
        )

        return combined_output

    def _run_hierarchical(
        self, task: str, agent_results: List[AgentResult], **kwargs
    ) -> str:
        """
        Run the task using a hierarchical approach

        The first agent is the coordinator who delegates to other agents.
        """
        if not self.agents:
            raise ValueError("No agents in the system")

        # The first agent is the coordinator
        coordinator = self.agents[0]
        worker_agents = self.agents[1:]

        if not worker_agents:
            logger.warning("No worker agents, running with coordinator only")
            result = coordinator.run(task, **kwargs)
            agent_results.append(result)
            return result.output

        # Create a mapping of agent IDs to names for the coordinator to reference
        agent_mapping = {agent.id: agent.name for agent in worker_agents}

        # Create a special task for the coordinator
        coordinator_task = f"""
Task: {task}

You are the coordinator agent. You have the following worker agents to help you:
{chr(10).join([f"- {agent.name}: {agent.description}" for agent in worker_agents])}

Break down the task and decide which agent should handle each part.
Provide a plan with specific instructions for each agent.
"""

        # Run the coordinator to get the plan
        logger.info(f"Coordinator agent {coordinator.name} planning task")
        coordinator_result = coordinator.run(coordinator_task, **kwargs)
        agent_results.append(coordinator_result)

        if not coordinator_result.successful:
            raise RuntimeError(f"Coordinator failed: {coordinator_result.error}")

        # Parse the coordinator's plan - this is a simplified implementation
        # In a real system, this would use the LLM to parse the plan more intelligently
        plan = coordinator_result.output

        # Execute the plan with each worker agent
        worker_results = []
        for agent in worker_agents:
            # Create a task for this agent from the plan
            agent_task = f"""
Task: {task}

Coordinator's instructions for you ({agent.name}):
{plan}

Focus only on the parts relevant to your role as {agent.name}.
"""

            logger.info(f"Worker agent {agent.name} processing assigned task")
            worker_result = agent.run(agent_task, **kwargs)
            agent_results.append(worker_result)
            worker_results.append(worker_result)

        # Have the coordinator synthesize the results
        synthesis_task = f"""
Original Task: {task}

You received results from your worker agents. Synthesize these into a cohesive final answer.

Worker Results:
{chr(10).join([f"--- {self.get_agent_name(r.agent_id)} ---\n{r.output}\n" for r in worker_results])}
"""

        logger.info(f"Coordinator agent {coordinator.name} synthesizing results")
        final_result = coordinator.run(synthesis_task, **kwargs)
        agent_results.append(final_result)

        if not final_result.successful:
            raise RuntimeError(f"Coordinator failed at synthesis: {final_result.error}")

        return final_result.output

    def get_agent_name(self, agent_id: str) -> str:
        """
        Get an agent's name by ID
        """
        for agent in self.agents:
            if agent.id == agent_id:
                return agent.name
        return "Unknown Agent"

    def _check_task_completion(self, current_output: str, original_task: str) -> bool:
        """
        Check if the task appears complete based on heuristics
        """
        # This is a simplified implementation
        # In a real system, this would use more sophisticated heuristics or an LLM

        # Check for common completion indicators
        completion_indicators = [
            "In conclusion",
            "To summarize",
            "In summary",
            "Final answer",
            "The answer is",
        ]

        for indicator in completion_indicators:
            if indicator.lower() in current_output.lower():
                return True

        # Check if output is reasonably long compared to the task
        if len(current_output) > len(original_task) * 2:
            return True

        return False

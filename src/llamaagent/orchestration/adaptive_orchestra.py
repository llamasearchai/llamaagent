"""Adaptive orchestration system for intelligent multi-agent coordination.

This module implements sophisticated orchestration capabilities including:
- Dynamic agent selection and assignment
- Multi-agent task coordination
- Intelligent workload distribution
- Performance-based agent optimization
- Collaborative task execution

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AgentSpecialty(Enum):
    """Agent specialization areas."""

    CODE = "code"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    MULTIMODAL = "multimodal"
    GENERAL = "general"


class TaskComplexity(Enum):
    """Task complexity levels."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskInput:
    """Input specification for orchestrated tasks."""

    id: str
    prompt: str
    task_type: str
    priority: int = 1
    max_agents: int = 3
    timeout: int = 300
    require_consensus: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskOutput:
    """Output from orchestrated task execution."""

    task_id: str
    result: str
    confidence: float
    agents_used: List[str]
    execution_time: float
    status: TaskStatus
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskRequirement:
    """Requirements analysis for task execution."""

    complexity: float
    required_specialties: List[AgentSpecialty]
    estimated_time: int
    confidence_threshold: float = 0.7
    collaboration_needed: bool = False


@dataclass
class AgentPerformance:
    """Performance metrics for individual agents."""

    agent_id: str
    task_count: int = 0
    success_rate: float = 0.0
    average_confidence: float = 0.0
    average_execution_time: float = 0.0
    specialty_scores: Dict[AgentSpecialty, float] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class BaseAgent:
    """Base class for orchestrated agents."""

    def __init__(
        self,
        agent_id: str,
        specialty: AgentSpecialty,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.agent_id = agent_id
        self.specialty = specialty
        self.config = config or {}
        self.performance = AgentPerformance(agent_id=agent_id)
        self.is_busy = False

    async def execute_task(self, task_input: TaskInput) -> TaskOutput:
        """Execute a task - to be overridden by specific agent implementations."""
        from src.llamaagent.agents.base import TaskOutput, TaskStatus

        # Default implementation for base agent
        return TaskOutput(
            result=f"Task executed by {self.agent_id}",
            status=TaskStatus.COMPLETED,
            metadata={"agent_id": self.agent_id, "specialty": self.specialty.value},
        )

    async def estimate_task_fit(self, task_input: TaskInput) -> float:
        """Estimate how well this agent fits the task (0-1 score)."""
        # Basic fitness estimation - can be overridden
        base_score = 0.5

        # Boost score if task type matches specialty
        if task_input.task_type.lower() in self.specialty.value:
            base_score += 0.3

        # Consider performance history
        if self.performance.success_rate > 0:
            base_score += (self.performance.success_rate - 0.5) * 0.4

        return min(1.0, max(0.0, base_score))


class CodeSpecialistAgent(BaseAgent):
    """Agent specialized in code generation and analysis."""

    def __init__(self, agent_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, AgentSpecialty.CODE, config)

    async def execute_task(self, task_input: TaskInput) -> TaskOutput:
        """Execute code-related tasks."""
        start_time = time.time()

        try:
            # Simulate code generation/analysis
            await asyncio.sleep(0.2)  # Simulate processing time

            # In real implementation, this would call the LLM
            result = f"Code solution for: {task_input.prompt}\n\n```python\n# Implementation here\ndef solution():\n    # Code logic\n    return result\n```"

            return TaskOutput(
                task_id=task_input.id,
                result=result,
                confidence=0.85,
                agents_used=[self.agent_id],
                execution_time=time.time() - start_time,
                status=TaskStatus.COMPLETED,
                reasoning="Applied code generation best practices",
            )

        except Exception as e:
            return TaskOutput(
                task_id=task_input.id,
                result=f"Error: {str(e)}",
                confidence=0.0,
                agents_used=[self.agent_id],
                execution_time=time.time() - start_time,
                status=TaskStatus.FAILED,
            )


class ResearchSpecialistAgent(BaseAgent):
    """Agent specialized in research and analysis."""

    def __init__(self, agent_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, AgentSpecialty.RESEARCH, config)

    async def execute_task(self, task_input: TaskInput) -> TaskOutput:
        """Execute research-related tasks."""
        start_time = time.time()

        try:
            # Simulate research process
            await asyncio.sleep(0.3)  # Simulate processing time

            result = f"Research analysis for: {task_input.prompt}\n\nBased on comprehensive research, here are the key findings and recommendations."

            return TaskOutput(
                task_id=task_input.id,
                result=result,
                confidence=0.80,
                agents_used=[self.agent_id],
                execution_time=time.time() - start_time,
                status=TaskStatus.COMPLETED,
                reasoning="Applied systematic research methodology",
            )

        except Exception as e:
            return TaskOutput(
                task_id=task_input.id,
                result=f"Error: {str(e)}",
                confidence=0.0,
                agents_used=[self.agent_id],
                execution_time=time.time() - start_time,
                status=TaskStatus.FAILED,
            )


class AnalysisSpecialistAgent(BaseAgent):
    """Agent specialized in data analysis and insights."""

    def __init__(self, agent_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, AgentSpecialty.ANALYSIS, config)

    async def execute_task(self, task_input: TaskInput) -> TaskOutput:
        """Execute analysis-related tasks."""
        start_time = time.time()

        try:
            # Simulate analysis process
            await asyncio.sleep(0.25)  # Simulate processing time

            result = f"Analysis results for: {task_input.prompt}\n\nKey insights and data-driven recommendations based on systematic analysis."

            return TaskOutput(
                task_id=task_input.id,
                result=result,
                confidence=0.82,
                agents_used=[self.agent_id],
                execution_time=time.time() - start_time,
                status=TaskStatus.COMPLETED,
                reasoning="Applied statistical analysis and pattern recognition",
            )

        except Exception as e:
            return TaskOutput(
                task_id=task_input.id,
                result=f"Error: {str(e)}",
                confidence=0.0,
                agents_used=[self.agent_id],
                execution_time=time.time() - start_time,
                status=TaskStatus.FAILED,
            )


class CreativeSpecialistAgent(BaseAgent):
    """Agent specialized in creative and innovative tasks."""

    def __init__(self, agent_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, AgentSpecialty.CREATIVE, config)

    async def execute_task(self, task_input: TaskInput) -> TaskOutput:
        """Execute creative tasks."""
        start_time = time.time()

        try:
            # Simulate creative process
            await asyncio.sleep(0.35)  # Simulate processing time

            result = f"Creative solution for: {task_input.prompt}\n\nInnovative approach combining multiple perspectives and creative problem-solving techniques."

            return TaskOutput(
                task_id=task_input.id,
                result=result,
                confidence=0.75,
                agents_used=[self.agent_id],
                execution_time=time.time() - start_time,
                status=TaskStatus.COMPLETED,
                reasoning="Applied creative problem-solving and innovative thinking",
            )

        except Exception as e:
            return TaskOutput(
                task_id=task_input.id,
                result=f"Error: {str(e)}",
                confidence=0.0,
                agents_used=[self.agent_id],
                execution_time=time.time() - start_time,
                status=TaskStatus.FAILED,
            )


class AdaptiveOrchestrator:
    """Main orchestration engine for coordinating multiple agents."""

    def __init__(self, max_parallel_agents: int = 3):
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_performance: Dict[str, AgentPerformance] = {}
        self.task_queue: List[TaskInput] = []
        self.active_tasks: Dict[str, TaskInput] = {}
        self.completed_tasks: Dict[str, TaskOutput] = {}
        self.max_parallel_agents = max_parallel_agents
        self.orchestration_stats = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "average_execution_time": 0.0,
        }
        self._initialize_agents()

    def _initialize_agents(self) -> None:
        """Initialize default specialized agents."""

        # Code specialist
        code_agent = CodeSpecialistAgent("code_specialist_001")
        self.register_agent(code_agent)
        # Research specialist
        research_agent = ResearchSpecialistAgent("research_specialist_001")
        self.register_agent(research_agent)
        # Analysis specialist
        analysis_agent = AnalysisSpecialistAgent("analysis_specialist_001")
        self.register_agent(analysis_agent)
        # Creative specialist
        creative_agent = CreativeSpecialistAgent("creative_specialist_001")
        self.register_agent(creative_agent)
        logger.info(f"Initialized orchestrator with {len(self.agents)} agents")

    def register_agent(self, agent: BaseAgent) -> None:
        """Register a new agent with the orchestrator."""
        self.agents[agent.agent_id] = agent
        self.agent_performance[agent.agent_id] = agent.performance
        logger.info(
            f"Registered agent: {agent.agent_id} (specialty: {agent.specialty.value})"
        )

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the orchestrator."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            del self.agent_performance[agent_id]
            logger.info(f"Unregistered agent: {agent_id}")

    async def execute_task(self, task_input: TaskInput) -> TaskOutput:
        """Execute a task using the most appropriate agent(s)."""

        logger.info(f"Executing task: {task_input.id}")
        # Analyze task requirements
        requirements = await self._analyze_task_requirements(task_input)
        # Select best agent(s) for the task
        selected_agents = await self._select_agents(task_input, requirements)
        if not selected_agents:
            return TaskOutput(
                task_id=task_input.id,
                result="No suitable agents available",
                confidence=0.0,
                agents_used=[],
                execution_time=0.0,
                status=TaskStatus.FAILED,
            )
        # Execute task with selected agents
        if len(selected_agents) == 1:
            result = await self._execute_single_agent(selected_agents[0], task_input)
        else:
            result = await self._execute_multi_agent(selected_agents, task_input)
        # Update performance metrics
        await self._update_performance_metrics(result)
        # Store completed task
        self.completed_tasks[task_input.id] = result

        return result

    async def _analyze_task_requirements(
        self, task_input: TaskInput
    ) -> TaskRequirement:
        """Analyze task to determine requirements and complexity."""

        try:
            # Simulate task analysis
            await asyncio.sleep(0.05)  # Simulate processing time

            # Determine complexity based on prompt length and keywords
            complexity = 0.5  # Default moderate complexity

            if len(task_input.prompt) > 500:
                complexity += 0.2

            # Check for complexity indicators
            complex_keywords = [
                "analyze",
                "research",
                "comprehensive",
                "detailed",
                "complex",
            ]
            if any(
                keyword in task_input.prompt.lower() for keyword in complex_keywords
            ):
                complexity += 0.2

            # Determine required specialties
            required_specialties: List[AgentSpecialty] = []
            if any(
                word in task_input.prompt.lower()
                for word in ["code", "programming", "function", "algorithm"]
            ):
                required_specialties.append(AgentSpecialty.CODE)
            if any(
                word in task_input.prompt.lower()
                for word in ["research", "study", "investigate", "analyze"]
            ):
                required_specialties.append(AgentSpecialty.RESEARCH)
            if any(
                word in task_input.prompt.lower()
                for word in ["data", "statistics", "analysis", "insights"]
            ):
                required_specialties.append(AgentSpecialty.ANALYSIS)
            if any(
                word in task_input.prompt.lower()
                for word in ["creative", "innovative", "design", "brainstorm"]
            ):
                required_specialties.append(AgentSpecialty.CREATIVE)
            if not required_specialties:
                required_specialties = [AgentSpecialty.GENERAL]

            return TaskRequirement(
                complexity=min(1.0, complexity),
                required_specialties=required_specialties,
                estimated_time=max(30, len(task_input.prompt) // 10),
                confidence_threshold=0.7,
                collaboration_needed=len(required_specialties) > 1,
            )

        except Exception as e:
            logger.error(f"Task analysis error: {e}")
            # Default requirements
            return TaskRequirement(
                complexity=0.5,
                required_specialties=[AgentSpecialty.GENERAL],
                estimated_time=60,
                confidence_threshold=0.7,
                collaboration_needed=False,
            )

    async def _select_agents(
        self, task_input: TaskInput, requirements: TaskRequirement
    ) -> List[BaseAgent]:
        """Select the best agents for the task."""

        # Calculate fitness scores for all agents
        agent_scores: List[Tuple[BaseAgent, float]] = []
        for agent in self.agents.values():
            if agent.is_busy:
                continue

            # Calculate base fitness score
            fitness_score = await agent.estimate_task_fit(task_input)
            # Boost score for matching specialties
            if agent.specialty in requirements.required_specialties:
                fitness_score += 0.3

            # Consider performance history
            if agent.performance.success_rate > 0:
                fitness_score += (agent.performance.success_rate - 0.5) * 0.2

            agent_scores.append((agent, fitness_score))
        # Sort by fitness score
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        # Select top agents
        selected_agents = []
        max_agents = min(task_input.max_agents, len(agent_scores))
        for i in range(max_agents):
            if agent_scores[i][1] > 0.4:  # Minimum fitness threshold
                selected_agents.append(agent_scores[i][0])
        # Ensure we have at least one agent if any are available
        if not selected_agents and agent_scores:
            selected_agents = [agent_scores[0][0]]

        return selected_agents

    async def _execute_single_agent(
        self, agent: BaseAgent, task_input: TaskInput
    ) -> TaskOutput:
        """Execute task with a single agent."""

        agent.is_busy = True
        try:
            result = await agent.execute_task(task_input)
            return result
        finally:
            agent.is_busy = False

    async def _execute_multi_agent(
        self, agents: List[BaseAgent], task_input: TaskInput
    ) -> TaskOutput:
        """Execute task with multiple agents and combine results."""

        # Mark agents as busy
        for agent in agents:
            agent.is_busy = True

        try:
            # Execute tasks in parallel
            tasks = [agent.execute_task(task_input) for agent in agents]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Filter successful results
            successful_results: List[TaskOutput] = []
            for result in results:
                if (
                    isinstance(result, TaskOutput)
                    and result.status == TaskStatus.COMPLETED
                ):
                    successful_results.append(result)
            if not successful_results:
                # If no successful results, return the first result (even if failed)
                first_result = results[0] if results else None
                if isinstance(first_result, TaskOutput):
                    return first_result
                else:
                    return TaskOutput(
                        task_id=task_input.id,
                        result="All agents failed",
                        confidence=0.0,
                        agents_used=[agent.agent_id for agent in agents],
                        execution_time=0.0,
                        status=TaskStatus.FAILED,
                    )
            # Combine results
            combined_result = await self._combine_results(
                successful_results, task_input
            )
            return combined_result

        finally:
            # Mark agents as available
            for agent in agents:
                agent.is_busy = False

    async def _combine_results(
        self, results: List[TaskOutput], task_input: TaskInput
    ) -> TaskOutput:
        """Combine multiple agent results into a single output."""

        if len(results) == 1:
            return results[0]

        # Calculate combined metrics
        total_confidence = sum(r.confidence for r in results) / len(results)
        total_execution_time = max(r.execution_time for r in results)
        all_agents = [agent for result in results for agent in result.agents_used]

        # Combine result text
        combined_text = f"Combined response from {len(results)} agents:\n\n"
        for i, result in enumerate(results, 1):
            combined_text += f"Agent {i} Response:\n{result.result}\n\n"

        combined_text += "Synthesis: This response incorporates insights from multiple specialized agents to provide comprehensive coverage of the task."

        return TaskOutput(
            task_id=task_input.id,
            result=combined_text,
            confidence=total_confidence,
            agents_used=all_agents,
            execution_time=total_execution_time,
            status=TaskStatus.COMPLETED,
            reasoning="Combined insights from multiple specialized agents",
        )

    async def _update_performance_metrics(self, result: TaskOutput) -> None:
        """Update performance metrics for agents involved in the task."""

        for agent_id in result.agents_used:
            if agent_id in self.agent_performance:
                perf = self.agent_performance[agent_id]
                perf.task_count += 1

                # Update success rate
                if result.status == TaskStatus.COMPLETED:
                    perf.success_rate = (
                        perf.success_rate * (perf.task_count - 1) + 1.0
                    ) / perf.task_count
                else:
                    perf.success_rate = (
                        perf.success_rate * (perf.task_count - 1)
                    ) / perf.task_count

                # Update average confidence
                perf.average_confidence = (
                    perf.average_confidence * (perf.task_count - 1) + result.confidence
                ) / perf.task_count

                # Update average execution time
                perf.average_execution_time = (
                    perf.average_execution_time * (perf.task_count - 1)
                    + result.execution_time
                ) / perf.task_count

                perf.last_updated = datetime.now(timezone.utc)
        # Update orchestration stats
        self.orchestration_stats["total_tasks"] += 1
        if result.status == TaskStatus.COMPLETED:
            self.orchestration_stats["successful_tasks"] += 1
        else:
            self.orchestration_stats["failed_tasks"] += 1

        # Update average execution time
        total_tasks = self.orchestration_stats["total_tasks"]
        current_avg = self.orchestration_stats["average_execution_time"]
        self.orchestration_stats["average_execution_time"] = (
            current_avg * (total_tasks - 1) + result.execution_time
        ) / total_tasks

    def get_orchestration_stats(self) -> Dict[str, Any]:
        """Get orchestration statistics."""
        return {
            **self.orchestration_stats,
            "active_agents": len([a for a in self.agents.values() if not a.is_busy]),
            "busy_agents": len([a for a in self.agents.values() if a.is_busy]),
            "total_agents": len(self.agents),
            "completed_tasks": len(self.completed_tasks),
            "success_rate": self.orchestration_stats["successful_tasks"]
            / max(1, self.orchestration_stats["total_tasks"]),
        }

    def get_agent_performance(self, agent_id: str) -> Optional[AgentPerformance]:
        """Get performance metrics for a specific agent."""
        return self.agent_performance.get(agent_id)

    def get_all_agent_performance(self) -> Dict[str, AgentPerformance]:
        """Get performance metrics for all agents."""
        return self.agent_performance.copy()


# Factory function
def create_adaptive_orchestrator(max_parallel_agents: int = 3) -> AdaptiveOrchestrator:
    """Create and return an adaptive orchestrator instance."""
    return AdaptiveOrchestrator(max_parallel_agents=max_parallel_agents)

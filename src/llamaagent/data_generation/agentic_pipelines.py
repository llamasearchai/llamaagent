"""
Advanced Agentic Data Generation Pipelines

Provides comprehensive data generation capabilities using multi-agent workflows
with quality control, validation, and human feedback loops.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PipelineStage(str, Enum):
    """Stages in the data generation pipeline"""

    INITIAL_GENERATION = "initial_generation"
    QUALITY_REVIEW = "quality_review"
    REFINEMENT = "refinement"
    VALIDATION = "validation"
    HUMAN_FEEDBACK = "human_feedback"
    FINAL_APPROVAL = "final_approval"
    COMPLETED = "completed"


class GenerationStrategy(str, Enum):
    """Strategies for data generation"""

    TEMPLATE_BASED = "template_based"
    LLM_BASED = "llm_based"
    HYBRID = "hybrid"
    ADVERSARIAL = "adversarial"
    COLLABORATIVE = "collaborative"


@dataclass
class GenerationTask:
    """Individual data generation task"""

    id: str
    task_type: str
    description: str
    requirements: Dict[str, Any]
    priority: int = 5
    estimated_duration: float = 1.0
    current_stage: PipelineStage = PipelineStage.INITIAL_GENERATION
    assigned_agent: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "task_type": self.task_type,
            "description": self.description,
            "requirements": self.requirements,
            "priority": self.priority,
            "estimated_duration": self.estimated_duration,
            "current_stage": self.current_stage.value,
            "assigned_agent": self.assigned_agent,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class GenerationResult:
    """Result from data generation"""

    task_id: str
    generated_data: Dict[str, Any]
    quality_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    stage: PipelineStage = PipelineStage.INITIAL_GENERATION
    generation_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "task_id": self.task_id,
            "generated_data": self.generated_data,
            "quality_score": self.quality_score,
            "metadata": self.metadata,
            "issues": self.issues,
            "stage": self.stage.value,
            "generation_time": self.generation_time,
        }


@dataclass
class PipelineConfig:
    """Configuration for data generation pipeline"""

    max_parallel_tasks: int = 5
    quality_threshold: float = 0.8
    enable_human_feedback: bool = False
    max_refinement_iterations: int = 3
    default_strategy: GenerationStrategy = GenerationStrategy.HYBRID
    agents_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "max_parallel_tasks": self.max_parallel_tasks,
            "quality_threshold": self.quality_threshold,
            "enable_human_feedback": self.enable_human_feedback,
            "max_refinement_iterations": self.max_refinement_iterations,
            "default_strategy": self.default_strategy.value,
            "agents_config": self.agents_config,
        }


class DataGenerationAgent:
    """Base class for data generation agents"""

    def __init__(self, agent_id: str, specialization: str = "general"):
        """Initialize agent"""
        self.agent_id = agent_id
        self.specialization = specialization
        self.completion_history: List[Dict[str, Any]] = []
        self.performance_metrics = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "average_quality": 0.0,
            "average_duration": 0.0,
        }

    async def generate_data(self, task: GenerationTask) -> GenerationResult:
        """Generate data for a given task"""
        start_time = asyncio.get_event_loop().time()

        try:
            # Simulate data generation
            generated_data = await self._generate_task_data(task)

            # Calculate quality score
            quality_score = await self._calculate_quality_score(generated_data, task)

            # Create result
            result = GenerationResult(
                task_id=task.id,
                generated_data=generated_data,
                quality_score=quality_score,
                metadata={
                    "agent_id": self.agent_id,
                    "specialization": self.specialization,
                    "generation_strategy": task.requirements.get("strategy", "default"),
                },
                generation_time=asyncio.get_event_loop().time() - start_time,
            )

            # Update performance metrics
            self._update_performance_metrics(result)

            return result

        except Exception as e:
            logger.error(f"Agent {self.agent_id} failed to generate data: {e}")
            return GenerationResult(
                task_id=task.id,
                generated_data={},
                quality_score=0.0,
                metadata={"error": str(e)},
                issues=[f"Generation failed: {str(e)}"],
                generation_time=asyncio.get_event_loop().time() - start_time,
            )

    async def _generate_task_data(self, task: GenerationTask) -> Dict[str, Any]:
        """Generate data based on task requirements"""
        # This would contain the actual data generation logic
        # For now, we'll simulate based on task type

        if task.task_type == "conversation":
            return await self._generate_conversation_data(task)
        elif task.task_type == "reasoning":
            return await self._generate_reasoning_data(task)
        elif task.task_type == "multimodal":
            return await self._generate_multimodal_data(task)
        else:
            return await self._generate_generic_data(task)

    async def _generate_conversation_data(self, task: GenerationTask) -> Dict[str, Any]:
        """Generate conversation data"""
        num_turns = task.requirements.get("num_turns", 3)
        topic = task.requirements.get("topic", "general")

        turns = []
        for i in range(num_turns):
            turn = {
                "turn_id": i,
                "speaker": "user" if i % 2 == 0 else "assistant",
                "text": f"Sample {topic} conversation turn {i}",
                "metadata": {"turn_type": "dialogue"},
            }
            turns.append(turn)

        return {
            "conversation_id": f"conv_{task.id}",
            "topic": topic,
            "turns": turns,
            "metadata": {"num_turns": num_turns, "generation_method": "template_based"},
        }

    async def _generate_reasoning_data(self, task: GenerationTask) -> Dict[str, Any]:
        """Generate reasoning data"""
        difficulty = task.requirements.get("difficulty", "medium")
        domain = task.requirements.get("domain", "general")

        return {
            "reasoning_id": f"reason_{task.id}",
            "domain": domain,
            "difficulty": difficulty,
            "problem": f"Sample {difficulty} {domain} reasoning problem",
            "solution": f"Step-by-step solution for {difficulty} {domain} problem",
            "steps": [
                {"step": 1, "description": "Analyze the problem"},
                {"step": 2, "description": "Identify key concepts"},
                {"step": 3, "description": "Apply reasoning rules"},
                {"step": 4, "description": "Verify solution"},
            ],
            "metadata": {
                "reasoning_type": "analytical",
                "generation_method": "structured",
            },
        }

    async def _generate_multimodal_data(self, task: GenerationTask) -> Dict[str, Any]:
        """Generate multimodal data"""
        modalities = task.requirements.get("modalities", ["text", "image"])

        return {
            "multimodal_id": f"multi_{task.id}",
            "modalities": modalities,
            "text_component": "Sample text for multimodal task",
            "image_component": {
                "description": "Sample image description",
                "metadata": {"format": "description_only"},
            },
            "alignment": {"text_image_correlation": 0.8, "coherence_score": 0.9},
            "metadata": {"generation_method": "multimodal_template"},
        }

    async def _generate_generic_data(self, task: GenerationTask) -> Dict[str, Any]:
        """Generate generic data"""
        return {
            "data_id": f"generic_{task.id}",
            "task_type": task.task_type,
            "description": task.description,
            "generated_content": f"Generated content for {task.task_type}",
            "metadata": {"generation_method": "generic_template"},
        }

    async def _calculate_quality_score(
        self, generated_data: Dict[str, Any], task: GenerationTask
    ) -> float:
        """Calculate quality score for generated data"""
        score = 0.8  # Base score

        # Check completeness
        if "metadata" in generated_data:
            score += 0.1

        # Check structure
        if len(generated_data) >= 3:
            score += 0.1

        # Task-specific quality checks
        if task.task_type == "conversation":
            if "turns" in generated_data and len(generated_data["turns"]) > 0:
                score += 0.1
        elif task.task_type == "reasoning":
            if "steps" in generated_data and len(generated_data["steps"]) > 0:
                score += 0.1

        return min(1.0, score)

    def _update_performance_metrics(self, result: GenerationResult) -> None:
        """Update agent performance metrics"""
        self.performance_metrics["total_tasks"] += 1
        if result.quality_score > 0.7:
            self.performance_metrics["successful_tasks"] += 1

        # Update averages
        total_tasks = self.performance_metrics["total_tasks"]
        self.performance_metrics["average_quality"] = (
            self.performance_metrics["average_quality"] * (total_tasks - 1)
            + result.quality_score
        ) / total_tasks
        self.performance_metrics["average_duration"] = (
            self.performance_metrics["average_duration"] * (total_tasks - 1)
            + result.generation_time
        ) / total_tasks

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            "agent_id": self.agent_id,
            "specialization": self.specialization,
            "metrics": self.performance_metrics,
            "success_rate": (
                self.performance_metrics["successful_tasks"]
                / max(1, self.performance_metrics["total_tasks"])
            ),
        }


class QualityReviewAgent:
    """Agent specialized in quality review and validation"""

    def __init__(self, agent_id: str = "quality_reviewer"):
        """Initialize quality review agent"""
        self.agent_id = agent_id
        self.review_history: List[Dict[str, Any]] = []

    async def review_data(self, result: GenerationResult) -> Dict[str, Any]:
        """Review and validate generated data"""
        review_result = {
            "task_id": result.task_id,
            "original_quality": result.quality_score,
            "review_score": 0.0,
            "issues": [],
            "suggestions": [],
            "approved": False,
        }

        # Perform quality checks
        quality_checks = await self._perform_quality_checks(result)
        review_result["review_score"] = quality_checks["score"]
        review_result["issues"] = quality_checks["issues"]
        review_result["suggestions"] = quality_checks["suggestions"]
        review_result["approved"] = quality_checks["score"] >= 0.8

        self.review_history.append(review_result)

        return review_result

    async def _perform_quality_checks(self, result: GenerationResult) -> Dict[str, Any]:
        """Perform detailed quality checks"""
        score = result.quality_score
        issues = []
        suggestions = []

        # Check data completeness
        if not result.generated_data:
            score -= 0.3
            issues.append("Generated data is empty")
            suggestions.append("Regenerate with proper data structure")

        # Check metadata presence
        if "metadata" not in result.generated_data:
            score -= 0.1
            issues.append("Missing metadata")
            suggestions.append("Add metadata for better traceability")

        # Check for required fields based on task type
        if result.task_id.startswith("conv_"):
            if "turns" not in result.generated_data:
                score -= 0.2
                issues.append("Conversation missing turns")
        elif result.task_id.startswith("reason_"):
            if "steps" not in result.generated_data:
                score -= 0.2
                issues.append("Reasoning missing solution steps")

        # Check content quality
        if result.generation_time > 10.0:
            score -= 0.1
            issues.append("Generation took too long")
            suggestions.append("Optimize generation process")

        return {
            "score": max(0.0, min(1.0, score)),
            "issues": issues,
            "suggestions": suggestions,
        }


class AgenticDataPipeline:
    """Main orchestrator for agentic data generation pipeline"""

    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize pipeline"""
        self.config = config or PipelineConfig()
        self.agents: Dict[str, DataGenerationAgent] = {}
        self.quality_reviewer = QualityReviewAgent()
        self.task_queue: List[GenerationTask] = []
        self.results: Dict[str, GenerationResult] = {}
        self.active_tasks: Dict[str, GenerationTask] = {}

        # Initialize default agents
        self._initialize_default_agents()

    def _initialize_default_agents(self) -> None:
        """Initialize default set of agents"""
        agent_configs = [
            ("conversation_agent", "conversation"),
            ("reasoning_agent", "reasoning"),
            ("multimodal_agent", "multimodal"),
            ("general_agent", "general"),
        ]

        for agent_id, specialization in agent_configs:
            agent = DataGenerationAgent(agent_id, specialization)
            self.agents[agent_id] = agent

    def add_agent(self, agent: DataGenerationAgent) -> None:
        """Add an agent to the pipeline"""
        self.agents[agent.agent_id] = agent
        logger.info(
            f"Added agent {agent.agent_id} with specialization {agent.specialization}"
        )

    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from the pipeline"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Removed agent {agent_id}")

    async def submit_task(self, task: GenerationTask) -> str:
        """Submit a task to the pipeline"""
        self.task_queue.append(task)
        logger.info(f"Submitted task {task.id} of type {task.task_type}")
        return task.id

    async def process_pipeline(self) -> Dict[str, Any]:
        """Process the entire pipeline"""
        pipeline_start = asyncio.get_event_loop().time()

        logger.info(f"Starting pipeline with {len(self.task_queue)} tasks")

        # Process tasks in parallel
        semaphore = asyncio.Semaphore(self.config.max_parallel_tasks)

        async def process_task(task: GenerationTask) -> GenerationResult:
            async with semaphore:
                return await self._process_single_task(task)

        # Create tasks for parallel processing
        task_coroutines = [process_task(task) for task in self.task_queue]

        # Wait for all tasks to complete
        results = await asyncio.gather(*task_coroutines, return_exceptions=True)

        # Process results
        completed_tasks = 0
        failed_tasks = 0
        total_quality = 0.0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_tasks += 1
                logger.error(f"Task {self.task_queue[i].id} failed: {result}")
            else:
                completed_tasks += 1
                total_quality += result.quality_score
                self.results[result.task_id] = result

        # Clear processed tasks
        self.task_queue.clear()

        pipeline_duration = asyncio.get_event_loop().time() - pipeline_start

        summary = {
            "pipeline_duration": pipeline_duration,
            "total_tasks": len(results),
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "average_quality": total_quality / max(1, completed_tasks),
            "throughput": len(results) / pipeline_duration,
            "agent_performance": {
                agent_id: agent.get_performance_summary()
                for agent_id, agent in self.agents.items()
            },
        }

        logger.info(
            f"Pipeline completed: {completed_tasks}/{len(results)} tasks successful"
        )
        return summary

    async def _process_single_task(self, task: GenerationTask) -> GenerationResult:
        """Process a single task through the pipeline stages"""
        current_result = None

        try:
            # Stage 1: Initial Generation
            task.current_stage = PipelineStage.INITIAL_GENERATION
            agent = self._select_agent(task)
            if not agent:
                raise ValueError(f"No suitable agent found for task {task.id}")

            task.assigned_agent = agent.agent_id
            current_result = await agent.generate_data(task)

            # Stage 2: Quality Review
            task.current_stage = PipelineStage.QUALITY_REVIEW
            review_result = await self.quality_reviewer.review_data(current_result)

            # Stage 3: Refinement (if needed)
            if not review_result["approved"] and current_result.quality_score > 0.5:
                task.current_stage = PipelineStage.REFINEMENT
                current_result = await self._refine_result(
                    current_result, review_result
                )

            # Stage 4: Final Validation
            task.current_stage = PipelineStage.VALIDATION
            current_result = await self._validate_result(current_result)

            # Stage 5: Human Feedback (if enabled)
            if self.config.enable_human_feedback:
                task.current_stage = PipelineStage.HUMAN_FEEDBACK
                current_result = await self._collect_human_feedback(current_result)

            # Stage 6: Final Approval
            task.current_stage = PipelineStage.FINAL_APPROVAL
            current_result.stage = PipelineStage.COMPLETED

            return current_result

        except Exception as e:
            logger.error(f"Failed to process task {task.id}: {e}")
            if current_result:
                current_result.issues.append(f"Pipeline error: {str(e)}")
                current_result.quality_score = 0.0
                return current_result
            else:
                return GenerationResult(
                    task_id=task.id,
                    generated_data={},
                    quality_score=0.0,
                    issues=[f"Pipeline error: {str(e)}"],
                )

    def _select_agent(self, task: GenerationTask) -> Optional[DataGenerationAgent]:
        """Select the best agent for a given task"""
        # First, try to find an agent with matching specialization
        for agent in self.agents.values():
            if agent.specialization == task.task_type:
                return agent

        # If no specialized agent, use general agent
        for agent in self.agents.values():
            if agent.specialization == "general":
                return agent

        # If no general agent, use any available agent
        if self.agents:
            return list(self.agents.values())[0]

        return None

    async def _refine_result(
        self, result: GenerationResult, review_result: Dict[str, Any]
    ) -> GenerationResult:
        """Refine the result based on review feedback"""
        # Simple refinement - in practice, this would use more sophisticated methods
        result.quality_score += 0.1  # Slight improvement
        result.issues.extend(review_result["issues"])
        result.metadata["refinement_applied"] = True
        result.metadata["review_feedback"] = review_result["suggestions"]

        return result

    async def _validate_result(self, result: GenerationResult) -> GenerationResult:
        """Final validation of the result"""
        # Perform final checks
        if result.quality_score < self.config.quality_threshold:
            result.issues.append("Quality below threshold")

        result.metadata["validation_complete"] = True
        return result

    async def _collect_human_feedback(
        self, result: GenerationResult
    ) -> GenerationResult:
        """Collect human feedback (simulated)"""
        # In practice, this would involve actual human review
        # For now, we'll simulate feedback
        simulated_feedback = {
            "rating": random.uniform(0.7, 1.0),
            "comments": ["Good quality", "Meets requirements"],
            "suggestions": ["Minor improvements possible"],
        }

        result.metadata["human_feedback"] = simulated_feedback
        result.quality_score = (result.quality_score + simulated_feedback["rating"]) / 2

        return result

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            "config": self.config.to_dict(),
            "agents": {
                agent_id: agent.get_performance_summary()
                for agent_id, agent in self.agents.items()
            },
            "queued_tasks": len(self.task_queue),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.results),
            "quality_distribution": self._calculate_quality_distribution(),
        }

    def _calculate_quality_distribution(self) -> Dict[str, int]:
        """Calculate quality score distribution"""
        distribution = {"high": 0, "medium": 0, "low": 0}

        for result in self.results.values():
            if result.quality_score >= 0.8:
                distribution["high"] += 1
            elif result.quality_score >= 0.6:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1

        return distribution

    async def export_results(self, output_path: str) -> None:
        """Export pipeline results to file"""
        export_data = {
            "pipeline_config": self.config.to_dict(),
            "results": [result.to_dict() for result in self.results.values()],
            "agent_performance": {
                agent_id: agent.get_performance_summary()
                for agent_id, agent in self.agents.items()
            },
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported pipeline results to {output_path}")


async def create_sample_pipeline() -> AgenticDataPipeline:
    """Create a sample pipeline for demonstration"""
    config = PipelineConfig(
        max_parallel_tasks=3,
        quality_threshold=0.8,
        enable_human_feedback=False,
        max_refinement_iterations=2,
    )

    pipeline = AgenticDataPipeline(config)

    # Add some sample tasks
    tasks = [
        GenerationTask(
            id="conv_001",
            task_type="conversation",
            description="Generate a conversation about AI safety",
            requirements={"num_turns": 4, "topic": "AI safety"},
        ),
        GenerationTask(
            id="reason_001",
            task_type="reasoning",
            description="Generate a mathematical reasoning problem",
            requirements={"difficulty": "medium", "domain": "mathematics"},
        ),
        GenerationTask(
            id="multi_001",
            task_type="multimodal",
            description="Generate multimodal content",
            requirements={"modalities": ["text", "image"]},
        ),
    ]

    for task in tasks:
        await pipeline.submit_task(task)

    return pipeline

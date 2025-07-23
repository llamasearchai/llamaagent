"""
Advanced reasoning chains agent with O1-style thinking patterns.

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from ..llm.factory import LLMFactory
from ..llm.messages import LLMMessage, LLMResponse
from ..types import TaskInput, TaskOutput, TaskResult, TaskStatus
from .base import AgentConfig, BaseAgent

logger = logging.getLogger(__name__)


class ThinkingPattern(Enum):
    """Different thinking patterns for advanced reasoning."""

    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHTS = "tree_of_thoughts"
    SELF_REFLECTION = "self_reflection"
    ITERATIVE_REFINEMENT = "iterative_refinement"
    MULTI_PERSPECTIVE = "multi_perspective"


@dataclass
class ReasoningStep:
    """A single step in the reasoning process."""

    step_id: int
    pattern: ThinkingPattern
    prompt: str
    response: str
    confidence: float
    timestamp: datetime


class AdvancedReasoningAgent(BaseAgent):
    """
    Advanced Reasoning Agent with O1-style thinking patterns.

    Features:
    - Chain-of-thought reasoning
    - Tree-of-thoughts exploration
    - Self-reflection and correction
    - Iterative refinement
    - Multi-perspective analysis
    """

    def __init__(
        self,
        name: str = "AdvancedReasoningAgent",
        thinking_pattern: ThinkingPattern = ThinkingPattern.CHAIN_OF_THOUGHT,
        max_iterations: int = 3,
        confidence_threshold: float = 0.85,
        config: Optional[AgentConfig] = None,
        **kwargs: Any,
    ):
        if config is None:
            config = AgentConfig(
                name=name, llm_provider=self._create_reasoning_provider(), **kwargs
            )

        super().__init__(config=config)
        self.thinking_pattern = thinking_pattern
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        self.reasoning_history: List[ReasoningStep] = []

        logger.info(f"Initialized {name} with pattern: {thinking_pattern.value}")

    def _create_reasoning_provider(self):
        """Create a specialized provider for reasoning tasks."""
        factory = LLMFactory()
        return factory.create_provider(
            provider_name="openai", model="gpt-4o", temperature=0.1, max_tokens=8192
        )

    async def execute_task(self, task_input: TaskInput) -> TaskOutput:
        """Execute reasoning task using the configured thinking pattern."""
        try:
            logger.info(f"Executing reasoning task: {task_input.id}")

            # Execute based on thinking pattern
            if self.thinking_pattern == ThinkingPattern.CHAIN_OF_THOUGHT:
                response = await self._chain_of_thought(task_input)
            elif self.thinking_pattern == ThinkingPattern.TREE_OF_THOUGHTS:
                response = await self._tree_of_thoughts(task_input)
            elif self.thinking_pattern == ThinkingPattern.SELF_REFLECTION:
                response = await self._self_reflection(task_input)
            elif self.thinking_pattern == ThinkingPattern.ITERATIVE_REFINEMENT:
                response = await self._iterative_refinement(task_input)
            elif self.thinking_pattern == ThinkingPattern.MULTI_PERSPECTIVE:
                response = await self._multi_perspective(task_input)
            else:
                response = await self._chain_of_thought(task_input)

            # Create task result
            task_result = TaskResult(
                success=True,
                result=response,
                model=(
                    self.config.llm_provider.model
                    if self.config.llm_provider
                    else "unknown"
                ),
                metadata={"reasoning_steps": len(self.reasoning_history)},
            )

            return TaskOutput(
                task_id=task_input.id, status=TaskStatus.COMPLETED, result=task_result
            )

        except Exception as e:
            logger.error(f"Reasoning task failed: {e}")
            return TaskOutput(
                task_id=task_input.id,
                status=TaskStatus.FAILED,
                result=TaskResult(success=False, error=str(e)),
            )

    async def _chain_of_thought(self, task_input: TaskInput) -> LLMResponse:
        """Chain-of-thought reasoning implementation."""

        prompt = f"""
Let me think through this step by step:

{task_input.prompt if hasattr(task_input, 'prompt') else str(task_input.data)}

I'll work through this systematically:

Step 1: Understanding the problem
Let me first make sure I understand what's being asked...

Step 2: Breaking down the components
I need to identify the key elements...

Step 3: Analyzing relationships
How do these elements connect...

Step 4: Working towards a solution
Based on my analysis...

Step 5: Verification and conclusion
Let me verify my reasoning...
"""

        response = await self.config.llm_provider.complete(
            [LLMMessage(role="user", content=prompt)]
        )

        self._add_reasoning_step(
            step_id=1,
            pattern=ThinkingPattern.CHAIN_OF_THOUGHT,
            prompt=prompt,
            response=response.content,
            confidence=0.8,
        )

        return response

    async def _tree_of_thoughts(self, task_input: TaskInput) -> LLMResponse:
        """Tree-of-thoughts reasoning with multiple branches."""

        # Generate initial branches
        branches = []

        for i in range(3):  # Generate 3 initial branches
            branch_prompt = f"""
Consider this problem from approach {i+1}:

{task_input.prompt if hasattr(task_input, 'prompt') else str(task_input.data)}

My initial approach is:
[Think differently about this problem and provide a unique perspective]
"""

            branch_response = await self.config.llm_provider.complete(
                [LLMMessage(role="user", content=branch_prompt)]
            )

            branches.append(branch_response)

            self._add_reasoning_step(
                step_id=i + 1,
                pattern=ThinkingPattern.TREE_OF_THOUGHTS,
                prompt=branch_prompt,
                response=branch_response.content,
                confidence=0.7,
            )

        # Select and expand the best branch
        best_branch = await self._select_best_branch(branches)

        # Expand the best branch
        expansion_prompt = f"""
My initial approach was:
{best_branch.content}

Now let me develop this further with more detailed reasoning and reach a comprehensive conclusion:
"""

        expanded_response = await self.config.llm_provider.complete(
            [LLMMessage(role="user", content=expansion_prompt)]
        )

        return expanded_response

    async def _self_reflection(self, task_input: TaskInput) -> LLMResponse:
        """Self-reflection reasoning with correction."""

        # Initial response
        initial_prompt = f"""
{task_input.prompt if hasattr(task_input, 'prompt') else str(task_input.data)}
"""

        initial_response = await self.config.llm_provider.complete(
            [LLMMessage(role="user", content=initial_prompt)]
        )

        # Self-reflection
        reflection_prompt = f"""
I just provided this response:
{initial_response.content}

Let me reflect on this:
1. Is my reasoning sound?
2. Did I miss anything important?
3. Are there any errors or weak points?
4. How can I improve this response?

Based on my reflection, here's my improved response:
"""

        reflected_response = await self.config.llm_provider.complete(
            [LLMMessage(role="user", content=reflection_prompt)]
        )

        return reflected_response

    async def _iterative_refinement(self, task_input: TaskInput) -> LLMResponse:
        """Iterative refinement with multiple improvement cycles."""

        # Initial response
        current_response = await self.config.llm_provider.complete(
            [
                LLMMessage(
                    role="user",
                    content=(
                        task_input.prompt
                        if hasattr(task_input, 'prompt')
                        else str(task_input.data)
                    ),
                )
            ]
        )

        # Iteratively refine
        for iteration in range(self.max_iterations - 1):
            refinement_prompt = f"""
Current response:
{current_response.content}

How can this be improved? Provide a refined version that:
1. Addresses any gaps or weaknesses
2. Adds more depth and insight
3. Improves clarity and precision
4. Ensures completeness

Refined response:
"""

            refined_response = await self.config.llm_provider.complete(
                [LLMMessage(role="user", content=refinement_prompt)]
            )

            # Check confidence (simplified)
            confidence = 0.6 + (iteration * 0.1)  # Gradually increase confidence

            self._add_reasoning_step(
                step_id=iteration + 1,
                pattern=ThinkingPattern.ITERATIVE_REFINEMENT,
                prompt=refinement_prompt,
                response=refined_response.content,
                confidence=confidence,
            )

            current_response = refined_response

            # Check if we've reached good quality
            if confidence >= self.confidence_threshold:
                break

        return current_response

    async def _multi_perspective(self, task_input: TaskInput) -> LLMResponse:
        """Multi-perspective analysis with synthesis."""

        perspectives = ["analytical", "creative", "critical", "practical"]
        perspective_responses = []

        for i, perspective in enumerate(perspectives):
            perspective_prompt = f"""
Analyze this from a {perspective} perspective:

{task_input.prompt if hasattr(task_input, 'prompt') else str(task_input.data)}

Focus on insights unique to this viewpoint:
"""

            response = await self.config.llm_provider.complete(
                [LLMMessage(role="user", content=perspective_prompt)]
            )

            perspective_responses.append(response)

            self._add_reasoning_step(
                step_id=i + 1,
                pattern=ThinkingPattern.MULTI_PERSPECTIVE,
                prompt=perspective_prompt,
                response=response.content,
                confidence=0.7,
            )

        # Synthesize perspectives
        synthesis_prompt = f"""
I've analyzed this from multiple perspectives:

{chr(10).join([f"Perspective {i+1}: {resp.content}" for i, resp in enumerate(perspective_responses)])}

Now, synthesize these perspectives into a comprehensive, balanced response:
"""

        synthesized_response = await self.config.llm_provider.complete(
            [LLMMessage(role="user", content=synthesis_prompt)]
        )

        return synthesized_response

    async def _select_best_branch(self, branches: List[LLMResponse]) -> LLMResponse:
        """Select the best branch from multiple options."""

        # Simple heuristic: select the longest response (assuming more detailed)
        # In practice, this could use more sophisticated scoring
        return max(branches, key=lambda b: len(b.content))

    def _add_reasoning_step(
        self,
        step_id: int,
        pattern: ThinkingPattern,
        prompt: str,
        response: str,
        confidence: float,
    ) -> None:
        """Add a reasoning step to the history."""
        step = ReasoningStep(
            step_id=step_id,
            pattern=pattern,
            prompt=prompt,
            response=response,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc),
        )
        self.reasoning_history.append(step)

    def get_reasoning_summary(self) -> Dict[str, Any]:
        """Get summary of reasoning process."""
        if not self.reasoning_history:
            return {}

        return {
            "total_steps": len(self.reasoning_history),
            "pattern": self.thinking_pattern.value,
            "avg_confidence": sum(step.confidence for step in self.reasoning_history)
            / len(self.reasoning_history),
            "duration_seconds": (
                (
                    self.reasoning_history[-1].timestamp
                    - self.reasoning_history[0].timestamp
                ).total_seconds()
                if len(self.reasoning_history) > 1
                else 0
            ),
            "steps": [step.__dict__ for step in self.reasoning_history],
        }


# Convenience function for creating reasoning agents
def create_reasoning_agent(
    name: str = "ReasoningAgent",
    pattern: ThinkingPattern = ThinkingPattern.CHAIN_OF_THOUGHT,
    **kwargs: Any,
) -> AdvancedReasoningAgent:
    """Create a reasoning agent with specified pattern."""
    return AdvancedReasoningAgent(name=name, thinking_pattern=pattern, **kwargs)


if __name__ == "__main__":
    # Example usage
    async def main():
        agent = create_reasoning_agent(pattern=ThinkingPattern.CHAIN_OF_THOUGHT)

        task = TaskInput(
            id="test_reasoning",
            prompt="Explain the concept of machine learning and its applications",
            data={},
        )

        result = await agent.execute_task(task)
        print(f"Result: {result}")

        summary = agent.get_reasoning_summary()
        print(f"Reasoning Summary: {summary}")

    asyncio.run(main())

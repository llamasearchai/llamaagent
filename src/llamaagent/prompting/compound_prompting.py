"""
Compound Prompting Strategies

Implements advanced compound prompting techniques that combine multiple
prompting strategies for enhanced performance.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

from ..llm.messages import LLMMessage
from ..llm.providers.base_provider import BaseLLMProvider

logger = logging.getLogger(__name__)


class PromptStrategy(Enum):
    """Available prompting strategies"""

    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHT = "tree_of_thought"
    DECOMPOSITION = "decomposition"
    ANALOGY = "analogy"
    ROLE_BASED = "role_based"
    ITERATIVE_REFINEMENT = "iterative_refinement"
    FEW_SHOT = "few_shot"
    ZERO_SHOT = "zero_shot"


class CombinationStrategy(Enum):
    """Strategy combination methods"""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    WEIGHTED = "weighted"
    ADAPTIVE = "adaptive"


@dataclass
class PromptResult:
    """Result from a single prompting strategy"""

    strategy: PromptStrategy
    content: str
    confidence: float
    reasoning_path: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompoundResult:
    """Result from compound prompting"""

    final_answer: str
    confidence: float
    strategy_results: Dict[PromptStrategy, PromptResult] = field(default_factory=dict)
    reasoning_trace: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompoundPromptConfig:
    """Configuration for compound prompting"""

    strategies: List[PromptStrategy] = field(
        default_factory=lambda: [PromptStrategy.CHAIN_OF_THOUGHT]
    )
    combination_method: CombinationStrategy = CombinationStrategy.SEQUENTIAL
    weights: Dict[PromptStrategy, float] = field(default_factory=dict)
    max_iterations: int = 3
    confidence_threshold: float = 0.8
    examples: List[Dict[str, str]] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)


class BasePromptStrategy(ABC):
    """Base class for prompt strategies"""

    def __init__(self, llm_provider: BaseLLMProvider):
        self.llm_provider = llm_provider

    @abstractmethod
    async def generate(
        self, query: str, context: Optional[Dict[str, Any]] = None, **kwargs
    ) -> PromptResult:
        """Generate response using this strategy"""
        pass

    async def _get_llm_response(self, prompt: str, **kwargs) -> str:
        """Get response from LLM"""
        message = LLMMessage(role="user", content=prompt)
        response = await self.llm_provider.complete([message], **kwargs)
        return response.content


class ChainOfThoughtStrategy(BasePromptStrategy):
    """Chain-of-thought prompting strategy"""

    async def generate(
        self, query: str, context: Optional[Dict[str, Any]] = None, **kwargs
    ) -> PromptResult:
        """Generate chain-of-thought response"""
        prompt = f"""Think step by step to solve this problem.

Problem: {query}

Let's work through this step by step:
1. First, identify what is being asked
2. Break down the problem into smaller parts
3. Solve each part systematically
4. Combine the solutions to get the final answer

Step-by-step reasoning:"""

        response = await self._get_llm_response(prompt, **kwargs)

        # Extract reasoning steps
        reasoning_steps = []
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('Step', '1.', '2.', '3.', '4.', '5.'):
                reasoning_steps.append(line)

        return PromptResult(
            strategy=PromptStrategy.CHAIN_OF_THOUGHT,
            content=response,
            confidence=0.8,
            reasoning_path=reasoning_steps,
        )


class DecompositionStrategy(BasePromptStrategy):
    """Problem decomposition strategy"""

    async def generate(
        self, query: str, context: Optional[Dict[str, Any]] = None, **kwargs
    ) -> PromptResult:
        """Generate decomposition response"""
        prompt = f"""Break down this complex problem into smaller, manageable parts.

Problem: {query}

Decomposition approach:
1. Identify the main components of the problem
2. Break each component into sub-problems
3. Solve each sub-problem independently
4. Combine the solutions

Analysis:"""

        response = await self._get_llm_response(prompt, **kwargs)

        return PromptResult(
            strategy=PromptStrategy.DECOMPOSITION,
            content=response,
            confidence=0.75,
            reasoning_path=["Problem decomposition completed"],
        )


class AnalogyStrategy(BasePromptStrategy):
    """Analogy-based prompting strategy"""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        domain_analogies: Optional[List[str]] = None,
    ):
        super().__init__(llm_provider)
        self.domain_analogies = domain_analogies or [
            "cooking recipe",
            "building construction",
            "solving a puzzle",
            "scientific experiment",
            "sports strategy",
        ]

    async def generate(
        self, query: str, context: Optional[Dict[str, Any]] = None, **kwargs
    ) -> PromptResult:
        """Generate analogy-based response"""
        analogy = self.domain_analogies[0]  # Simple selection

        prompt = f"""Think about this problem like {analogy}.

Problem: {query}

Using the analogy of {analogy}, let's approach this:
- How would you handle a similar situation in {analogy}?
- What principles from {analogy} apply here?
- What steps would you take in that context?

Analogy-based solution:"""

        response = await self._get_llm_response(prompt, **kwargs)

        return PromptResult(
            strategy=PromptStrategy.ANALOGY,
            content=response,
            confidence=0.7,
            reasoning_path=[f"Applied {analogy} analogy"],
            metadata={"analogy_used": analogy},
        )


class RoleBasedStrategy(BasePromptStrategy):
    """Role-based prompting strategy"""

    def __init__(self, llm_provider: BaseLLMProvider, role: str = "expert consultant"):
        super().__init__(llm_provider)
        self.role = role

    async def generate(
        self, query: str, context: Optional[Dict[str, Any]] = None, **kwargs
    ) -> PromptResult:
        """Generate role-based response"""
        prompt = f"""You are an experienced {self.role}. Please provide your expert analysis.

Question: {query}

As a {self.role}, I need to consider:
- What expertise and experience I bring to this problem
- What methodologies and frameworks are most appropriate
- What potential challenges or considerations should be addressed

Expert analysis:"""

        response = await self._get_llm_response(prompt, **kwargs)

        return PromptResult(
            strategy=PromptStrategy.ROLE_BASED,
            content=response,
            confidence=0.85,
            reasoning_path=[f"Applied {self.role} perspective"],
            metadata={"role": self.role},
        )


class IterativeRefinementStrategy(BasePromptStrategy):
    """Iterative refinement prompting strategy"""

    async def generate(
        self, query: str, context: Optional[Dict[str, Any]] = None, **kwargs
    ) -> PromptResult:
        """Generate iterative refinement response"""
        max_iterations = context.get("max_iterations", 3) if context else 3
        current_answer = ""

        for iteration in range(max_iterations):
            if iteration == 0:
                prompt = f"""Problem: {query}

Provide your initial solution:"""
            else:
                prompt = f"""Problem: {query}

Previous solution (iteration {iteration}): {current_answer}

Please refine and improve your solution:
1. Identify any errors or weaknesses
2. Consider alternative approaches
3. Provide an improved solution"""

            response = await self._get_llm_response(prompt, **kwargs)
            current_answer = response

        confidence = 0.6 + (
            0.1 * min(iteration + 1, 4)
        )  # Increase confidence with iterations

        return PromptResult(
            strategy=PromptStrategy.ITERATIVE_REFINEMENT,
            content=current_answer,
            confidence=confidence,
            reasoning_path=[
                f"Iteration {i+1} completed" for i in range(max_iterations)
            ],
            metadata={"iterations": max_iterations},
        )


class CompoundPromptEngine:
    """Main engine for compound prompting"""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        config: Optional[CompoundPromptConfig] = None,
    ):
        self.llm_provider = llm_provider
        self.config = config or CompoundPromptConfig()

        # Initialize strategy instances
        self.strategy_instances = {
            PromptStrategy.CHAIN_OF_THOUGHT: ChainOfThoughtStrategy(llm_provider),
            PromptStrategy.DECOMPOSITION: DecompositionStrategy(llm_provider),
            PromptStrategy.ANALOGY: AnalogyStrategy(llm_provider),
            PromptStrategy.ROLE_BASED: RoleBasedStrategy(llm_provider),
            PromptStrategy.ITERATIVE_REFINEMENT: IterativeRefinementStrategy(
                llm_provider
            ),
        }

    async def generate(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> CompoundResult:
        """Generate response using compound prompting"""
        if self.config.combination_method == CombinationStrategy.SEQUENTIAL:
            return await self._sequential_combination(query, context)
        elif self.config.combination_method == CombinationStrategy.PARALLEL:
            return await self._parallel_combination(query, context)
        elif self.config.combination_method == CombinationStrategy.WEIGHTED:
            return await self._weighted_combination(query, context)
        elif self.config.combination_method == CombinationStrategy.ADAPTIVE:
            return await self._adaptive_combination(query, context)
        else:
            # Default to sequential
            return await self._sequential_combination(query, context)

    async def _sequential_combination(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> CompoundResult:
        """Sequential application of strategies"""
        current_query = query
        current_context = context or {}
        strategy_results = {}
        reasoning_trace = []

        for strategy in self.config.strategies:
            if strategy in self.strategy_instances:
                result = await self.strategy_instances[strategy].generate(
                    current_query, current_context
                )
                strategy_results[strategy] = result
                current_query = f"Based on the reasoning: {result.content}"

                if result.reasoning_path:
                    reasoning_trace.extend(result.reasoning_path)
                else:
                    reasoning_trace.append(f"{strategy.value}: Completed")

        # Synthesize final answer
        synthesis_prompt = (
            "Given the following analyses, provide a comprehensive final answer:\n\n"
        )

        for strategy, result in strategy_results.items():
            synthesis_prompt += f"{strategy.value} approach:\n{result.content}\n\n"

        synthesis_prompt += "Synthesized final answer:"

        final_response = await self._get_llm_response(synthesis_prompt)

        return CompoundResult(
            final_answer=final_response,
            confidence=0.9,
            strategy_results=strategy_results,
            reasoning_trace=reasoning_trace,
        )

    async def _parallel_combination(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> CompoundResult:
        """Parallel application of strategies"""
        tasks = []
        strategy_order = []

        for strategy in self.config.strategies:
            if strategy in self.strategy_instances:
                task = self.strategy_instances[strategy].generate(query, context)
                tasks.append(task)
                strategy_order.append(strategy)

        # Execute all strategies in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        strategy_results = {}
        reasoning_trace = []
        confidences = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Strategy {strategy_order[i]} failed: {result}")
                continue

            strategy_results[strategy_order[i]] = result
            confidences.append(result.confidence)

            if result.reasoning_path:
                reasoning_trace.extend(result.reasoning_path)
            else:
                reasoning_trace.append(f"{strategy_order[i].value}: Completed")

        overall_confidence = (
            np.mean(confidences) if confidences and NUMPY_AVAILABLE else 0.5
        )

        # Combine results
        combined_content = "\n\n".join(
            [
                f"{strategy.value}: {result.content}"
                for strategy, result in strategy_results.items()
            ]
        )

        return CompoundResult(
            final_answer=combined_content,
            confidence=overall_confidence,
            strategy_results=strategy_results,
            reasoning_trace=reasoning_trace,
        )

    async def _weighted_combination(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> CompoundResult:
        """Weighted combination of strategies"""
        # Execute strategies in parallel
        result = await self._parallel_combination(query, context)

        # Apply weights to results
        weighted_responses = []
        total_weight = 0

        for strategy, strategy_result in result.strategy_results.items():
            weight = self.config.weights.get(strategy, 1.0)
            weighted_responses.append(
                {
                    "content": strategy_result.content,
                    "weight": weight,
                    "confidence": strategy_result.confidence * weight,
                }
            )
            total_weight += weight

        # Normalize weights and create final answer
        if total_weight > 0:
            final_answer = ""
            for response in weighted_responses:
                normalized_weight = response["weight"] / total_weight
                final_answer += (
                    f"Weight {normalized_weight:.2f}: {response['content']}\n\n"
                )

            # Update result with weighted final answer
            result.final_answer = final_answer
            result.confidence = sum(r["confidence"] for r in weighted_responses) / len(
                weighted_responses
            )
            result.metadata["weights_applied"] = True

        return result

    async def _adaptive_combination(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> CompoundResult:
        """Adaptive strategy selection based on query characteristics"""
        # Analyze query to determine best strategies
        query_analysis = await self._analyze_query(query)

        # Select strategies based on analysis
        selected_strategies = self._select_strategies(query_analysis)

        # Update config temporarily
        original_strategies = self.config.strategies
        self.config.strategies = selected_strategies

        # Execute with selected strategies
        if query_analysis.get("complexity") == "high":
            result = await self._sequential_combination(query, context)
        else:
            result = await self._parallel_combination(query, context)

        # Restore original configuration
        self.config.strategies = original_strategies

        return result

    async def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query characteristics"""
        analysis_prompt = f"""Analyze this query: {query}

Determine:
1. Complexity level (low/medium/high)
2. Type (factual/reasoning/creative/analytical)
3. Domain (general/technical/mathematical/etc.)
4. Best reasoning approaches

Provide analysis in JSON format."""

        response = await self._get_llm_response(analysis_prompt)

        # Parse response (simplified)
        try:
            return json.loads(response)
        except:
            return {
                "complexity": "medium",
                "type": "reasoning",
                "domain": "general",
                "approaches": ["chain_of_thought"],
            }

    def _select_strategies(
        self, query_analysis: Dict[str, Any]
    ) -> List[PromptStrategy]:
        """Select strategies based on query analysis"""
        selected = []

        # Map analysis to strategies
        if query_analysis.get("complexity") == "high":
            selected.extend(
                [PromptStrategy.DECOMPOSITION, PromptStrategy.CHAIN_OF_THOUGHT]
            )

        if query_analysis.get("type") in ["creative", "analytical"]:
            selected.append(PromptStrategy.ANALOGY)

        if not selected:
            selected.append(PromptStrategy.CHAIN_OF_THOUGHT)

        return selected

    async def _get_llm_response(self, prompt: str, **kwargs) -> str:
        """Get response from LLM"""
        message = LLMMessage(role="user", content=prompt)
        response = await self.llm_provider.complete([message], **kwargs)
        return response.content

    def add_strategy(
        self, strategy: PromptStrategy, instance: BasePromptStrategy
    ) -> 'CompoundPromptEngine':
        """Add custom strategy instance"""
        self.strategy_instances[strategy] = instance
        if strategy not in self.config.strategies:
            self.config.strategies.append(strategy)
        return self

    def add_constraints(self, constraints: List[str]) -> 'CompoundPromptEngine':
        """Add constraints to the configuration"""
        self.config.constraints.extend(constraints)
        return self

    def add_examples(self, examples: List[Dict[str, str]]) -> 'CompoundPromptEngine':
        """Add examples to the configuration"""
        self.config.examples.extend(examples)
        return self

    def set_weights(
        self, weights: Dict[PromptStrategy, float]
    ) -> 'CompoundPromptEngine':
        """Set strategy weights"""
        self.config.weights.update(weights)
        return self


# Convenience functions
async def chain_of_thought(llm_provider: BaseLLMProvider, query: str, **kwargs) -> str:
    """Simple chain-of-thought prompting"""
    strategy = ChainOfThoughtStrategy(llm_provider)
    result = await strategy.generate(query, **kwargs)
    return result.content


async def compound_prompt(
    llm_provider: BaseLLMProvider,
    query: str,
    strategies: Optional[List[PromptStrategy]] = None,
    combination_method: CombinationStrategy = CombinationStrategy.SEQUENTIAL,
    **kwargs,
) -> CompoundResult:
    """Convenience function for compound prompting"""
    config = CompoundPromptConfig(
        strategies=strategies or [PromptStrategy.CHAIN_OF_THOUGHT],
        combination_method=combination_method,
    )

    engine = CompoundPromptEngine(llm_provider, config)
    return await engine.generate(query, **kwargs)

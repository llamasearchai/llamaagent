"""
Advanced Prompt Optimization Module

Implements automatic prompt optimization using various techniques including
genetic algorithms, bayesian optimization, and reinforcement learning.
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..llm.messages import LLMMessage
from ..llm.providers.base_provider import BaseLLMProvider
from .prompt_templates import PromptTemplate


@dataclass
class OptimizationMetrics:
    """Metrics for evaluating prompt performance"""

    accuracy: float = 0.0
    consistency: float = 0.0
    relevance: float = 0.0
    coherence: float = 0.0
    efficiency: float = 0.0  # Based on token usage
    cost: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def overall_score(self) -> float:
        """Calculate overall optimization score"""
        base_metrics = [
            self.accuracy,
            self.consistency,
            self.relevance,
            self.coherence,
            self.efficiency,
        ]

        # Filter out zero values
        valid_metrics = [m for m in base_metrics if m > 0]

        if not valid_metrics:
            return 0.0

        # Weighted average
        weights = {
            "accuracy": 0.3,
            "consistency": 0.2,
            "relevance": 0.2,
            "coherence": 0.2,
            "efficiency": 0.1,
        }

        weighted_sum = (
            weights["accuracy"] * self.accuracy
            + weights["consistency"] * self.consistency
            + weights["relevance"] * self.relevance
            + weights["coherence"] * self.coherence
            + weights["efficiency"] * self.efficiency
        )

        return weighted_sum

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "accuracy": self.accuracy,
            "consistency": self.consistency,
            "relevance": self.relevance,
            "coherence": self.coherence,
            "efficiency": self.efficiency,
            "cost": self.cost,
            "overall_score": self.overall_score,
            "custom_metrics": self.custom_metrics,
        }


@dataclass
class PromptCandidate:
    """A candidate prompt for optimization"""

    template: str
    variables: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance: Optional[OptimizationMetrics] = None
    generation: int = 0
    parent_id: Optional[str] = None
    id: str = field(default_factory=lambda: f"candidate_{datetime.now().timestamp()}")


@dataclass
class OptimizationConfig:
    """Configuration for prompt optimization"""

    population_size: int = 20
    generations: int = 10
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elite_size: int = 5
    temperature_range: Tuple[float, float] = (0.1, 1.0)
    optimization_method: str = "genetic"  # genetic, bayesian, reinforcement
    early_stopping_patience: int = 3
    diversity_weight: float = 0.2


class BaseOptimizationStrategy(ABC):
    """Base class for optimization strategies"""

    @abstractmethod
    async def optimize(
        self,
        initial_prompt: str,
        test_cases: List[Dict[str, Any]],
        evaluator: Callable,
        config: OptimizationConfig,
    ) -> PromptCandidate:
        """Optimize the prompt"""


class GeneticOptimization(BaseOptimizationStrategy):
    """Genetic algorithm-based prompt optimization"""

    def __init__(self, llm_provider: Optional[BaseLLMProvider] = None):
        self.llm_provider = llm_provider
        self.population: List[PromptCandidate] = []
        self.best_candidate: Optional[PromptCandidate] = None
        self.generation_history: List[Dict[str, Any]] = []

    async def optimize(
        self,
        initial_prompt: str,
        test_cases: List[Dict[str, Any]],
        evaluator: Callable,
        config: OptimizationConfig,
    ) -> PromptCandidate:
        """Optimize using genetic algorithm"""

        # Initialize population
        self.population = await self._initialize_population(
            initial_prompt, config.population_size
        )
        # Evaluate initial population
        for candidate in self.population:
            candidate.performance = await self._evaluate_candidate(
                candidate, test_cases, evaluator
            )
        # Evolution loop
        no_improvement_count = 0
        best_score = 0.0

        for generation in range(config.generations):
            # Sort by fitness
            self.population.sort(
                key=lambda x: x.performance.overall_score if x.performance else 0,
                reverse=True,
            )
            # Check for improvement
            current_best_score = self.population[0].performance.overall_score
            if current_best_score > best_score:
                best_score = current_best_score
                self.best_candidate = self.population[0]
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Early stopping
            if no_improvement_count >= config.early_stopping_patience:
                break

            # Record generation stats
            self.generation_history.append(
                {
                    "generation": generation,
                    "best_score": current_best_score,
                    "avg_score": np.mean(
                        [
                            c.performance.overall_score
                            for c in self.population
                            if c.performance
                        ]
                    ),
                    "diversity": self._calculate_diversity(),
                }
            )

            # Create next generation
            next_population = []

            # Elite selection
            next_population.extend(self.population[: config.elite_size])
            # Generate offspring
            while len(next_population) < config.population_size:
                # Tournament selection
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()

                # Crossover
                if random.random() < config.crossover_rate:
                    offspring = await self._crossover(parent1, parent2)
                else:
                    offspring = parent1

                # Mutation
                if random.random() < config.mutation_rate:
                    offspring = await self._mutate(offspring)
                # Set generation
                offspring.generation = generation + 1

                # Evaluate offspring
                offspring.performance = await self._evaluate_candidate(
                    offspring, test_cases, evaluator
                )
                next_population.append(offspring)
            self.population = next_population

        return self.best_candidate or self.population[0]

    async def _initialize_population(
        self, initial_prompt: str, size: int
    ) -> List[PromptCandidate]:
        """Initialize population with variations"""
        population = []

        # Add original
        population.append(
            PromptCandidate(
                template=initial_prompt,
                variables=self._extract_variables(initial_prompt),
            )
        )

        # Generate variations
        variation_strategies = [
            self._add_instruction_variation,
            self._rephrase_variation,
            self._structure_variation,
            self._detail_variation,
            self._constraint_variation,
        ]

        for i in range(size - 1):
            strategy = variation_strategies[i % len(variation_strategies)]
            variation = await strategy(initial_prompt)
            population.append(
                PromptCandidate(
                    template=variation,
                    variables=self._extract_variables(variation),
                    generation=0,
                )
            )

        return population

    async def _add_instruction_variation(self, prompt: str) -> str:
        """Add instruction prefixes"""
        prefixes = [
            "Let's approach this systematically:\n\n",
            "I'll analyze this step by step:\n\n",
            "Breaking this down carefully:\n\n",
            "Let me think through this methodically:\n\n",
            "Here's my detailed analysis:\n\n",
        ]
        return random.choice(prefixes) + prompt

    async def _rephrase_variation(self, prompt: str) -> str:
        """Rephrase the prompt"""
        if not self.llm_provider:
            # Simple rephrasing
            return prompt.replace("Please", "Kindly").replace("analyze", "examine")
        rephrase_prompt = f"""Rephrase this prompt while maintaining its meaning:

Original: {prompt}

Rephrased version:"""

        response = await self.llm_provider.complete(
            [LLMMessage(role="user", content=rephrase_prompt)], temperature=0.8
        )

        return response.content.strip()

    async def _structure_variation(self, prompt: str) -> str:
        """Add structure to the prompt"""
        structured = f"""{prompt}

Please structure your response as follows:
1. Initial Analysis
2. Key Considerations
3. Detailed Solution
4. Summary"""
        return structured

    async def _detail_variation(self, prompt: str) -> str:
        """Add detail requirements"""
        detailed = f"""{prompt}

Requirements:
- Provide comprehensive analysis
- Include specific examples
- Explain your reasoning
- Consider edge cases"""
        return detailed

    async def _constraint_variation(self, prompt: str) -> str:
        """Add constraints"""
        constraints = [
            "\n\nBe concise but thorough.",
            "\n\nFocus on practical applications.",
            "\n\nProvide actionable insights.",
            "\n\nConsider multiple perspectives.",
            "\n\nEnsure accuracy and clarity.",
        ]
        return prompt + random.choice(constraints)

    def _extract_variables(self, template: str) -> List[str]:
        """Extract variable names from template"""
        import re

        # Match {{variable}} pattern
        variables = re.findall(r"\{\{\s*(\w+)\s*\}\}", template)
        # Match {variable} pattern
        variables.extend(re.findall(r"\{\s*(\w+)\s*\}", template))
        return list(set(variables))

    async def _evaluate_candidate(
        self,
        candidate: PromptCandidate,
        test_cases: List[Dict[str, Any]],
        evaluator: Callable,
    ) -> OptimizationMetrics:
        """Evaluate a candidate prompt"""
        metrics = OptimizationMetrics()

        if not test_cases:
            return metrics

        scores = {"accuracy": [], "consistency": [], "relevance": [], "coherence": []}

        # Run test cases
        outputs = []
        for test_case in test_cases:
            try:
                # Format prompt with test case variables
                formatted_prompt = candidate.template
                for var, value in test_case.get("input", {}).items():
                    formatted_prompt = formatted_prompt.replace(
                        f"{{{{{var}}}}}", str(value)
                    )
                    formatted_prompt = formatted_prompt.replace(
                        f"{{{var}}}", str(value)
                    )

                # Get LLM response
                if self.llm_provider:
                    response = await self.llm_provider.complete(
                        [LLMMessage(role="user", content=formatted_prompt)]
                    )
                    output = response.content
                else:
                    output = f"Mock output for: {formatted_prompt[:50]}..."

                outputs.append(output)
                # Evaluate output
                if evaluator:
                    eval_result = await evaluator(
                        output, test_case.get("expected", ""), test_case
                    )

                    if isinstance(eval_result, dict):
                        for metric, value in eval_result.items():
                            if metric in scores:
                                scores[metric].append(value)
                    else:
                        scores["accuracy"].append(float(eval_result))
            except Exception as e:
                print(f"Evaluation error: {e}")
                continue

        # Calculate consistency (variance in outputs for similar inputs)
        if len(outputs) > 1:
            # Simple consistency metric based on output length variance
            lengths = [len(o) for o in outputs]
            metrics.consistency = 1.0 - (np.std(lengths) / (np.mean(lengths) + 1))
        else:
            metrics.consistency = 1.0

        # Average scores
        metrics.accuracy = np.mean(scores["accuracy"]) if scores["accuracy"] else 0.0
        metrics.relevance = np.mean(scores["relevance"]) if scores["relevance"] else 0.8
        metrics.coherence = np.mean(scores["coherence"]) if scores["coherence"] else 0.8

        # Efficiency based on prompt length (shorter is more efficient)
        metrics.efficiency = 1.0 / (1.0 + len(candidate.template) / 1000)

        return metrics

    def _calculate_diversity(self) -> float:
        """Calculate population diversity"""
        if len(self.population) < 2:
            return 0.0

        # Simple diversity based on template length variance
        lengths = [len(c.template) for c in self.population]
        return np.std(lengths) / (np.mean(lengths) + 1)

    def _tournament_selection(self, tournament_size: int = 3) -> PromptCandidate:
        """Tournament selection for genetic algorithm"""
        tournament = random.sample(
            self.population, min(tournament_size, len(self.population))
        )
        return max(
            tournament,
            key=lambda x: x.performance.overall_score if x.performance else 0,
        )

    async def _crossover(
        self, parent1: PromptCandidate, parent2: PromptCandidate
    ) -> PromptCandidate:
        """Crossover two parent prompts"""
        # Split prompts into sentences
        sentences1 = parent1.template.split(". ")
        sentences2 = parent2.template.split(". ")
        # Random crossover point
        if sentences1 and sentences2:
            crossover_point = random.randint(
                1, min(len(sentences1), len(sentences2)) - 1
            )

            # Create offspring
            offspring_template = (
                ". ".join(sentences1[:crossover_point])
                + ". "
                + ". ".join(sentences2[crossover_point:])
            )
        else:
            # Fallback to simple combination
            offspring_template = (
                parent1.template[: len(parent1.template) // 2]
                + parent2.template[len(parent2.template) // 2 :]
            )

        return PromptCandidate(
            template=offspring_template,
            variables=list(set(parent1.variables + parent2.variables)),
            parent_id=f"{parent1.id}+{parent2.id}",
        )

    async def _mutate(self, candidate: PromptCandidate) -> PromptCandidate:
        """Mutate a candidate prompt"""
        mutation_strategies = [
            self._word_substitution,
            self._sentence_reordering,
            self._instruction_addition,
            self._detail_modification,
        ]

        strategy = random.choice(mutation_strategies)
        mutated_template = await strategy(candidate.template)
        return PromptCandidate(
            template=mutated_template,
            variables=candidate.variables,
            parent_id=candidate.id,
            metadata={**candidate.metadata, "mutation": strategy.__name__},
        )

    async def _word_substitution(self, template: str) -> str:
        """Substitute words with synonyms"""
        substitutions = {
            "analyze": ["examine", "investigate", "study", "explore"],
            "provide": ["give", "supply", "offer", "present"],
            "explain": ["describe", "clarify", "elaborate", "detail"],
            "consider": ["think about", "take into account", "evaluate", "assess"],
            "important": ["crucial", "significant", "essential", "key"],
        }

        result = template
        for word, synonyms in substitutions.items():
            if word in result.lower():
                result = result.replace(word, random.choice(synonyms), 1)

        return result

    async def _sentence_reordering(self, template: str) -> str:
        """Reorder sentences in the template"""
        sentences = template.split(". ")
        if len(sentences) > 2:
            # Keep first and last, shuffle middle
            middle = sentences[1:-1]
            random.shuffle(middle)
            return sentences[0] + ". " + ". ".join(middle) + ". " + sentences[-1]
        return template

    async def _instruction_addition(self, template: str) -> str:
        """Add new instructions"""
        additions = [
            "\nBe specific in your analysis.",
            "\nProvide clear reasoning.",
            "\nConsider all aspects.",
            "\nBe thorough but concise.",
            "\nFocus on key insights.",
        ]

        if random.random() < 0.5:
            return template + random.choice(additions)
        return template

    async def _detail_modification(self, template: str) -> str:
        """Modify level of detail requested"""
        if "detailed" in template:
            return template.replace("detailed", "comprehensive")
        elif "brief" in template:
            return template.replace("brief", "concise")
        else:
            return template + "\nProvide appropriate level of detail."


class PromptOptimizer:
    """Main prompt optimization interface"""

    def __init__(
        self, llm_provider: Optional[BaseLLMProvider] = None, strategy: str = "genetic"
    ):
        self.llm_provider = llm_provider
        self.strategy = self._create_strategy(strategy)
        self.optimization_history: List[Dict[str, Any]] = []

    def _create_strategy(self, strategy_name: str) -> BaseOptimizationStrategy:
        """Create optimization strategy"""
        if strategy_name == "genetic":
            return GeneticOptimization(self.llm_provider)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

    async def optimize(
        self,
        prompt: Union[str, PromptTemplate],
        test_cases: List[Dict[str, Any]],
        evaluator: Optional[Callable] = None,
        config: Optional[OptimizationConfig] = None,
    ) -> Tuple[str, OptimizationMetrics]:
        """
        Optimize a prompt using test cases

        Args:
            prompt: Initial prompt or template
            test_cases: List of test cases with input/expected output
            evaluator: Custom evaluation function
            config: Optimization configuration

        Returns:
            Optimized prompt and performance metrics
        """
        config = config or OptimizationConfig()

        # Convert template to string if needed
        if isinstance(prompt, PromptTemplate):
            prompt_str = prompt.template
        else:
            prompt_str = prompt

        # Use default evaluator if none provided
        if evaluator is None:
            evaluator = self._default_evaluator

        # Run optimization
        best_candidate = await self.strategy.optimize(
            prompt_str, test_cases, evaluator, config
        )
        # Record optimization results
        self.optimization_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "original_prompt": prompt_str,
                "optimized_prompt": best_candidate.template,
                "performance": (
                    best_candidate.performance.to_dict()
                    if best_candidate.performance
                    else {}
                ),
                "config": {
                    "method": config.optimization_method,
                    "generations": config.generations,
                    "population_size": config.population_size,
                },
            }
        )

        return best_candidate.template, best_candidate.performance

    async def _default_evaluator(
        self, output: str, expected: str, test_case: Dict[str, Any]
    ) -> Dict[str, float]:
        """Default evaluation function"""
        # Simple similarity-based evaluation
        output_lower = output.lower().strip()
        expected_lower = expected.lower().strip()

        # Exact match
        if output_lower == expected_lower:
            accuracy = 1.0
        # Partial match
        elif expected_lower in output_lower:
            accuracy = 0.8
        # Word overlap
        else:
            output_words = set(output_lower.split())
            expected_words = set(expected_lower.split())
            if output_words and expected_words:
                overlap = len(output_words & expected_words) / len(expected_words)
                accuracy = min(overlap, 0.7)
            else:
                accuracy = 0.0

        # Length-based relevance
        length_ratio = min(len(output), len(expected)) / max(len(output), len(expected))
        relevance = length_ratio

        # Basic coherence check
        coherence = 1.0 if output.count(".") > 0 and len(output) > 10 else 0.5

        return {"accuracy": accuracy, "relevance": relevance, "coherence": coherence}

    def get_optimization_report(self) -> Dict[str, Any]:
        """Get detailed optimization report"""
        if not self.optimization_history:
            return {"message": "No optimizations performed yet"}

        latest = self.optimization_history[-1]

        # Calculate improvements
        if hasattr(self.strategy, "generation_history"):
            generation_data = self.strategy.generation_history
        else:
            generation_data = []

        return {
            "latest_optimization": latest,
            "total_optimizations": len(self.optimization_history),
            "generation_progress": generation_data,
            "best_score": latest["performance"].get("overall_score", 0),
            "optimization_method": latest["config"]["method"],
        }

    async def auto_optimize_template(
        self,
        template: PromptTemplate,
        sample_inputs: List[Dict[str, Any]],
        expected_outputs: List[str],
    ) -> PromptTemplate:
        """Automatically optimize a prompt template"""
        # Create test cases
        test_cases = []
        for inp, exp in zip(sample_inputs, expected_outputs, strict=False):
            test_cases.append({"input": inp, "expected": exp})
        # Optimize
        optimized_text, metrics = await self.optimize(template, test_cases)
        # Create new template
        optimized_template = PromptTemplate(
            name=f"{template.name}_optimized",
            template=optimized_text,
            type=template.type,
            variables=template.variables,
            description=f"{template.description} (Optimized)",
            examples=template.examples,
            constraints=template.constraints,
            metadata={
                **template.metadata,
                "optimization_score": metrics.overall_score if metrics else 0,
                "optimization_date": datetime.now().isoformat(),
            },
        )

        return optimized_template


# Utility functions
async def compare_prompts(
    prompts: List[str],
    test_cases: List[Dict[str, Any]],
    llm_provider: BaseLLMProvider,
    evaluator: Optional[Callable] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compare multiple prompts on same test cases"""
    optimizer = PromptOptimizer(llm_provider)
    if evaluator is None:
        evaluator = optimizer._default_evaluator

    results = {}

    for i, prompt in enumerate(prompts):
        prompt_name = f"Prompt_{i + 1}"
        candidate = PromptCandidate(template=prompt, variables=[])
        # Evaluate
        metrics = await optimizer.strategy._evaluate_candidate(
            candidate, test_cases, evaluator
        )
        results[prompt_name] = {
            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "metrics": metrics.to_dict(),
            "score": metrics.overall_score,
        }

    # Rank prompts
    ranked = sorted(results.items(), key=lambda x: x[1]["score"], reverse=True)

    return {
        "comparison": results,
        "ranking": [name for name, _ in ranked],
        "best_prompt": ranked[0][0] if ranked else None,
    }

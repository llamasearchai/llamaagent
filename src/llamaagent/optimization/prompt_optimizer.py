"""
Advanced prompt optimization system with evolutionary algorithms and multi-objective optimization.

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import asyncio
import random
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class OptimizationStrategy(Enum):
    """Optimization strategies for prompt optimization."""

    GENETIC_ALGORITHM = "genetic_algorithm"
    RANDOM_SEARCH = "random_search"
    EVOLUTIONARY_SEARCH = "evolutionary_search"
    MULTI_OBJECTIVE = "multi_objective"
    ADAPTIVE = "adaptive"


class PromptType(Enum):
    """Types of prompts for optimization."""

    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    INSTRUCTION = "instruction"
    CONVERSATION = "conversation"


@dataclass
class PromptVariant:
    """A variant of a prompt with associated metadata."""

    id: str
    content: str
    fitness: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate unique ID for prompt variant."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        return f"variant_{timestamp}_{random.randint(1000, 9999)}"


@dataclass
class OptimizationConfig:
    """Configuration for prompt optimization."""

    strategy: OptimizationStrategy = OptimizationStrategy.GENETIC_ALGORITHM
    prompt_type: PromptType = PromptType.INSTRUCTION
    population_size: int = 20
    generations: int = 10
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 2
    objective_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "accuracy": 0.4,
            "clarity": 0.3,
            "completeness": 0.2,
            "safety": 0.1,
        }
    )
    early_stopping: bool = True
    patience: int = 3
    target_fitness: float = 0.95


@dataclass
class OptimizationResult:
    """Result of prompt optimization."""

    best_prompt: PromptVariant
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    total_generations: int = 0
    total_evaluations: int = 0
    convergence_generation: Optional[int] = None
    final_fitness: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PromptOptimizer:
    """
    Advanced prompt optimizer using evolutionary algorithms and multi-objective optimization.
    """

    def __init__(
        self,
        evaluation_function: Optional[Callable[[str, Dict[str, Any]], float]] = None,
        storage_path: Optional[Path] = None,
    ):
        self.evaluation_function = (
            evaluation_function or self._default_evaluation_function
        )
        self.storage_path = storage_path or Path("./prompt_optimization")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Prompt patterns and templates
        self.prompt_templates = {
            PromptType.ZERO_SHOT: [
                "Act as a {role}. {instruction}",
                "You are a {role}. Please {instruction}",
                "Task: {instruction}. Please provide a detailed response.",
                "I need you to {instruction}. Here's what I'm looking for: {requirements}",
            ],
            PromptType.FEW_SHOT: [
                "Here are some examples:\n{examples}\n\nNow, {instruction}",
                "Learn from these examples:\n{examples}\n\nTask: {instruction}",
                "Examples:\n{examples}\n\nBased on these examples, {instruction}",
            ],
            PromptType.CHAIN_OF_THOUGHT: [
                "Let's think step by step. {instruction}",
                "First, let me break this down:\n1. {step1}\n2. {step2}\n3. {step3}\n\nNow, {instruction}",
                "To solve this, I need to:\n- {approach}\n\n{instruction}",
            ],
            PromptType.INSTRUCTION: [
                "Please {instruction}",
                "Your task is to {instruction}",
                "I want you to {instruction}",
                "Can you {instruction}?",
            ],
        }

        self.prompt_patterns = {
            "clarity_enhancers": [
                "clearly explain",
                "provide detailed information about",
                "thoroughly describe",
                "give a comprehensive overview of",
                "explain in detail",
            ],
            "context_markers": [
                "Background:",
                "Context:",
                "Details:",
                "Information:",
                "Note:",
            ],
            "output_formatters": [
                "Please format your response as:",
                "Structure your answer with:",
                "Organize your response using:",
                "Present your findings in the following format:",
            ],
        }

    async def optimize(
        self,
        base_prompt: str,
        config: OptimizationConfig,
        context: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        """Optimize a prompt using the specified strategy."""
        print(f"Starting prompt optimization with {config.strategy.value}")

        context = context or {}

        # Initialize population
        initial_population = await self._create_initial_population(
            base_prompt, config, context
        )

        # Run optimization based on strategy
        if config.strategy == OptimizationStrategy.GENETIC_ALGORITHM:
            return await self._genetic_algorithm_optimization(
                initial_population, config, context
            )
        elif config.strategy == OptimizationStrategy.RANDOM_SEARCH:
            return await self._random_search_optimization(
                initial_population, config, context
            )
        elif config.strategy == OptimizationStrategy.EVOLUTIONARY_SEARCH:
            return await self._evolutionary_search_optimization(
                initial_population, config, context
            )
        else:
            return await self._genetic_algorithm_optimization(
                initial_population, config, context
            )

    async def _create_initial_population(
        self, base_prompt: str, config: OptimizationConfig, context: Dict[str, Any]
    ) -> List[PromptVariant]:
        """Create initial population of prompt variants."""
        population: List[PromptVariant] = []

        # Add the base prompt
        base_variant = PromptVariant(
            id=self._generate_prompt_id(base_prompt), content=base_prompt, generation=0
        )
        base_variant.fitness = await self._evaluate_prompt(base_variant, context)
        population.append(base_variant)

        # Generate variations
        for i in range(config.population_size - 1):
            variant = await self._generate_prompt_variant(base_prompt, config, context)
            variant.generation = 0
            variant.fitness = await self._evaluate_prompt(variant, context)
            population.append(variant)

        return population

    async def _genetic_algorithm_optimization(
        self,
        initial_population: List[PromptVariant],
        config: OptimizationConfig,
        context: Dict[str, Any],
    ) -> OptimizationResult:
        """Run genetic algorithm optimization."""

        population = initial_population
        optimization_history: List[Dict[str, Any]] = []
        best_fitness_history = []

        for generation in range(config.generations):
            print(f"Generation {generation + 1}/{config.generations}")

            # Evaluate population
            fitness_scores = [p.fitness for p in population]
            best_fitness = max(fitness_scores)
            avg_fitness = statistics.mean(fitness_scores)

            best_fitness_history.append(best_fitness)

            # Record generation stats
            generation_stats = {
                "generation": generation + 1,
                "best_fitness": best_fitness,
                "avg_fitness": avg_fitness,
                "population_size": len(population),
                "diversity": self._calculate_diversity(population),
            }
            optimization_history.append(generation_stats)

            # Check for convergence
            if config.early_stopping and best_fitness >= config.target_fitness:
                print(f"Early stopping: target fitness {config.target_fitness} reached")
                break

            # Create next generation
            population = await self._create_next_generation(population, config, context)

        # Find best prompt
        best_prompt = max(population, key=lambda p: p.fitness)

        return OptimizationResult(
            best_prompt=best_prompt,
            optimization_history=optimization_history,
            total_generations=len(optimization_history),
            total_evaluations=len(optimization_history) * config.population_size,
            final_fitness=best_prompt.fitness,
            metadata={"best_fitness_history": best_fitness_history},
        )

    async def _random_search_optimization(
        self,
        initial_population: List[PromptVariant],
        config: OptimizationConfig,
        context: Dict[str, Any],
    ) -> OptimizationResult:
        """Run random search optimization."""

        base_prompt = initial_population[0].content
        best_variant = initial_population[0]
        all_variants = initial_population[:]
        optimization_history: List[Dict[str, Any]] = []

        for iteration in range(config.generations * config.population_size):
            # Generate random variant
            variant = await self._generate_prompt_variant(base_prompt, config, context)
            variant.fitness = await self._evaluate_prompt(variant, context)
            all_variants.append(variant)

            # Update best
            if variant.fitness > best_variant.fitness:
                best_variant = variant

            # Record stats every generation
            if iteration % config.population_size == 0:
                generation = iteration // config.population_size + 1
                recent_variants = all_variants[-config.population_size :]
                avg_fitness = statistics.mean([v.fitness for v in recent_variants])

                optimization_history.append(
                    {
                        "generation": generation,
                        "best_fitness": best_variant.fitness,
                        "avg_fitness": avg_fitness,
                        "total_variants": len(all_variants),
                    }
                )

        return OptimizationResult(
            best_prompt=best_variant,
            optimization_history=optimization_history,
            total_generations=config.generations,
            total_evaluations=len(all_variants),
            final_fitness=best_variant.fitness,
        )

    async def _evolutionary_search_optimization(
        self,
        initial_population: List[PromptVariant],
        config: OptimizationConfig,
        context: Dict[str, Any],
    ) -> OptimizationResult:
        """Run evolutionary search optimization with advanced operators."""

        # Similar to genetic algorithm but with more sophisticated operators
        # This is a simplified version - could be expanded with more advanced techniques
        return await self._genetic_algorithm_optimization(
            initial_population, config, context
        )

    async def _create_next_generation(
        self,
        population: List[PromptVariant],
        config: OptimizationConfig,
        context: Dict[str, Any],
    ) -> List[PromptVariant]:
        """Create next generation using selection, crossover, and mutation."""

        # Sort by fitness
        population.sort(key=lambda p: p.fitness, reverse=True)

        # Elite selection
        next_generation = population[: config.elite_size]

        # Generate offspring
        while len(next_generation) < config.population_size:
            # Selection
            parent1 = self._tournament_selection(population, k=3)
            parent2 = self._tournament_selection(population, k=3)

            # Crossover
            if random.random() < config.crossover_rate:
                child = await self._crossover(parent1, parent2, config, context)
            else:
                child = parent1

            # Mutation
            if random.random() < config.mutation_rate:
                child = await self._mutate(child, config, context)

            # Evaluate child
            child.fitness = await self._evaluate_prompt(child, context)
            next_generation.append(child)

        return next_generation

    def _tournament_selection(
        self, population: List[PromptVariant], k: int = 3
    ) -> PromptVariant:
        """Tournament selection for genetic algorithm."""
        tournament = random.sample(population, k)
        return max(tournament, key=lambda p: p.fitness)

    async def _crossover(
        self,
        parent1: PromptVariant,
        parent2: PromptVariant,
        config: OptimizationConfig,
        context: Dict[str, Any],
    ) -> PromptVariant:
        """Crossover operation between two parents."""

        # Simple crossover: combine parts of both prompts
        p1_words = parent1.content.split()
        p2_words = parent2.content.split()

        # Take random portions from each parent
        crossover_point = random.randint(1, min(len(p1_words), len(p2_words)) - 1)

        if random.random() < 0.5:
            child_words = p1_words[:crossover_point] + p2_words[crossover_point:]
        else:
            child_words = p2_words[:crossover_point] + p1_words[crossover_point:]

        child_content = " ".join(child_words)

        return PromptVariant(
            id=self._generate_prompt_id(child_content),
            content=child_content,
            parent_ids=[parent1.id, parent2.id],
            generation=max(parent1.generation, parent2.generation) + 1,
        )

    async def _mutate(
        self, parent: PromptVariant, config: OptimizationConfig, context: Dict[str, Any]
    ) -> PromptVariant:
        """Mutation operation."""

        if random.random() < config.mutation_rate:
            mutated_content = await self._generate_prompt_variant(
                parent.content, config, context
            )
            return PromptVariant(
                id=self._generate_prompt_id(mutated_content.content),
                content=mutated_content.content,
                parent_ids=[parent.id],
                generation=parent.generation + 1,
            )
        return parent

    async def _generate_prompt_variant(
        self, base_prompt: str, config: OptimizationConfig, context: Dict[str, Any]
    ) -> PromptVariant:
        """Generate a new prompt variant."""

        # Select mutation strategy
        mutation_strategies = [
            self._mutate_with_templates,
            self._mutate_with_structure,
            self._mutate_with_context,
            self._mutate_with_examples,
        ]

        mutation_strategy = random.choice(mutation_strategies)
        mutated_content = await mutation_strategy(base_prompt, config, context)

        return PromptVariant(
            id=self._generate_prompt_id(mutated_content), content=mutated_content
        )

    async def _mutate_with_templates(
        self, prompt: str, config: OptimizationConfig, context: Dict[str, Any]
    ) -> str:
        """Mutate prompt using templates."""
        templates = self.prompt_templates.get(config.prompt_type, [])
        if templates:
            template = random.choice(templates)
            # Simple template filling - could be more sophisticated
            return template.format(
                instruction=prompt,
                role=context.get("role", "helpful assistant"),
                requirements=context.get("requirements", "accurate information"),
            )
        return prompt

    async def _mutate_with_structure(
        self, prompt: str, config: OptimizationConfig, context: Dict[str, Any]
    ) -> str:
        """Mutate prompt by changing structure."""

        mutations = [
            lambda p: f"Task: {p}\n\nPlease complete this task thoroughly.",
            lambda p: f"Instructions: {p}\n\nProvide a detailed response.",
            lambda p: f"Question: {p}\n\nAnswer:",
            lambda p: f"Please help me with the following: {p}",
        ]

        mutation = random.choice(mutations)
        return mutation(prompt)

    async def _mutate_with_context(
        self, prompt: str, config: OptimizationConfig, context: Dict[str, Any]
    ) -> str:
        """Mutate prompt by adding context."""

        context_additions = [
            "Consider the context and provide a comprehensive response.",
            "Take into account all relevant factors when responding.",
            "Provide a thorough and well-reasoned answer.",
            "Please be specific and detailed in your response.",
        ]

        addition = random.choice(context_additions)
        return f"{prompt}\n\n{addition}"

    async def _mutate_with_examples(
        self, prompt: str, config: OptimizationConfig, context: Dict[str, Any]
    ) -> str:
        """Mutate prompt by adding example structure."""

        if config.prompt_type == PromptType.FEW_SHOT:
            return f"""Here's an example of the type of response I'm looking for:

Example: [Placeholder for example]

Now, {prompt}"""

        return prompt

    def _calculate_diversity(self, population: List[PromptVariant]) -> float:
        """Calculate diversity score for population."""
        if len(population) < 2:
            return 0.0

        # Simple diversity measure based on content length variation
        lengths = [len(p.content) for p in population]
        return (
            statistics.stdev(lengths) / statistics.mean(lengths)
            if statistics.mean(lengths) > 0
            else 0.0
        )

    async def _evaluate_prompt(
        self, prompt: PromptVariant, context: Dict[str, Any]
    ) -> float:
        """Evaluate prompt fitness."""
        return await asyncio.to_thread(
            self.evaluation_function, prompt.content, context
        )

    def _default_evaluation_function(
        self, prompt: str, context: Dict[str, Any]
    ) -> float:
        """Default evaluation function for prompts."""
        scores = {}

        # Length score (prefer moderate length)
        length = len(prompt)
        if length < 50:
            length_score = length / 50.0
        elif length > 500:
            length_score = 1.0 - (length - 500) / 1000.0
        else:
            length_score = 1.0

        scores["length"] = max(0.0, length_score)

        # Clarity score (based on structure and common words)
        clarity_indicators = [
            "please",
            "explain",
            "describe",
            "provide",
            "help",
            "task",
            "question",
        ]
        clarity_score = sum(
            1 for indicator in clarity_indicators if indicator in prompt.lower()
        ) / len(clarity_indicators)
        scores["clarity"] = clarity_score

        # Completeness score (based on instruction completeness)
        completeness_indicators = [
            "detailed",
            "comprehensive",
            "thorough",
            "complete",
            "specific",
        ]
        completeness_score = sum(
            1 for indicator in completeness_indicators if indicator in prompt.lower()
        )
        scores["completeness"] = min(completeness_score / 3.0, 1.0)

        # Safety score (simple keyword check)
        safety_score = (
            1.0
            if not any(
                word in prompt.lower() for word in ["harm", "illegal", "dangerous"]
            )
            else 0.5
        )
        scores["safety"] = safety_score

        # Weighted average
        weights = {"length": 0.2, "clarity": 0.3, "completeness": 0.3, "safety": 0.2}

        total_score = sum(scores[key] * weights[key] for key in scores)
        return total_score

    def _generate_prompt_id(self, prompt: str) -> str:
        """Generate unique ID for prompt."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        prompt_hash = hash(prompt) % 10000
        return f"prompt_{timestamp}_{prompt_hash}"


# Convenience functions
async def optimize_prompt(
    prompt: str,
    strategy: OptimizationStrategy = OptimizationStrategy.GENETIC_ALGORITHM,
    generations: int = 10,
    population_size: int = 20,
    evaluation_function: Optional[Callable] = None,
) -> OptimizationResult:
    """Convenience function for prompt optimization."""

    optimizer = PromptOptimizer(evaluation_function=evaluation_function)
    config = OptimizationConfig(
        strategy=strategy, generations=generations, population_size=population_size
    )

    return await optimizer.optimize(prompt, config)


if __name__ == "__main__":
    # Example usage
    async def main():
        base_prompt = "Explain the concept of machine learning"

        optimizer = PromptOptimizer()
        config = OptimizationConfig(
            strategy=OptimizationStrategy.GENETIC_ALGORITHM,
            generations=5,
            population_size=10,
        )

        result = await optimizer.optimize(base_prompt, config)
        print(f"Best prompt: {result.best_prompt.content}")
        print(f"Fitness: {result.best_prompt.fitness}")

    asyncio.run(main())

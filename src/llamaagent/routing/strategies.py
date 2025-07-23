"""
Routing strategies for intelligent AI provider selection.
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

if TYPE_CHECKING:
    from .provider_registry import ProviderRegistry

from .task_analyzer import TaskCharacteristics, TaskComplexity, TaskType
from .types import RoutingDecision

logger = logging.getLogger(__name__)


class RoutingStrategy(ABC):
    """Base class for routing strategies."""

    @abstractmethod
    async def route(
        self,
        task: str,
        characteristics: TaskCharacteristics,
        available_providers: List[str],
        context: Dict[str, Any],
        metrics: Dict[str, Dict[str, float]],
    ) -> RoutingDecision:
        """
        Route a task to the most appropriate provider.

        Args:
            task: The task description
            characteristics: Analyzed task characteristics
            available_providers: List of available provider IDs
            context: Additional context
            metrics: Provider performance metrics

        Returns:
            RoutingDecision with selected provider
        """


class TaskBasedRouting(RoutingStrategy):
    """Route based on task type (debugging, refactoring, new code, etc.)."""

    def __init__(self, provider_registry: 'ProviderRegistry') -> None:
        self.provider_registry = provider_registry

        # Define task type to provider mappings
        self.task_type_preferences = {
            TaskType.DEBUGGING: {
                "claude-code": 0.9,  # Claude excels at debugging
                "openai-codex": 0.8,
                "github-copilot": 0.7,
                "local-model": 0.5,
            },
            TaskType.REFACTORING: {
                "openai-codex": 0.9,  # Codex great for refactoring
                "claude-code": 0.85,
                "github-copilot": 0.7,
                "local-model": 0.6,
            },
            TaskType.CODE_GENERATION: {
                "github-copilot": 0.9,  # Copilot optimized for generation
                "openai-codex": 0.85,
                "claude-code": 0.8,
                "local-model": 0.7,
            },
            TaskType.CODE_REVIEW: {
                "claude-code": 0.95,  # Claude best for analysis
                "openai-codex": 0.8,
                "github-copilot": 0.6,
                "local-model": 0.5,
            },
            TaskType.DOCUMENTATION: {
                "claude-code": 0.9,  # Claude excels at documentation
                "openai-codex": 0.85,
                "local-model": 0.7,
                "github-copilot": 0.6,
            },
            TaskType.TESTING: {
                "openai-codex": 0.9,  # Codex great for test generation
                "claude-code": 0.85,
                "github-copilot": 0.8,
                "local-model": 0.6,
            },
            TaskType.OPTIMIZATION: {
                "openai-codex": 0.85,
                "claude-code": 0.9,  # Claude good at optimization analysis
                "local-model": 0.7,
                "github-copilot": 0.65,
            },
        }

    async def route(
        self,
        task: str,
        characteristics: TaskCharacteristics,
        available_providers: List[str],
        context: Dict[str, Any],
        metrics: Dict[str, Dict[str, float]],
    ) -> RoutingDecision:
        """Route based on task type."""
        task_type = characteristics.task_type
        preferences = self.task_type_preferences.get(task_type, {})
        # Score each available provider
        scores = {}
        for provider_id in available_providers:
            base_score = preferences.get(provider_id, 0.5)
            # Adjust based on recent performance
            if provider_id in metrics:
                success_rate = metrics[provider_id].get("success_rate", 1.0)
                base_score *= success_rate

            scores[provider_id] = base_score

        # Select best provider
        if not scores:
            # Fallback to first available
            selected = available_providers[0]
            confidence = 0.5
        else:
            selected = max(scores, key=lambda k: scores[k])
            confidence = scores[selected]

        reasoning = (
            f"Selected {selected} for {task_type.value} task "
            "based on historical performance"
        )

        return RoutingDecision(
            provider_id=selected,
            confidence=confidence,
            reasoning=reasoning,
            estimated_cost=0.0,  # Will be filled by router
            estimated_duration=0.0,  # Will be filled by router
            metadata={"task_type": task_type.value, "scores": scores},
        )


class LanguageBasedRouting(RoutingStrategy):
    """Route based on programming language expertise."""

    def __init__(self, provider_registry: 'ProviderRegistry') -> None:
        self.provider_registry = provider_registry

        # Define language to provider expertise mappings
        self.language_expertise = {
            "python": {
                "claude-code": 0.95,
                "openai-codex": 0.9,
                "github-copilot": 0.85,
                "local-model": 0.8,
            },
            "javascript": {
                "github-copilot": 0.95,
                "openai-codex": 0.9,
                "claude-code": 0.85,
                "local-model": 0.7,
            },
            "typescript": {
                "github-copilot": 0.95,
                "openai-codex": 0.9,
                "claude-code": 0.85,
                "local-model": 0.7,
            },
            "java": {
                "openai-codex": 0.9,
                "claude-code": 0.85,
                "github-copilot": 0.85,
                "local-model": 0.7,
            },
            "cpp": {
                "openai-codex": 0.85,
                "claude-code": 0.9,
                "github-copilot": 0.8,
                "local-model": 0.6,
            },
            "rust": {
                "claude-code": 0.9,
                "openai-codex": 0.85,
                "github-copilot": 0.75,
                "local-model": 0.6,
            },
            "go": {
                "openai-codex": 0.85,
                "claude-code": 0.85,
                "github-copilot": 0.8,
                "local-model": 0.7,
            },
        }

    async def route(
        self,
        task: str,
        characteristics: TaskCharacteristics,
        available_providers: List[str],
        context: Dict[str, Any],
        metrics: Dict[str, Dict[str, float]],
    ) -> RoutingDecision:
        """Route based on programming language."""
        languages = characteristics.languages

        if not languages:
            # No specific language detected, use general routing
            selected = available_providers[0]
            confidence = 0.5
            reasoning = "No specific language detected, using default provider"
        else:
            # Score providers based on language expertise
            scores = {}
            for provider_id in available_providers:
                score = 0.0
                for lang in languages:
                    lang_lower = lang.lower()
                    if lang_lower in self.language_expertise:
                        score += self.language_expertise[lang_lower].get(
                            provider_id, 0.5
                        )
                    else:
                        # Unknown language, use default score
                        score += 0.5

                # Average score across languages
                scores[provider_id] = score / len(languages)
                # Adjust based on metrics
                if provider_id in metrics:
                    success_rate = metrics[provider_id].get("success_rate", 1.0)
                    scores[provider_id] *= success_rate

            selected = max(scores, key=lambda k: scores[k])
            confidence = scores[selected]
            reasoning = (
                f"Selected {selected} based on expertise in {', '.join(languages)}"
            )

        return RoutingDecision(
            provider_id=selected,
            confidence=confidence,
            reasoning=reasoning,
            estimated_cost=0.0,
            estimated_duration=0.0,
            metadata={
                "languages": list(languages),
                "scores": scores if 'scores' in locals() else {},
            },
        )


class ComplexityBasedRouting(RoutingStrategy):
    """Route based on task complexity."""

    def __init__(self, provider_registry: 'ProviderRegistry') -> None:
        self.provider_registry = provider_registry

        # Define complexity to provider suitability
        self.complexity_suitability = {
            TaskComplexity.TRIVIAL: {
                "local-model": 0.9,  # Use cheap/fast for simple tasks
                "github-copilot": 0.8,
                "openai-codex": 0.6,
                "claude-code": 0.5,
            },
            TaskComplexity.SIMPLE: {
                "github-copilot": 0.85,
                "local-model": 0.8,
                "openai-codex": 0.7,
                "claude-code": 0.65,
            },
            TaskComplexity.MODERATE: {
                "openai-codex": 0.85,
                "claude-code": 0.85,
                "github-copilot": 0.75,
                "local-model": 0.6,
            },
            TaskComplexity.COMPLEX: {
                "claude-code": 0.9,  # Use advanced for complex tasks
                "openai-codex": 0.85,
                "github-copilot": 0.6,
                "local-model": 0.4,
            },
            TaskComplexity.VERY_COMPLEX: {
                "claude-code": 0.95,  # Claude best for very complex
                "openai-codex": 0.8,
                "github-copilot": 0.5,
                "local-model": 0.3,
            },
        }

    async def route(
        self,
        task: str,
        characteristics: TaskCharacteristics,
        available_providers: List[str],
        context: Dict[str, Any],
        metrics: Dict[str, Dict[str, float]],
    ) -> RoutingDecision:
        """Route based on task complexity."""
        complexity = characteristics.complexity
        suitability = self.complexity_suitability.get(complexity, {})
        # Score each provider
        scores = {}
        for provider_id in available_providers:
            base_score = suitability.get(provider_id, 0.5)
            # Adjust based on performance metrics
            if provider_id in metrics:
                # For complex tasks, weight success rate higher
                weight = (
                    0.7
                    if complexity
                    in [TaskComplexity.COMPLEX, TaskComplexity.VERY_COMPLEX]
                    else 0.3
                )
                success_rate = metrics[provider_id].get("success_rate", 1.0)
                base_score = base_score * (1 - weight) + success_rate * weight

            scores[provider_id] = base_score

        selected = (
            max(scores, key=lambda k: scores[k]) if scores else available_providers[0]
        )
        confidence = scores.get(selected, 0.5)
        reasoning = f"Selected {selected} for {complexity.value} complexity task"

        return RoutingDecision(
            provider_id=selected,
            confidence=confidence,
            reasoning=reasoning,
            estimated_cost=0.0,
            estimated_duration=0.0,
            metadata={"complexity": complexity.value, "scores": scores},
        )


class PerformanceBasedRouting(RoutingStrategy):
    """Route based on historical performance metrics."""

    def __init__(
        self,
        provider_registry: 'ProviderRegistry',
        weight_success: float = 0.4,
        weight_latency: float = 0.3,
        weight_cost: float = 0.3,
    ) -> None:
        self.provider_registry = provider_registry
        self.weight_success = weight_success
        self.weight_latency = weight_latency
        self.weight_cost = weight_cost

    async def route(
        self,
        task: str,
        characteristics: TaskCharacteristics,
        available_providers: List[str],
        context: Dict[str, Any],
        metrics: Dict[str, Dict[str, float]],
    ) -> RoutingDecision:
        """Route based on performance metrics."""
        scores = {}

        # Normalize metrics across providers
        max_latency = max(
            (m.get("avg_latency", 1.0) for m in metrics.values() if "avg_latency" in m),
            default=1.0,
        )
        max_cost = max(
            (m.get("avg_cost", 1.0) for m in metrics.values() if "avg_cost" in m),
            default=1.0,
        )

        for provider_id in available_providers:
            if provider_id not in metrics:
                # No metrics available, use default score
                scores[provider_id] = 0.5
                continue

            provider_metrics = metrics[provider_id]

            # Calculate weighted score
            success_score = provider_metrics.get("success_rate", 0.5)
            # Normalize latency (lower is better)
            latency = provider_metrics.get("avg_latency", max_latency)
            latency_score = 1.0 - (latency / max_latency) if max_latency > 0 else 0.5

            # Normalize cost (lower is better)
            cost = provider_metrics.get("avg_cost", max_cost)
            cost_score = 1.0 - (cost / max_cost) if max_cost > 0 else 0.5

            # Calculate weighted score
            total_score = (
                self.weight_success * success_score
                + self.weight_latency * latency_score
                + self.weight_cost * cost_score
            )

            scores[provider_id] = total_score

        # Select best performer
        selected = (
            max(scores, key=lambda k: scores[k]) if scores else available_providers[0]
        )
        confidence = scores.get(selected, 0.5)
        reasoning = (
            f"Selected {selected} based on performance metrics "
            f"(success: {self.weight_success}, latency: {self.weight_latency}, "
            f"cost: {self.weight_cost})"
        )

        return RoutingDecision(
            provider_id=selected,
            confidence=confidence,
            reasoning=reasoning,
            estimated_cost=0.0,
            estimated_duration=0.0,
            metadata={
                "scores": scores,
                "weights": {
                    "success": self.weight_success,
                    "latency": self.weight_latency,
                    "cost": self.weight_cost,
                },
            },
        )


class CostOptimizedRouting(RoutingStrategy):
    """Route to minimize costs while maintaining quality."""

    def __init__(
        self,
        provider_registry: 'ProviderRegistry',
        quality_threshold: float = 0.8,
        budget_limit: Optional[float] = None,
    ) -> None:
        self.provider_registry = provider_registry
        self.quality_threshold = quality_threshold
        self.budget_limit = budget_limit

    async def route(
        self,
        task: str,
        characteristics: TaskCharacteristics,
        available_providers: List[str],
        context: Dict[str, Any],
        metrics: Dict[str, Dict[str, float]],
    ) -> RoutingDecision:
        """Route to minimize cost while maintaining quality threshold."""
        eligible_providers = []

        for provider_id in available_providers:
            # Check if provider meets quality threshold
            if provider_id in metrics:
                success_rate = metrics[provider_id].get("success_rate", 1.0)
                if success_rate < self.quality_threshold:
                    continue

            # Get cost information
            caps = self.provider_registry.get_capabilities(provider_id)
            if caps:
                cost = caps.cost_per_token

                # Check budget limit if specified
                if self.budget_limit and cost > self.budget_limit:
                    continue

                eligible_providers.append(
                    (
                        provider_id,
                        cost,
                        metrics.get(provider_id, {}).get("success_rate", 1.0),
                    )
                )

        if not eligible_providers:
            # No providers meet criteria, relax constraints
            selected = available_providers[0]
            confidence = 0.5
            reasoning = "No providers meet cost/quality criteria, using fallback"
        else:
            # Sort by cost (ascending) then by quality (descending)
            eligible_providers.sort(
                key=lambda x: (cast(float, x[1]), -cast(float, x[2]))
            )
            selected = eligible_providers[0][0]
            confidence = eligible_providers[0][2]
            reasoning = (
                f"Selected {selected} as lowest cost provider meeting "
                f"quality threshold {self.quality_threshold}"
            )

        return RoutingDecision(
            provider_id=selected,
            confidence=confidence,
            reasoning=reasoning,
            estimated_cost=0.0,
            estimated_duration=0.0,
            metadata={
                "quality_threshold": self.quality_threshold,
                "budget_limit": self.budget_limit,
                "eligible_providers": [p[0] for p in eligible_providers],
            },
        )


class HybridRouting(RoutingStrategy):
    """Combine multiple routing strategies with configurable weights."""

    def __init__(self, strategies: List[Tuple[RoutingStrategy, float]]):
        """
        Initialize hybrid routing.

        Args:
            strategies: List of (strategy, weight) tuples
        """
        self.strategies = strategies

        # Normalize weights
        total_weight = sum(weight for _, weight in strategies)
        self.strategies = [(s, w / total_weight) for s, w in strategies]

    async def route(
        self,
        task: str,
        characteristics: TaskCharacteristics,
        available_providers: List[str],
        context: Dict[str, Any],
        metrics: Dict[str, Dict[str, float]],
    ) -> RoutingDecision:
        """Route using weighted combination of strategies."""
        # Get decisions from all strategies
        decisions = []
        for strategy, weight in self.strategies:
            decision = await strategy.route(
                task,
                characteristics,
                available_providers,
                context,
                metrics,
            )
            decisions.append((decision, weight))
        # Aggregate scores for each provider
        provider_scores = {}
        total_confidence = 0.0

        for decision, weight in decisions:
            provider_id = decision.provider_id
            score = decision.confidence * weight

            if provider_id not in provider_scores:
                provider_scores[provider_id] = 0.0
            provider_scores[provider_id] += score
            total_confidence += decision.confidence * weight

        # Select provider with highest aggregate score
        selected = max(provider_scores, key=lambda k: provider_scores[k])
        confidence = min(provider_scores[selected], 1.0)  # Cap at 1.0

        # Combine reasoning from all strategies
        reasoning_parts = []
        for i, (decision, weight) in enumerate(decisions):
            if decision.provider_id == selected:
                strategy_name = type(self.strategies[i][0]).__name__
                reasoning_parts.append(f"{strategy_name} ({weight:.1%})")

        reasoning = f"Hybrid routing selected {selected} based on: " + ", ".join(
            reasoning_parts
        )
        return RoutingDecision(
            provider_id=selected,
            confidence=confidence,
            reasoning=reasoning,
            estimated_cost=0.0,
            estimated_duration=0.0,
            metadata={
                "strategy_decisions": [
                    (type(d).__name__, d.provider_id, d.confidence)
                    for d, _ in decisions
                ],
                "provider_scores": provider_scores,
            },
        )


class ConsensusRouting(RoutingStrategy):
    """Route to multiple providers and aggregate results."""

    def __init__(
        self,
        provider_registry: 'ProviderRegistry',
        min_providers: int = 3,
        consensus_threshold: float = 0.7,
    ) -> None:
        self.provider_registry = provider_registry
        self.min_providers = min_providers
        self.consensus_threshold = consensus_threshold

    async def route(
        self,
        task: str,
        characteristics: TaskCharacteristics,
        available_providers: List[str],
        context: Dict[str, Any],
        metrics: Dict[str, Dict[str, float]],
    ) -> RoutingDecision:
        """Select multiple providers for consensus."""
        # Score providers based on diversity and quality
        provider_scores = []

        for provider_id in available_providers:
            caps = self.provider_registry.get_capabilities(provider_id)
            if not caps:
                continue

            # Calculate diversity score (prefer different provider types)
            diversity_score = (
                1.0  # Would be calculated based on provider characteristics
            )

            # Get quality score from metrics
            quality_score = metrics.get(provider_id, {}).get("success_rate", 0.8)
            # Combined score
            combined_score = diversity_score * 0.3 + quality_score * 0.7
            provider_scores.append((provider_id, combined_score))
        # Sort by score and select top providers
        provider_scores.sort(key=lambda x: cast(float, x[1]), reverse=True)
        selected_providers = [p[0] for p in provider_scores[: self.min_providers]]

        if len(selected_providers) < self.min_providers:
            # Not enough providers, fall back to single provider
            selected = (
                selected_providers[0] if selected_providers else available_providers[0]
            )
            confidence = 0.6
            reasoning = (
                f"Insufficient providers for consensus, "
                f"using single provider {selected}"
            )
        else:
            # Return first provider as primary, others as alternatives
            selected = selected_providers[0]
            confidence = 0.9  # High confidence due to consensus approach
            reasoning = (
                f"Selected {self.min_providers} providers for consensus: "
                f"{', '.join(selected_providers)}"
            )

        return RoutingDecision(
            provider_id=selected,
            confidence=confidence,
            reasoning=reasoning,
            estimated_cost=0.0,
            estimated_duration=0.0,
            metadata={
                "consensus_providers": selected_providers,
                "min_providers": self.min_providers,
                "consensus_threshold": self.consensus_threshold,
            },
        )


class AdaptiveRouting(RoutingStrategy):
    """Routing strategy that adapts based on historical performance."""

    def __init__(
        self,
        provider_registry: 'ProviderRegistry',
        base_strategy: RoutingStrategy,
        learning_rate: float = 0.1,
    ) -> None:
        self.provider_registry = provider_registry
        self.base_strategy = base_strategy
        self.learning_rate = learning_rate
        # Dynamic weights for each provider
        self.provider_weights: Dict[str, float] = {}

    async def route(
        self,
        task: str,
        characteristics: TaskCharacteristics,
        available_providers: List[str],
        context: Dict[str, Any],
        metrics: Dict[str, Dict[str, float]],
    ) -> RoutingDecision:
        """Route with adaptive learning from past performance."""
        # Get base decision
        base_decision = await self.base_strategy.route(
            task,
            characteristics,
            available_providers,
            context,
            metrics,
        )
        # Apply learned weights
        adjusted_scores = {}
        for provider_id in available_providers:
            base_score = (
                base_decision.confidence
                if provider_id == base_decision.provider_id
                else 0.5
            )

            # Apply learned weight
            weight = self.provider_weights.get(provider_id, 1.0)
            adjusted_score = base_score * weight

            # Apply recent performance adjustment
            if provider_id in metrics:
                recent_success = metrics[provider_id].get(
                    "recent_success_rate", metrics[provider_id].get("success_rate", 1.0)
                )
                adjusted_score *= recent_success

            adjusted_scores[provider_id] = adjusted_score

        # Select best provider after adjustments
        selected = max(adjusted_scores, key=lambda k: adjusted_scores[k])
        confidence = adjusted_scores[selected]

        reasoning = (
            f"Adaptive routing selected {selected} based on learned "
            f"performance (base: {base_decision.provider_id})"
        )

        return RoutingDecision(
            provider_id=selected,
            confidence=confidence,
            reasoning=reasoning,
            estimated_cost=0.0,
            estimated_duration=0.0,
            metadata={
                "base_decision": base_decision.provider_id,
                "adjusted_scores": adjusted_scores,
                "provider_weights": dict(self.provider_weights),
            },
        )

    def update_weights(self, provider_id: str, success: bool) -> None:
        """Update provider weights based on execution results."""
        current_weight = self.provider_weights.get(provider_id, 1.0)
        # Update weight based on success/failure
        if success:
            new_weight = current_weight + self.learning_rate * (1.5 - current_weight)
        else:
            new_weight = current_weight - self.learning_rate * current_weight

        # Clamp weight to reasonable range
        self.provider_weights[provider_id] = max(0.1, min(2.0, new_weight))
        logger.debug(
            f"Updated weight for {provider_id}: "
            f"{current_weight:.2f} -> {new_weight:.2f}"
        )

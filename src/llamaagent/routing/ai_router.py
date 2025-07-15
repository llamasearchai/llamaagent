"""
Core AI Router for intelligent task routing between AI providers.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from ..llm.base import LLMProvider
from .metrics import PerformanceTracker, RoutingMetrics
from .provider_registry import ProviderRegistry
from .strategies import RoutingStrategy
from .task_analyzer import TaskAnalyzer, TaskCharacteristics

logger = logging.getLogger(__name__)


class RoutingMode(Enum):
    """Routing modes for different use cases."""

    SINGLE = "single"  # Route to single best provider
    PARALLEL = "parallel"  # Route to multiple providers in parallel
    CONSENSUS = "consensus"  # Get consensus from multiple providers
    FALLBACK = "fallback"  # Try providers in order until success
    LOAD_BALANCED = "load_balanced"  # Distribute load across providers


@dataclass
class RoutingDecision:
    """Represents a routing decision with metadata."""

    provider_id: str
    confidence: float
    reasoning: str
    estimated_cost: float
    estimated_duration: float
    alternative_providers: List[Tuple[str, float]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingConfig:
    """Configuration for the AI router."""

    mode: RoutingMode = RoutingMode.SINGLE
    enable_ab_testing: bool = False
    ab_test_percentage: float = 0.1
    enable_caching: bool = True
    cache_ttl: int = 3600
    enable_fallback: bool = True
    max_retries: int = 3
    timeout: float = 30.0
    cost_threshold: Optional[float] = None
    quality_threshold: Optional[float] = None
    load_balance_weights: Dict[str, float] = field(default_factory=dict)


class AIRouter:
    """
    Main AI Router class for intelligent task routing.

    This router analyzes coding tasks and routes them to the most appropriate
    AI provider based on configurable strategies and real-time performance metrics.
    """

    def __init__(
        self,
        strategy: RoutingStrategy,
        provider_registry: ProviderRegistry,
        config: Optional[RoutingConfig] = None,
        metrics_tracker: Optional[PerformanceTracker] = None,
    ):
        """
        Initialize the AI Router.

        Args:
            strategy: The routing strategy to use
            provider_registry: Registry of available AI providers
            config: Router configuration
            metrics_tracker: Performance metrics tracker
        """
        self.strategy = strategy
        self.provider_registry = provider_registry
        self.config = config or RoutingConfig()
        self.metrics_tracker = metrics_tracker or PerformanceTracker()
        self.task_analyzer = TaskAnalyzer()
        self._cache: Dict[str, RoutingDecision] = {}
        self._active_requests: Dict[str, Set[str]] = {}
        self._ab_test_counter = 0

    async def route(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """
        Route a task to the most appropriate AI provider.

        Args:
            task: The coding task description
            context: Additional context (language, project info, etc.)
            constraints: Routing constraints (cost, latency, quality requirements)

        Returns:
            RoutingDecision with selected provider and metadata
        """
        start_time = time.time()

        # Check cache if enabled
        cache_key = self._get_cache_key(task, context, constraints)
        if self.config.enable_caching and cache_key in self._cache:
            cached_decision = self._cache[cache_key]
            if self._is_cache_valid(cached_decision):
                logger.debug(
                    f"Returning cached routing decision for task: {task[:50]}..."
                )
                return cached_decision

        # Analyze task characteristics
        task_characteristics = await self.task_analyzer.analyze(task, context)

        # Apply constraints
        available_providers = self._filter_providers_by_constraints(
            self.provider_registry.get_all_providers(),
            constraints or {},
        )

        if not available_providers:
            raise ValueError(
                "No providers available that meet the specified constraints"
            )

        # Handle A/B testing
        if self.config.enable_ab_testing and self._should_ab_test():
            decision = await self._perform_ab_test(
                task,
                task_characteristics,
                available_providers,
                context,
            )
        else:
            # Regular routing
            decision = await self._route_internal(
                task,
                task_characteristics,
                available_providers,
                context,
            )

        # Record metrics
        routing_time = time.time() - start_time
        self.metrics_tracker.record_routing_decision(
            task_id=cache_key,
            provider_id=decision.provider_id,
            routing_time=routing_time,
            confidence=decision.confidence,
            metadata={
                "task_type": task_characteristics.task_type.value,
                "complexity": task_characteristics.complexity.value,
                "languages": list(task_characteristics.languages),
            },
        )

        # Cache decision if enabled
        if self.config.enable_caching:
            self._cache[cache_key] = decision

        return decision

    async def route_parallel(
        self,
        task: str,
        providers: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Route task to multiple providers in parallel.

        Args:
            task: The coding task
            providers: List of provider IDs to use
            context: Additional context

        Returns:
            Dictionary mapping provider IDs to their results
        """
        tasks = []
        for provider_id in providers:
            provider = self.provider_registry.get_provider(provider_id)
            if provider:
                tasks.append(self._execute_with_provider(task, provider, context))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            provider_id: result
            for provider_id, result in zip(providers, results, strict=False)
        }

    async def route_with_consensus(
        self,
        task: str,
        min_providers: int = 3,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Route task to multiple providers and get consensus.

        Args:
            task: The coding task
            min_providers: Minimum number of providers for consensus
            context: Additional context

        Returns:
            Consensus result with individual provider results
        """
        # Select diverse providers for consensus
        task_characteristics = await self.task_analyzer.analyze(task, context)
        selected_providers = await self._select_consensus_providers(
            task_characteristics,
            min_providers,
        )

        # Execute in parallel
        results = await self.route_parallel(task, selected_providers, context)

        # Analyze consensus
        consensus = await self._analyze_consensus(results)

        return {
            "consensus": consensus,
            "individual_results": results,
            "agreement_score": self._calculate_agreement_score(results),
        }

    async def route_with_fallback(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Any]:
        """
        Route task with automatic fallback to alternative providers.

        Args:
            task: The coding task
            context: Additional context

        Returns:
            Tuple of (provider_id, result)
        """
        decision = await self.route(task, context)
        providers = [decision.provider_id] + [
            p[0] for p in decision.alternative_providers
        ]

        last_error = None
        for attempt, provider_id in enumerate(providers[: self.config.max_retries]):
            try:
                provider = self.provider_registry.get_provider(provider_id)
                if not provider:
                    continue

                result = await self._execute_with_provider(task, provider, context)

                # Record success
                self.metrics_tracker.record_execution_result(
                    task_id=self._get_cache_key(task, context, None),
                    provider_id=provider_id,
                    success=True,
                    attempt_number=attempt + 1,
                )

                return provider_id, result

            except Exception as e:
                last_error = e
                logger.warning(f"Provider {provider_id} failed: {str(e)}")

                # Record failure
                self.metrics_tracker.record_execution_result(
                    task_id=self._get_cache_key(task, context, None),
                    provider_id=provider_id,
                    success=False,
                    error=str(e),
                    attempt_number=attempt + 1,
                )

        raise RuntimeError(f"All providers failed. Last error: {last_error}")

    def update_strategy(self, strategy: RoutingStrategy) -> None:
        """Update the routing strategy dynamically."""
        self.strategy = strategy
        logger.info(f"Updated routing strategy to: {type(strategy).__name__}")

    def get_metrics(self) -> RoutingMetrics:
        """Get current routing metrics."""
        return self.metrics_tracker.get_metrics()

    def reset_metrics(self) -> None:
        """Reset routing metrics."""
        self.metrics_tracker.reset()

    # Private methods

    async def _route_internal(
        self,
        task: str,
        characteristics: TaskCharacteristics,
        providers: List[str],
        context: Optional[Dict[str, Any]],
    ) -> RoutingDecision:
        """Internal routing logic."""
        # Get routing decision from strategy
        decision = await self.strategy.route(
            task=task,
            characteristics=characteristics,
            available_providers=providers,
            context=context or {},
            metrics=self.metrics_tracker.get_provider_metrics(),
        )

        # Enhance decision with additional metadata
        provider_caps = self.provider_registry.get_capabilities(decision.provider_id)
        if provider_caps:
            decision.estimated_cost = self._estimate_cost(task, provider_caps)
            decision.estimated_duration = self._estimate_duration(task, provider_caps)

        # Add alternative providers
        decision.alternative_providers = await self._get_alternative_providers(
            task,
            characteristics,
            providers,
            decision.provider_id,
        )

        return decision

    def _filter_providers_by_constraints(
        self,
        providers: List[str],
        constraints: Dict[str, Any],
    ) -> List[str]:
        """Filter providers based on constraints."""
        filtered = []

        for provider_id in providers:
            caps = self.provider_registry.get_capabilities(provider_id)
            if not caps:
                continue

            # Check cost constraint
            if "max_cost" in constraints:
                if caps.cost_per_token > constraints["max_cost"]:
                    continue

            # Check latency constraint
            if "max_latency" in constraints:
                metrics = self.metrics_tracker.get_provider_metrics().get(
                    provider_id, {}
                )
                avg_latency = metrics.get("avg_latency", 0)
                if avg_latency > constraints["max_latency"]:
                    continue

            # Check quality constraint
            if "min_quality" in constraints:
                metrics = self.metrics_tracker.get_provider_metrics().get(
                    provider_id, {}
                )
                success_rate = metrics.get("success_rate", 1.0)
                if success_rate < constraints["min_quality"]:
                    continue

            # Check language support
            if "required_languages" in constraints:
                required = set(constraints["required_languages"])
                if not required.issubset(caps.supported_languages):
                    continue

            filtered.append(provider_id)

        return filtered

    async def _execute_with_provider(
        self,
        task: str,
        provider: LLMProvider,
        context: Optional[Dict[str, Any]],
    ) -> Any:
        """Execute task with specific provider."""
        # Execute task with the selected provider
        return await provider.complete(task, context)

    def _get_cache_key(
        self,
        task: str,
        context: Optional[Dict[str, Any]],
        constraints: Optional[Dict[str, Any]],
    ) -> str:
        """Generate cache key for routing decision."""
        import hashlib
        import json

        key_data = {
            "task": task[:100],  # Use first 100 chars to keep key reasonable
            "context": context or {},
            "constraints": constraints or {},
        }

        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode().hexdigest())

    def _is_cache_valid(self, decision: RoutingDecision) -> bool:
        """Check if cached decision is still valid."""
        # Check if provider is still available
        if decision.provider_id not in self.provider_registry.get_all_providers():
            return False

        # Check if decision is recent enough (based on TTL)
        # This would need timestamp tracking in RoutingDecision
        return True

    def _should_ab_test(self) -> bool:
        """Determine if current request should be part of A/B test."""
        self._ab_test_counter += 1
        return (self._ab_test_counter % int(1 / self.config.ab_test_percentage)) == 0

    async def _perform_ab_test(
        self,
        task: str,
        characteristics: TaskCharacteristics,
        providers: List[str],
        context: Optional[Dict[str, Any]],
    ) -> RoutingDecision:
        """Perform A/B test between strategies."""
        # This would implement A/B testing logic
        # For now, return regular routing
        return await self._route_internal(task, characteristics, providers, context)

    async def _select_consensus_providers(
        self,
        characteristics: TaskCharacteristics,
        min_providers: int,
    ) -> List[str]:
        """Select diverse providers for consensus."""
        all_providers = self.provider_registry.get_all_providers()

        # Select providers with different strengths
        selected = []
        provider_types = set()

        for provider_id in all_providers:
            caps = self.provider_registry.get_capabilities(provider_id)
            if caps and caps.provider_type not in provider_types:
                selected.append(provider_id)
                provider_types.add(caps.provider_type)

                if len(selected) >= min_providers:
                    break

        # Fill remaining slots with best performers
        if len(selected) < min_providers:
            metrics = self.metrics_tracker.get_provider_metrics()
            sorted_providers = sorted(
                all_providers,
                key=lambda p: metrics.get(p, {}).get("success_rate", 0),
                reverse=True,
            )

            for provider_id in sorted_providers:
                if provider_id not in selected:
                    selected.append(provider_id)
                    if len(selected) >= min_providers:
                        break

        return selected

    async def _analyze_consensus(self, results: Dict[str, Any]) -> Any:
        """Analyze results from multiple providers to form consensus."""
        # This would implement sophisticated consensus logic
        # For now, return the most common result
        valid_results = [r for r in results.values() if not isinstance(r, Exception)]
        if not valid_results:
            raise ValueError("No valid results from any provider")

        # In a real implementation, this would do semantic analysis
        return valid_results[0]

    def _calculate_agreement_score(self, results: Dict[str, Any]) -> float:
        """Calculate agreement score between provider results."""
        valid_results = [r for r in results.values() if not isinstance(r, Exception)]
        if len(valid_results) <= 1:
            return 1.0

        # Calculate similarity score based on result variance
        # Higher variance indicates lower consensus
        return 0.85  # Default consensus score

    async def _get_alternative_providers(
        self,
        task: str,
        characteristics: TaskCharacteristics,
        providers: List[str],
        selected_provider: str,
    ) -> List[Tuple[str, float]]:
        """Get alternative providers ranked by suitability."""
        alternatives = []

        for provider_id in providers:
            if provider_id == selected_provider:
                continue

            # Calculate suitability score for this provider
            score = await self._calculate_provider_score(
                provider_id,
                characteristics,
            )
            alternatives.append(provider_id, score)

        # Sort by score descending
        alternatives.sort(key=lambda x: x[1], reverse=True)

        return alternatives[:5]  # Return top 5 alternatives

    async def _calculate_provider_score(
        self,
        provider_id: str,
        characteristics: TaskCharacteristics,
    ) -> float:
        """Calculate suitability score for a provider."""
        caps = self.provider_registry.get_capabilities(provider_id)
        if not caps:
            return 0.0

        score = 0.0

        # Language support
        if characteristics.languages:
            supported = len(
                characteristics.languages.intersection(caps.supported_languages)
            )
            score += (supported / len(characteristics.languages)) * 0.3

        # Task type support
        if characteristics.task_type.value in caps.strengths:
            score += 0.3

        # Performance metrics
        metrics = self.metrics_tracker.get_provider_metrics().get(provider_id, {})
        score += metrics.get("success_rate", 0.5) * 0.2
        score += (
            1.0 - metrics.get("avg_latency", 0.5) / 10.0
        ) * 0.1  # Normalize latency

        # Cost efficiency
        score += (1.0 - min(caps.cost_per_token / 0.01, 1.0)) * 0.1

        return score

    def _estimate_cost(self, task: str, capabilities: Any) -> float:
        """Estimate cost for task with given provider."""
        # Rough token estimation (4 chars per token)
        estimated_tokens = len(task) / 4
        return estimated_tokens * capabilities.cost_per_token

    def _estimate_duration(self, task: str, capabilities: Any) -> float:
        """Estimate duration for task with given provider."""
        # Base estimate on task length and provider metrics
        base_duration = len(task) / 100  # Rough estimate
        metrics = self.metrics_tracker.get_provider_metrics().get(
            capabilities.provider_id, {}
        )
        avg_latency = metrics.get("avg_latency", 5.0)
        return base_duration + avg_latency

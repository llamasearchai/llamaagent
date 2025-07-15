"""
Comprehensive tests for the AI routing system.
"""

import pytest

from llamaagent.routing import (
    AIRouter,
    PerformanceTracker,
    ProviderRegistry,
    RoutingDecision,
    TaskAnalyzer,
    TaskCharacteristics,
)
from llamaagent.routing.ai_router import RoutingConfig, RoutingMode
from llamaagent.routing.provider_registry import ProviderStatus, ProviderType
from llamaagent.routing.strategies import (
    AdaptiveRouting,
    ConsensusRouting,
    CostOptimizedRouting,
    HybridRouting,
    LanguageBasedRouting,
    PerformanceBasedRouting,
    TaskBasedRouting,
)
from llamaagent.routing.task_analyzer import TaskComplexity, TaskType


class TestTaskAnalyzer:
    """Test task analyzer functionality."""

    @pytest.mark.asyncio
    async def test_task_type_detection(self):
        """Test detection of different task types."""
        analyzer = TaskAnalyzer()

        # Test debugging task
        debug_task = (
            "Fix the error in the login function that causes a null pointer exception"
        )
        characteristics = await analyzer.analyze(debug_task)
        assert characteristics.task_type == TaskType.DEBUGGING
        assert characteristics.confidence > 0.5

        # Test refactoring task
        refactor_task = "Refactor the User class to follow SOLID principles"
        characteristics = await analyzer.analyze(refactor_task)
        assert characteristics.task_type == TaskType.REFACTORING

        # Test code generation task
        gen_task = "Create a new REST API endpoint for user registration"
        characteristics = await analyzer.analyze(gen_task)
        assert characteristics.task_type == TaskType.CODE_GENERATION

    @pytest.mark.asyncio
    async def test_language_detection(self):
        """Test programming language detection."""
        analyzer = TaskAnalyzer()

        # Test Python detection
        python_task = "Write a Python function using pandas to analyze CSV data"
        characteristics = await analyzer.analyze(python_task)
        assert "python" in characteristics.languages
        assert "pandas" in characteristics.frameworks

        # Test multiple languages
        multi_task = "Convert this JavaScript code to TypeScript with proper types"
        characteristics = await analyzer.analyze(multi_task)
        assert "javascript" in characteristics.languages
        assert "typescript" in characteristics.languages

    @pytest.mark.asyncio
    async def test_complexity_detection(self):
        """Test task complexity detection."""
        analyzer = TaskAnalyzer()

        # Simple task
        simple_task = "Add a comment to explain this function"
        characteristics = await analyzer.analyze(simple_task)
        assert characteristics.complexity in [
            TaskComplexity.TRIVIAL,
            TaskComplexity.SIMPLE,
        ]

        # Complex task
        complex_task = """
        Implement a distributed caching system with Redis that handles 
        concurrent requests, implements LRU eviction, supports clustering,
        and includes monitoring with Prometheus metrics.
        """
        characteristics = await analyzer.analyze(complex_task)
        assert characteristics.complexity in [
            TaskComplexity.COMPLEX,
            TaskComplexity.VERY_COMPLEX,
        ]
        assert characteristics.requires_external_apis
        assert characteristics.requires_database


class TestProviderRegistry:
    """Test provider registry functionality."""

    def test_default_providers(self):
        """Test that default providers are initialized."""
        registry = ProviderRegistry()

        # Check default providers exist
        assert "claude-code" in registry.get_all_providers()
        assert "openai-codex" in registry.get_all_providers()
        assert "github-copilot" in registry.get_all_providers()
        assert "local-codellama" in registry.get_all_providers()

    def test_provider_capabilities(self):
        """Test provider capability management."""
        registry = ProviderRegistry()

        # Get Claude capabilities
        claude = registry.get_capabilities("claude-code")
        assert claude is not None
        assert claude.provider_type == ProviderType.CLOUD
        assert "python" in claude.supported_languages
        assert claude.supports_vision
        assert claude.max_tokens > 50000

    def test_provider_filtering(self):
        """Test filtering providers by criteria."""
        registry = ProviderRegistry()

        # Filter by language
        python_providers = registry.get_providers_by_language("python")
        assert len(python_providers) >= 4
        assert "claude-code" in python_providers

        # Filter by task type
        debug_providers = registry.get_providers_by_task_type("debugging")
        assert len(debug_providers) > 0
        assert debug_providers[0][1] > 0.5  # Score should be reasonable

        # Filter by cost
        cheap_providers = registry.get_providers_by_cost(max_cost=0.00002)
        assert len(cheap_providers) > 0
        assert "local-codellama" in [p[0] for p in cheap_providers]

    def test_provider_status_update(self):
        """Test updating provider status."""
        registry = ProviderRegistry()

        # Update status
        assert registry.update_provider_status(
            "claude-code", ProviderStatus.MAINTENANCE
        )
        claude = registry.get_capabilities("claude-code")
        assert claude.status == ProviderStatus.MAINTENANCE

        # Get active providers shouldn't include maintenance
        active = registry.get_active_providers()
        assert "claude-code" not in active


class TestRoutingStrategies:
    """Test different routing strategies."""

    @pytest.mark.asyncio
    async def test_task_based_routing(self):
        """Test task-based routing strategy."""
        registry = ProviderRegistry()
        strategy = TaskBasedRouting(registry)

        task = "Debug the null pointer exception in the user service"
        characteristics = TaskCharacteristics(
            task_type=TaskType.DEBUGGING,
            complexity=TaskComplexity.MODERATE,
            languages={"java"},
        )

        decision = await strategy.route(
            task=task,
            characteristics=characteristics,
            available_providers=["claude-code", "openai-codex", "github-copilot"],
            context={},
            metrics={},
        )

        assert decision.provider_id in ["claude-code", "openai-codex"]
        assert decision.confidence > 0.5
        assert "debugging" in decision.reasoning.lower()

    @pytest.mark.asyncio
    async def test_language_based_routing(self):
        """Test language-based routing strategy."""
        registry = ProviderRegistry()
        strategy = LanguageBasedRouting(registry)

        task = "Write a React component with TypeScript"
        characteristics = TaskCharacteristics(
            task_type=TaskType.CODE_GENERATION,
            complexity=TaskComplexity.SIMPLE,
            languages={"javascript", "typescript"},
            frameworks={"react"},
        )

        decision = await strategy.route(
            task=task,
            characteristics=characteristics,
            available_providers=["claude-code", "openai-codex", "github-copilot"],
            context={},
            metrics={},
        )

        assert decision.provider_id in ["github-copilot", "openai-codex"]
        assert (
            "javascript" in decision.reasoning.lower()
            or "typescript" in decision.reasoning.lower()
        )

    @pytest.mark.asyncio
    async def test_cost_optimized_routing(self):
        """Test cost-optimized routing strategy."""
        registry = ProviderRegistry()
        strategy = CostOptimizedRouting(registry, quality_threshold=0.7)

        task = "Add logging to this function"
        characteristics = TaskCharacteristics(
            task_type=TaskType.CODE_GENERATION,
            complexity=TaskComplexity.TRIVIAL,
        )

        # Mock metrics with quality scores
        metrics = {
            "local-codellama": {"success_rate": 0.75},
            "github-copilot": {"success_rate": 0.85},
            "claude-code": {"success_rate": 0.92},
        }

        decision = await strategy.route(
            task=task,
            characteristics=characteristics,
            available_providers=["local-codellama", "github-copilot", "claude-code"],
            context={},
            metrics=metrics,
        )

        # Should select local model as it meets quality threshold and is free
        assert decision.provider_id == "local-codellama"
        assert "cost" in decision.reasoning.lower()

    @pytest.mark.asyncio
    async def test_hybrid_routing(self):
        """Test hybrid routing strategy."""
        registry = ProviderRegistry()

        # Create hybrid strategy combining task and language routing
        task_strategy = TaskBasedRouting(registry)
        lang_strategy = LanguageBasedRouting(registry)

        hybrid = HybridRouting(
            [
                (task_strategy, 0.6),
                (lang_strategy, 0.4),
            ]
        )

        task = "Refactor this Python class to improve performance"
        characteristics = TaskCharacteristics(
            task_type=TaskType.REFACTORING,
            complexity=TaskComplexity.MODERATE,
            languages={"python"},
        )

        decision = await hybrid.route(
            task=task,
            characteristics=characteristics,
            available_providers=["claude-code", "openai-codex", "github-copilot"],
            context={},
            metrics={},
        )

        assert decision.provider_id in ["claude-code", "openai-codex"]
        assert "hybrid" in decision.reasoning.lower()


class TestAIRouter:
    """Test main AI router functionality."""

    @pytest.mark.asyncio
    async def test_basic_routing(self):
        """Test basic routing functionality."""
        registry = ProviderRegistry()
        strategy = TaskBasedRouting(registry)
        router = AIRouter(strategy, registry)

        task = "Fix the bug in the authentication module"
        decision = await router.route(task)

        assert isinstance(decision, RoutingDecision)
        assert decision.provider_id in registry.get_all_providers()
        assert decision.confidence > 0
        assert decision.reasoning != ""

    @pytest.mark.asyncio
    async def test_routing_with_constraints(self):
        """Test routing with constraints."""
        registry = ProviderRegistry()
        strategy = PerformanceBasedRouting(registry)
        config = RoutingConfig(cost_threshold=0.00002)
        router = AIRouter(strategy, registry, config)

        task = "Generate unit tests for the UserService class"
        constraints = {
            "max_cost": 0.00002,
            "required_languages": ["java"],
        }

        decision = await router.route(task, constraints=constraints)

        # Should not select expensive providers
        assert decision.provider_id != "claude-code"  # Too expensive
        assert decision.provider_id in ["github-copilot", "local-codellama"]

    @pytest.mark.asyncio
    async def test_routing_with_fallback(self):
        """Test routing with fallback mechanism."""
        registry = ProviderRegistry()
        strategy = TaskBasedRouting(registry)
        config = RoutingConfig(enable_fallback=True, max_retries=3)
        router = AIRouter(strategy, registry, config)

        # This would test fallback in real implementation
        # For now, just verify the method exists
        task = "Complex refactoring task"
        result = await router.route(task)
        assert result.alternative_providers is not None
        assert len(result.alternative_providers) > 0

    @pytest.mark.asyncio
    async def test_parallel_routing(self):
        """Test parallel routing to multiple providers."""
        registry = ProviderRegistry()
        strategy = ConsensusRouting(registry)
        router = AIRouter(strategy, registry)

        task = "Analyze this code for security vulnerabilities"

        # Note: In real implementation, this would call actual providers
        # For testing, we just verify the structure
        decision = await router.route(task)
        assert "consensus" in decision.metadata


class TestPerformanceTracking:
    """Test performance tracking and metrics."""

    def test_routing_metrics(self):
        """Test recording routing decisions."""
        tracker = PerformanceTracker()

        # Record routing decisions
        tracker.record_routing_decision(
            task_id="task1",
            provider_id="claude-code",
            routing_time=0.1,
            confidence=0.9,
            strategy="task_based",
            metadata={"task_type": "debugging"},
        )

        tracker.record_routing_decision(
            task_id="task2",
            provider_id="openai-codex",
            routing_time=0.05,
            confidence=0.85,
            strategy="language_based",
            metadata={"languages": ["python", "javascript"]},
        )

        metrics = tracker.get_metrics()
        assert metrics.total_routing_decisions == 2
        assert metrics.avg_routing_confidence == pytest.approx(0.875)
        assert "task_based" in metrics.routing_strategy_usage
        assert metrics.task_type_distribution["debugging"] == 1

    def test_execution_tracking(self):
        """Test tracking execution results."""
        tracker = PerformanceTracker()

        # Track successful execution
        tracker.start_execution("task1", "claude-code")
        tracker.record_execution_result(
            task_id="task1",
            provider_id="claude-code",
            success=True,
            latency=2.5,
            tokens_used=1000,
            cost=0.03,
            metadata={"task_type": "code_generation"},
        )

        # Track failed execution
        tracker.record_execution_result(
            task_id="task2",
            provider_id="openai-codex",
            success=False,
            latency=1.0,
            error="API rate limit exceeded",
        )

        provider_metrics = tracker.get_provider_metrics()

        assert "claude-code" in provider_metrics
        assert provider_metrics["claude-code"]["success_rate"] == 1.0
        assert provider_metrics["claude-code"]["avg_latency"] == 2500.0

        assert "openai-codex" in provider_metrics
        assert provider_metrics["openai-codex"]["success_rate"] == 0.0

    def test_cost_analysis(self):
        """Test cost analysis functionality."""
        tracker = PerformanceTracker()

        # Record some executions with costs
        for i in range(5):
            tracker.record_execution_result(
                task_id=f"task{i}",
                provider_id="claude-code",
                success=True,
                tokens_used=1000,
                cost=0.03,
            )

        for i in range(10):
            tracker.record_execution_result(
                task_id=f"task{i + 5}",
                provider_id="github-copilot",
                success=True,
                tokens_used=500,
                cost=0.005,
            )

        cost_analysis = tracker.get_cost_analysis()

        assert cost_analysis["total_cost"] == pytest.approx(0.2)  # 5*0.03 + 10*0.005
        assert cost_analysis["total_tokens"] == 10000  # 5*1000 + 10*500
        assert "claude-code" in cost_analysis["provider_costs"]
        assert cost_analysis["provider_costs"]["claude-code"] == pytest.approx(0.15)

    def test_error_analysis(self):
        """Test error analysis functionality."""
        tracker = PerformanceTracker()

        # Record various errors
        errors = [
            "RateLimitError: API rate limit exceeded",
            "AuthenticationError: Invalid API key",
            "RateLimitError: API rate limit exceeded",
            "TimeoutError: Request timed out",
        ]

        for i, error in enumerate(errors):
            tracker.record_execution_result(
                task_id=f"task{i}",
                provider_id="openai-codex",
                success=False,
                error=error,
            )

        error_analysis = tracker.get_error_analysis()

        assert "openai-codex" in error_analysis
        assert error_analysis["openai-codex"]["total_errors"] == 4
        assert error_analysis["openai-codex"]["error_types"]["RateLimitError"] == 2
        assert error_analysis["openai-codex"]["most_common_error"] == "RateLimitError"


class TestAdaptiveRouting:
    """Test adaptive routing with learning."""

    @pytest.mark.asyncio
    async def test_adaptive_learning(self):
        """Test that adaptive routing learns from results."""
        registry = ProviderRegistry()
        base_strategy = TaskBasedRouting(registry)
        adaptive = AdaptiveRouting(registry, base_strategy, learning_rate=0.2)

        # Simulate successful execution
        adaptive.update_weights("claude-code", success=True)
        adaptive.update_weights("claude-code", success=True)

        # Simulate failed execution
        adaptive.update_weights("openai-codex", success=False)

        # Check weights were updated
        assert adaptive.provider_weights["claude-code"] > 1.0
        assert adaptive.provider_weights["openai-codex"] < 1.0

        # Test routing with updated weights
        task = "Debug this code"
        characteristics = TaskCharacteristics(
            task_type=TaskType.DEBUGGING,
            complexity=TaskComplexity.MODERATE,
        )

        decision = await adaptive.route(
            task=task,
            characteristics=characteristics,
            available_providers=["claude-code", "openai-codex"],
            context={},
            metrics={},
        )

        # Should prefer claude-code due to positive weight
        assert decision.provider_id == "claude-code"


@pytest.mark.asyncio
async def test_end_to_end_routing():
    """Test complete routing flow from task to execution."""
    # Initialize components
    registry = ProviderRegistry()
    tracker = PerformanceTracker()

    # Create hybrid strategy
    strategies = [
        (TaskBasedRouting(registry), 0.4),
        (LanguageBasedRouting(registry), 0.3),
        (PerformanceBasedRouting(registry), 0.3),
    ]
    hybrid_strategy = HybridRouting(strategies)

    # Create router with configuration
    config = RoutingConfig(
        mode=RoutingMode.SINGLE,
        enable_caching=True,
        enable_fallback=True,
        cost_threshold=0.05,
    )

    router = AIRouter(hybrid_strategy, registry, config, tracker)

    # Test various tasks
    tasks = [
        ("Fix the memory leak in the Java application", {"project": "backend"}),
        ("Create a React component for user profile", {"framework": "react"}),
        (
            "Optimize this Python data processing pipeline",
            {"performance_critical": True},
        ),
        ("Write comprehensive tests for the API endpoints", {"language": "typescript"}),
    ]

    for task, context in tasks:
        # Route the task
        decision = await router.route(task, context)

        # Verify decision
        assert decision.provider_id in registry.get_all_providers()
        assert decision.confidence > 0
        assert decision.estimated_cost >= 0
        assert decision.estimated_duration >= 0

        # Simulate execution
        tracker.start_execution(task[:20], decision.provider_id)
        tracker.record_execution_result(
            task_id=task[:20],
            provider_id=decision.provider_id,
            success=True,
            latency=2.0,
            tokens_used=500,
            cost=decision.estimated_cost,
        )

    # Check metrics
    metrics = tracker.get_metrics()
    assert metrics.total_routing_decisions >= 4
    assert metrics.total_executions >= 4
    assert metrics.successful_executions >= 4

    # Check provider distribution
    distribution = tracker.get_load_distribution()
    assert len(distribution) >= 1
    assert sum(distribution.values()) == pytest.approx(1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

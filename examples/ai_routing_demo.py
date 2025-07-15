"""
Demonstration of the AI Routing System for intelligent task distribution.

This example shows how to:
1. Set up the routing system with different strategies
2. Route various coding tasks to appropriate AI providers
3. Track performance and adapt routing decisions
4. Analyze costs and performance metrics
"""

import asyncio
import logging
from pathlib import Path

from llamaagent.routing import (
    AIRouter,
    ProviderRegistry,
    PerformanceTracker,
)
from llamaagent.routing.ai_router import RoutingConfig, RoutingMode
from llamaagent.routing.strategies import (
    TaskBasedRouting,
    LanguageBasedRouting,
    ComplexityBasedRouting,
    PerformanceBasedRouting,
    CostOptimizedRouting,
    HybridRouting,
    ConsensusRouting,
    AdaptiveRouting,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def demo_basic_routing():
    """Demonstrate basic routing functionality."""
    print("\n=== Basic Routing Demo ===\n")
    
    # Initialize components
    registry = ProviderRegistry()
    tracker = PerformanceTracker()
    
    # Create a simple task-based routing strategy
    strategy = TaskBasedRouting(registry)
    
    # Create router
    router = AIRouter(strategy, registry, metrics_tracker=tracker)
    
    # Example tasks
    tasks = [
        "Debug the null pointer exception in the login function",
        "Refactor the UserService class to follow SOLID principles",
        "Create a new REST API endpoint for user profile updates",
        "Write comprehensive documentation for the payment module",
        "Optimize the database queries in the reporting service",
    ]
    
    for task in tasks:
        print(f"\nTask: {task}")
        decision = await router.route(task)
        print(f"Selected Provider: {decision.provider_id}")
        print(f"Confidence: {decision.confidence:.2f}")
        print(f"Reasoning: {decision.reasoning}")
        
        # Simulate execution
        tracker.start_execution(task[:20], decision.provider_id)
        # In real use, this would be the actual API call
        await asyncio.sleep(0.1)  # Simulate API latency
        
        # Record result
        tracker.record_execution_result(
            task_id=task[:20],
            provider_id=decision.provider_id,
            success=True,
            latency=0.1,
            tokens_used=len(task) * 10,  # Rough estimate
            cost=len(task) * 10 * 0.00003,  # Example cost calculation
        )


async def demo_language_based_routing():
    """Demonstrate language-based routing."""
    print("\n=== Language-Based Routing Demo ===\n")
    
    registry = ProviderRegistry()
    strategy = LanguageBasedRouting(registry)
    router = AIRouter(strategy, registry)
    
    # Tasks in different languages
    language_tasks = [
        ("Write a Python script to parse JSON and extract specific fields", {"language": "python"}),
        ("Create a TypeScript interface for the user authentication response", {"language": "typescript"}),
        ("Implement a sorting algorithm in Rust with zero-copy optimization", {"language": "rust"}),
        ("Build a React component with styled-components", {"language": "javascript", "framework": "react"}),
    ]
    
    for task, context in language_tasks:
        print(f"\nTask: {task}")
        print(f"Context: {context}")
        
        decision = await router.route(task, context)
        print(f"Selected Provider: {decision.provider_id}")
        print(f"Reasoning: {decision.reasoning}")


async def demo_cost_optimized_routing():
    """Demonstrate cost-optimized routing."""
    print("\n=== Cost-Optimized Routing Demo ===\n")
    
    registry = ProviderRegistry()
    tracker = PerformanceTracker()
    
    # Set up cost-optimized strategy with quality threshold
    strategy = CostOptimizedRouting(registry, quality_threshold=0.8)
    config = RoutingConfig(cost_threshold=0.01)  # Max $0.01 per request
    router = AIRouter(strategy, registry, config, tracker)
    
    # Simulate some historical performance data
    providers = ["local-codellama", "github-copilot", "openai-codex", "claude-code"]
    for provider in providers:
        for i in range(10):
            success = i < 8 if provider == "local-codellama" else i < 9
            tracker.record_execution_result(
                task_id=f"hist_{provider}_{i}",
                provider_id=provider,
                success=success,
                latency=0.5 if provider == "local-codellama" else 2.0,
                tokens_used=100,
                cost=0.0 if provider == "local-codellama" else 0.001,
            )
    
    # Route tasks with cost optimization
    tasks = [
        "Add a simple logging statement to this function",
        "Implement a complex caching strategy with Redis",
        "Fix a minor typo in the documentation",
    ]
    
    for task in tasks:
        print(f"\nTask: {task}")
        decision = await router.route(task)
        print(f"Selected Provider: {decision.provider_id}")
        print(f"Estimated Cost: ${decision.estimated_cost:.4f}")
        print(f"Reasoning: {decision.reasoning}")


async def demo_hybrid_routing():
    """Demonstrate hybrid routing strategy."""
    print("\n=== Hybrid Routing Demo ===\n")
    
    registry = ProviderRegistry()
    tracker = PerformanceTracker()
    
    # Create multiple strategies
    task_strategy = TaskBasedRouting(registry)
    language_strategy = LanguageBasedRouting(registry)
    performance_strategy = PerformanceBasedRouting(registry)
    
    # Combine them with weights
    hybrid_strategy = HybridRouting([
        (task_strategy, 0.4),      # 40% weight on task type
        (language_strategy, 0.3),   # 30% weight on language
        (performance_strategy, 0.3), # 30% weight on performance
    ])
    
    router = AIRouter(hybrid_strategy, registry, metrics_tracker=tracker)
    
    # Complex task that benefits from hybrid routing
    task = "Refactor this Python machine learning pipeline to improve performance and add type hints"
    context = {
        "language": "python",
        "frameworks": ["tensorflow", "pandas"],
        "performance_critical": True,
    }
    
    print(f"\nTask: {task}")
    print(f"Context: {context}")
    
    decision = await router.route(task, context)
    print(f"\nSelected Provider: {decision.provider_id}")
    print(f"Confidence: {decision.confidence:.2f}")
    print(f"Reasoning: {decision.reasoning}")
    
    # Show strategy contributions
    if "strategy_decisions" in decision.metadata:
        print("\nStrategy Contributions:")
        for strategy, provider, confidence in decision.metadata["strategy_decisions"]:
            print(f"  - {strategy}: {provider} (confidence: {confidence:.2f})")


async def demo_consensus_routing():
    """Demonstrate consensus routing with multiple providers."""
    print("\n=== Consensus Routing Demo ===\n")
    
    registry = ProviderRegistry()
    strategy = ConsensusRouting(registry, min_providers=3)
    config = RoutingConfig(mode=RoutingMode.CONSENSUS)
    router = AIRouter(strategy, registry, config)
    
    # Critical task that needs consensus
    task = "Review this authentication code for security vulnerabilities and suggest improvements"
    
    print(f"\nTask: {task}")
    decision = await router.route(task)
    
    print(f"\nPrimary Provider: {decision.provider_id}")
    print(f"Consensus Providers: {decision.metadata.get('consensus_providers', [])}")
    print(f"Reasoning: {decision.reasoning}")
    
    # In real use, you would call router.route_with_consensus()
    # to actually get results from multiple providers


async def demo_adaptive_routing():
    """Demonstrate adaptive routing that learns from results."""
    print("\n=== Adaptive Routing Demo ===\n")
    
    registry = ProviderRegistry()
    base_strategy = TaskBasedRouting(registry)
    adaptive_strategy = AdaptiveRouting(registry, base_strategy, learning_rate=0.2)
    router = AIRouter(adaptive_strategy, registry)
    
    # Simulate a series of tasks with feedback
    tasks = [
        ("Debug this Python function", "debugging", "claude-code", True),
        ("Generate JavaScript tests", "testing", "openai-codex", True),
        ("Debug another function", "debugging", "claude-code", True),
        ("Generate more tests", "testing", "openai-codex", False),  # Failure
        ("Debug complex issue", "debugging", "claude-code", True),
        ("Generate unit tests", "testing", "github-copilot", True),  # Try alternative
    ]
    
    print("\nSimulating task executions with feedback...")
    
    for task, task_type, expected_provider, success in tasks:
        print(f"\nTask: {task}")
        
        # Route the task
        decision = await router.route(task)
        print(f"Selected: {decision.provider_id}")
        
        # Simulate execution and update weights
        adaptive_strategy.update_weights(decision.provider_id, success)
        print(f"Result: {'Success' if success else 'Failed'}")
        
        # Show current weights
        if adaptive_strategy.provider_weights:
            print("Current provider weights:")
            for provider, weight in adaptive_strategy.provider_weights.items():
                print(f"  - {provider}: {weight:.2f}")


async def demo_performance_analysis():
    """Demonstrate performance analysis and metrics."""
    print("\n=== Performance Analysis Demo ===\n")
    
    registry = ProviderRegistry()
    tracker = PerformanceTracker()
    strategy = PerformanceBasedRouting(registry)
    router = AIRouter(strategy, registry, metrics_tracker=tracker)
    
    # Simulate a workload
    print("Simulating workload...")
    providers = ["claude-code", "openai-codex", "github-copilot", "local-codellama"]
    
    for i in range(50):
        task = f"Task {i}: Implement feature {i % 5}"
        provider = providers[i % len(providers)]
        
        # Record routing
        tracker.record_routing_decision(
            task_id=f"task_{i}",
            provider_id=provider,
            routing_time=0.01,
            confidence=0.8 + (i % 20) / 100,
            strategy="performance_based",
        )
        
        # Simulate execution
        success = (i % 10) != 0  # 90% success rate
        latency = 1.0 + (i % 5) * 0.5  # Variable latency
        cost = 0.001 if provider != "local-codellama" else 0.0
        
        tracker.record_execution_result(
            task_id=f"task_{i}",
            provider_id=provider,
            success=success,
            latency=latency,
            tokens_used=100 + i * 10,
            cost=cost,
        )
    
    # Analyze metrics
    print("\n=== Performance Metrics ===")
    
    # Overall metrics
    metrics = tracker.get_metrics()
    print(f"\nTotal Routing Decisions: {metrics.total_routing_decisions}")
    print(f"Total Executions: {metrics.total_executions}")
    print(f"Success Rate: {metrics.successful_executions / metrics.total_executions * 100:.1f}%")
    print(f"Average Routing Confidence: {metrics.avg_routing_confidence:.2f}")
    
    # Provider metrics
    print("\n=== Provider Performance ===")
    provider_metrics = tracker.get_provider_metrics()
    for provider, metrics in provider_metrics.items():
        print(f"\n{provider}:")
        print(f"  Success Rate: {metrics['success_rate'] * 100:.1f}%")
        print(f"  Avg Latency: {metrics['avg_latency']:.0f}ms")
        print(f"  P95 Latency: {metrics['p95_latency']:.0f}ms")
        print(f"  Avg Cost: ${metrics['avg_cost']:.4f}")
    
    # Cost analysis
    print("\n=== Cost Analysis ===")
    cost_analysis = tracker.get_cost_analysis()
    print(f"Total Cost: ${cost_analysis['total_cost']:.2f}")
    print(f"Total Tokens: {cost_analysis['total_tokens']:,}")
    print(f"Average Cost per Request: ${cost_analysis['avg_cost_per_request']:.4f}")
    
    print("\nCost per Provider:")
    for provider, cost in cost_analysis['provider_costs'].items():
        percentage = cost_analysis['cost_per_provider'][provider] * 100
        print(f"  {provider}: ${cost:.2f} ({percentage:.1f}%)")
    
    # Load distribution
    print("\n=== Load Distribution ===")
    distribution = tracker.get_load_distribution()
    for provider, percentage in distribution.items():
        print(f"  {provider}: {percentage * 100:.1f}%")
    
    # Error analysis
    print("\n=== Error Analysis ===")
    error_analysis = tracker.get_error_analysis()
    if error_analysis:
        for provider, errors in error_analysis.items():
            print(f"\n{provider}:")
            print(f"  Error Rate: {errors['error_rate'] * 100:.1f}%")
            print(f"  Total Errors: {errors['total_errors']}")
            if errors['error_types']:
                print(f"  Most Common: {errors['most_common_error']}")


async def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("AI Routing System Demonstration")
    print("=" * 60)
    
    # Run demonstrations
    await demo_basic_routing()
    await demo_language_based_routing()
    await demo_cost_optimized_routing()
    await demo_hybrid_routing()
    await demo_consensus_routing()
    await demo_adaptive_routing()
    await demo_performance_analysis()
    
    print("\n" + "=" * 60)
    print("Demonstration Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
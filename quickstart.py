#!/usr/bin/env python3
# pyright: reportGeneralTypeIssues=false, reportUnknownMemberType=false
"""
LlamaAgent Quickstart Demo

Author: Nik Jois <nikjois@llamasearch.ai>

This script demonstrates the complete LlamaAgent framework including:
- Agent creation and execution
- FastAPI endpoints  
- OpenAI integration
- SPRE planning
- Tool usage
- Database operations
- Testing automation
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, cast

# Core LlamaAgent imports
from src.llamaagent.agents.react import ReactAgent
from src.llamaagent.api.main import create_app
from src.llamaagent.config import AgentConfig
from src.llamaagent.config.settings import get_settings
from src.llamaagent.llm.factory import ProviderFactory
from src.llamaagent.llm.providers.base_provider import BaseLLMProvider
from src.llamaagent.tools.calculator import CalculatorTool
from src.llamaagent.tools.python_repl import PythonREPLTool
from src.llamaagent.tools.registry import ToolRegistry


def print_banner() -> None:
    """Print the LlamaAgent banner."""
    banner = """
                            
                  
                         
                         
                   
                         
    
    Advanced Multi-Agent AI Framework with SPRE Optimization
    Author: Nik Jois <nikjois@llamasearch.ai>
    """
    print(banner)


async def demo_basic_agent() -> None:
    """Demonstrate basic agent functionality."""
    print("\n=== Basic Agent Demo ===")
    
    # Create LLM provider with proper type annotation
    provider = cast(BaseLLMProvider, ProviderFactory.create_provider("mock"))
    print(f"Created LLM provider: {type(provider).__name__}")
    
    # Create agent configuration
    config = AgentConfig(
        name="DemoAgent",
        max_iterations=5,
        spree_enabled=True,
        debug=False
    )
    
    # Create agent
    agent = ReactAgent(config=config, llm_provider=provider)
    print(f"Created agent: {config.name}")
    
    # Execute simple task
    response = await agent.execute("What is 2 + 2?")
    print(f"Response: {response.content}")
    print(f"Execution time: {response.execution_time:.2f}s")
    print(f"Success: {response.success}")


async def demo_spre_planning() -> None:
    """Demonstrate SPRE (Strategic Planning & Resourceful Execution)."""
    print("\n=== SPRE Planning Demo ===")
    
    provider = cast(BaseLLMProvider, ProviderFactory.create_provider("mock"))
    
    config = AgentConfig(
        name="SPREAgent",
        spree_enabled=True,
        max_iterations=10
    )
    
    agent = ReactAgent(config=config, llm_provider=provider)
    
    # Complex task requiring planning
    task = "Calculate the factorial of 5 and then find the square root of that result"
    response = await agent.execute(task)
    
    print(f"Task: {task}")
    print(f"Response: {response.content}")
    
    if hasattr(response, 'plan') and response.plan:
        print(f"Plan steps: {len(response.plan.steps)}")
        for i, step in enumerate(response.plan.steps, 1):
            print(f"  {i}. {step.description}")


async def demo_tools_integration() -> None:
    """Demonstrate tool usage and integration."""
    print("\n=== Tools Integration Demo ===")
    
    # Create tool registry
    tool_registry = ToolRegistry()
    tool_registry.register(CalculatorTool())
    tool_registry.register(PythonREPLTool())
    
    print(f"Registered {len(tool_registry.list_tools())} tools:")
    for tool in tool_registry.list_tools():
        print(f"  - {tool.name}: {tool.description}")
    
    # Create agent with tools
    provider = cast(BaseLLMProvider, ProviderFactory.create_provider("mock"))
    config = AgentConfig(name="ToolAgent", spree_enabled=True)
    agent = ReactAgent(config=config, llm_provider=provider, tools=tool_registry)
    
    # Task requiring tool usage
    response = await agent.execute("Calculate 15 * 23 + 47")
    print(f"Calculator result: {response.content}")


async def demo_openai_integration() -> None:
    """Demonstrate OpenAI integration (requires API key)."""
    print("\n=== OpenAI Integration Demo ===")
    
    import os
    provider: BaseLLMProvider
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set, using mock provider")
        provider = cast(BaseLLMProvider, ProviderFactory.create_provider("mock"))
    else:
        try:
            provider = cast(BaseLLMProvider, ProviderFactory.create_provider("openai"))
            print("OpenAI provider created")
        except Exception as e:
            print(f"OpenAI provider failed, using mock: {e}")
            provider = cast(BaseLLMProvider, ProviderFactory.create_provider("mock"))
    
    config = AgentConfig(name="OpenAIAgent", spree_enabled=True)
    agent = ReactAgent(config=config, llm_provider=provider)
    
    response = await agent.execute("Explain quantum computing in simple terms")
    print(f"OpenAI Response: {response.content[:200]}...")


def demo_fastapi_integration() -> None:
    """Demonstrate FastAPI integration."""
    print("\n=== FastAPI Integration Demo ===")
    
    try:
        # Create FastAPI app
        app = create_app()
        print("FastAPI app created")
        
        # Get app routes with proper type handling
        routes: List[str] = []
        if hasattr(app, 'routes'):
            for route in app.routes:
                if hasattr(route, 'methods') and hasattr(route, 'path'):
                    # Explicitly cast to a concrete set[str] so Pyright knows the element type
                    empty_methods: Set[str] = set()
                    methods: Set[str] = cast(Set[str], getattr(route, 'methods', empty_methods))
                    if methods:
                        # Convert to list with an explicit element type annotation
                        method_list: list[str] = list(methods)  # type: ignore[var-annotated]
                        if method_list and hasattr(route, 'path'):
                            route_path = getattr(route, 'path', '')
                            if isinstance(route_path, str):
                                routes.append(f"{method_list[0]} {route_path}")
        
        print(f"Available endpoints ({len(routes)}):")
        for route in routes[:10]:  # Show first 10
            print(f"  - {route}")
        
        if len(routes) > 10:
            print(f"  ... and {len(routes) - 10} more")
            
    except Exception as e:
        print(f"FastAPI integration error: {e}")


def demo_configuration() -> None:
    """Demonstrate configuration management."""
    print("\n=== Configuration Demo ===")
    
    try:
        settings = get_settings()
        print(f"Settings loaded from environment: {settings.environment}")
        print(f"Database: {settings.database.url or 'SQLite default'}")
        print(f"LLM Provider: {settings.llm.provider}")
        print(f"Security enabled: {settings.security.rate_limit_enabled}")
    except Exception as e:
        print(f"Configuration error: {e}")


async def run_automated_tests() -> List[Tuple[str, bool]]:
    """Run basic automated tests."""
    print("\n=== Automated Testing Demo ===")
    
    test_results: List[Tuple[str, bool]] = []
    provider: Optional[BaseLLMProvider] = None
    
    # Test 1: Provider health check
    try:
        provider = cast(BaseLLMProvider, ProviderFactory.create_provider("mock"))
        if hasattr(provider, 'health_check'):
            health = await provider.health_check()
            test_results.append(("Provider Health Check", bool(health)))
            print(f"Provider health check: {'PASS' if health else 'FAIL'}")
        else:
            test_results.append(("Provider Health Check", False))
            print("Provider health check: FAIL (no health_check method)")
    except Exception as e:
        test_results.append(("Provider Health Check", False))
        print(f"Provider health check failed: {e}")
    
    # Test 2: Agent execution
    try:
        if provider is None:
            provider = cast(BaseLLMProvider, ProviderFactory.create_provider("mock"))
        config = AgentConfig(name="TestAgent")
        agent = ReactAgent(config=config, llm_provider=provider)
        response = await agent.execute("Test message")
        test_results.append(("Agent Execution", response.success))
        print(f"Agent execution: {'PASS' if response.success else 'FAIL'}")
    except Exception as e:
        test_results.append(("Agent Execution", False))
        print(f"Agent execution failed: {e}")
    
    # Test 3: Tool functionality
    try:
        calc_tool = CalculatorTool()
        result = await calc_tool.execute("2 + 2")
        test_results.append(("Calculator Tool", "4" in result))
        print(f"Calculator tool: {'PASS' if '4' in result else 'FAIL'}")
    except Exception as e:
        test_results.append(("Calculator Tool", False))
        print(f"Calculator tool failed: {e}")
    
    # Summary
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    print(f"\nTest Summary: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    return test_results


def save_demo_results(test_results: List[Tuple[str, bool]]) -> None:
    """Save demo results to file."""
    results: Dict[str, Any] = {
        "timestamp": time.time(),
        "demo_completed": True,
        "test_results": [
            {"test": name, "passed": result}
            for name, result in test_results
        ],
        "summary": {
            "total_tests": len(test_results),
            "passed_tests": sum(1 for _, result in test_results if result),
            "success_rate": sum(1 for _, result in test_results if result) / len(test_results) if test_results else 0
        }
    }
    
    output_file = Path("demo_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Demo results saved to {output_file}")


async def main() -> None:
    """Main demo function."""
    print_banner()
    
    print("Starting LlamaAgent Comprehensive Demo")
    print("=" * 60)
    
    start_time = time.time()
    
    # Run all demonstrations
    await demo_basic_agent()
    await demo_spre_planning()
    await demo_tools_integration()
    await demo_openai_integration()
    demo_fastapi_integration()
    demo_configuration()
    
    # Run automated tests
    test_results = await run_automated_tests()
    
    # Save results
    save_demo_results(test_results)
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Demo completed successfully in {elapsed:.2f} seconds!")
    print("\nNext steps:")
    print("  1. Start the API server: python -m uvicorn src.llamaagent.api:create_app --factory --reload")
    print("  2. Run full tests: python test_runner.py")
    print("  3. Build Docker: docker build -t llamaagent .")
    print("  4. Explore the documentation and examples!")


if __name__ == "__main__":
    asyncio.run(main()) 
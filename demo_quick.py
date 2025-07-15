#!/usr/bin/env python3
"""
Quick LlamaAgent Framework Demo
Author: Nik Jois <nikjois@llamasearch.ai>

This is a minimal demonstration of the LlamaAgent framework capabilities,
focusing on SPRE (Strategic Planning & Resourceful Execution) features.
"""

import asyncio
import time
from pathlib import Path

# Import the LlamaAgent framework
try:
    from llamaagent.agents import AgentConfig, AgentRole, ReactAgent
    from llamaagent.data_generation.spre import SPREDatasetGenerator
    from llamaagent.tools import ToolRegistry, get_all_tools
except ImportError as e:
    print(f"Error importing LlamaAgent: {e}")
    print("Please ensure the package is installed: pip install -e .")
    exit(1)


def print_header():
    """Print a nice header for the demo."""
    print("=" * 20)
    print("LlamaAgent Framework Demo")
    print("=" * 20)
    print()


def print_capabilities():
    """Print framework capabilities."""
    print("Framework Capabilities:")
    print("- SPRE planning capabilities")
    print("- Tool usage and integration")
    print("- Dataset generation")
    print("- FastAPI web service")
    print()


async def demo_basic_agents():
    """Demonstrate basic agent functionality."""
    print("DEMO 1: Basic Agent Functionality")
    print("-" * 40)

    # Create a basic agent configuration
    config = AgentConfig(
        name="DemoAgent",
        role=AgentRole.GENERALIST,
        temperature=0.7,
        spree_enabled=False,  # Start with basic functionality
    )

    # Set up tools
    tools = ToolRegistry()
    for tool in get_all_tools():
        tools.register(tool)

    # Create agent
    agent = ReactAgent(config, tools=tools)

    # Test tasks
    test_tasks = [
        "What is 15 * 23?",
        "Calculate the square root of 144",
        "What is the capital of France?",
        "Explain what machine learning is in one sentence",
    ]

    for i, task in enumerate(test_tasks, 1):
        print(f"\nTask {i}: {task}")

        start_time = time.time()
        result = await agent.execute(task)
        execution_time = time.time() - start_time

        print(f"Response: {result.content}")
        print(f"Success: {result.success}")
        print(f"Time: {execution_time:.3f}s")

        # Count tool calls
        tool_calls = [trace for trace in result.trace if trace.get("type") == "tool_execution_success"]
        print(f"Tools used: {len(tool_calls)}")
        print("-" * 30)


async def demo_spre_planning():
    """Demonstrate SPRE planning capabilities."""
    print("DEMO 2: SPRE Planning (Strategic Planning & Resourceful Execution)")
    print("-" * 70)

    # Create SPRE-enabled agent
    config = AgentConfig(
        name="SPREAgent",
        role=AgentRole.PLANNER,
        temperature=0.5,
        spree_enabled=True,  # Enable SPRE features
    )

    # Set up tools
    tools = ToolRegistry()
    for tool in get_all_tools():
        tools.register(tool)

    # Create agent
    agent = ReactAgent(config, tools=tools)

    # Complex task that benefits from planning
    complex_task = (
        "Calculate the compound interest on $1000 invested at 5% annual rate for 3 years, "
        "then create a Python function that can calculate compound interest for any principal, "
        "rate, and time period."
    )

    print(f"\nComplex Task: {complex_task}")
    print()

    start_time = time.time()
    result = await agent.execute(complex_task)
    execution_time = time.time() - start_time

    print(f"\nSPRE Response: {result.content}")
    print(f"Success: {result.success}")
    print(f"Time: {execution_time:.3f}s")
    print(f"SPRE Planning: {'Enabled' if config.spree_enabled else 'Disabled'}")

    # Show planning information
    planning_events = [trace for trace in result.trace if trace.get("type") == "plan_generated"]
    if planning_events:
        plan_data = planning_events[0].get("data", {})
        steps = plan_data.get("plan", {}).get("steps", [])
        print(f"Planning Steps: {len(steps)}")
        for i, step in enumerate(steps[:3], 1):  # Show first 3 steps
            print(f"  {i}. {step.get('description', 'N/A')}")


async def demo_dataset_generation():
    """Demonstrate SPRE dataset generation."""
    print("DEMO 3: SPRE Dataset Generation")
    print("-" * 40)

    generator = SPREDatasetGenerator(seed=42)
    output_file = Path("demo_output/demo_dataset.json")
    output_file.parent.mkdir(exist_ok=True)

    # Generate small dataset for demo
    try:
        await generator.generate_dataset(num_episodes=5, output_path=output_file)
        print(f"SPRE dataset generated successfully at [green]{output_file}[/green]")
    except Exception as e:
        print(f"Error generating dataset: {e}")


async def main():
    """Run all demonstrations."""
    print_header()
    print_capabilities()

    try:
        await demo_basic_agents()
        print("\n" + "=" * 60 + "\n")

        await demo_spre_planning()
        print("\n" + "=" * 60 + "\n")

        await demo_dataset_generation()

        print("\n" + "=" * 60)
        print("All demos completed successfully!")
        print("\nNext steps:")
        print("1. Explore the full API: python -m llamaagent.api")
        print("2. Run comprehensive tests: python -m pytest tests/ -v")
        print("3. Generate larger datasets: python -m llamaagent.data_generation.spre --help")
        print("4. Try the interactive CLI: python -m llamaagent.cli.interactive")

    except Exception as e:
        print(f"Demo error: {e}")
        print("Some features may require additional dependencies.")


if __name__ == "__main__":
    asyncio.run(main())

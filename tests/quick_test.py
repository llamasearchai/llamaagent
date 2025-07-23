#!/usr/bin/env python3
"""Quick test of LlamaAgent framework with multiple datasets."""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.append("src")

from llamaagent.agents import AgentConfig, AgentRole, ReactAgent
from llamaagent.benchmarks.gaia_benchmark import GAIABenchmark
from llamaagent.tools import ToolRegistry, get_all_tools


async def test_basic_functionality():
    """Test basic agent functionality."""
    print("Setting up tools...")
    tools = ToolRegistry()
    for tool in get_all_tools():
        tools.register(tool)

    print("Creating SPRE agent...")
    config = AgentConfig(name="SPRE-Test", role=AgentRole.PLANNER, spree_enabled=True)
    agent = ReactAgent(config, tools=tools)

    print("Testing basic calculation...")
    response = await agent.execute("Calculate 15 * 23 + 47")
    print(f"Basic test result: {response.content[:100]}...")
    print(f"Success: {response.success}")

    return agent, tools


async def test_gaia_benchmark(agent, max_tasks=5):
    """Test GAIA benchmark evaluation."""
    print(f"\nTesting GAIA benchmark with {max_tasks} tasks...")

    benchmark = GAIABenchmark(max_tasks=max_tasks)
    await benchmark.load_dataset()
    print(f"Loaded {len(benchmark.tasks)} tasks")

    start_time = time.time()
    results = await benchmark.evaluate_agent(agent, shuffle=False)
    eval_time = time.time() - start_time

    report = benchmark.generate_report(results)
    print(f"GAIA Results: {report['correct_answers']}/{report['total_tasks']} correct")
    print(f"Accuracy: {report['overall_accuracy']:.1%}")
    print(f"Evaluation time: {eval_time:.1f}s")

    return report


async def test_math_problems(agent):
    """Test mathematical reasoning tasks."""
    print("\nTesting mathematical reasoning...")

    math_tasks = [
        "Calculate the compound interest on $1000 at 5% annual rate for 3 years",
        "If a train travels 60 mph for 2 hours then 80 mph for 1.5 hours, what is the average speed?",
        "Find the area of a triangle with vertices at (0,0), (4,0), and (2,3)",
    ]

    results = []
    for i, task in enumerate(math_tasks):
        print(f"Math task {i + 1}: {task[:50]}...")
        response = await agent.execute(task)
        results.append(
            {
                "task": task,
                "response": (
                    response.content[:200] + "..."
                    if len(response.content) > 200
                    else response.content
                ),
                "success": response.success,
                "time": response.execution_time,
            }
        )

    return results


async def test_code_generation(agent):
    """Test code generation capabilities."""
    print("\nTesting code generation...")

    code_tasks = [
        "Write a Python function to calculate factorial of a number",
        "Create a function that finds the longest palindromic substring",
        "Write a function to reverse a linked list",
    ]

    results = []
    for i, task in enumerate(code_tasks):
        print(f"Code task {i + 1}: {task[:50]}...")
        response = await agent.execute(task)
        has_function = "def " in response.content
        results.append(
            {
                "task": task,
                "has_function": has_function,
                "success": response.success,
                "time": response.execution_time,
            }
        )

    return results


async def compare_configurations():
    """Compare Vanilla vs SPRE configurations."""
    print("\nComparing agent configurations...")

    tools = ToolRegistry()
    for tool in get_all_tools():
        tools.register(tool)

    configs = [
        (
            "Vanilla",
            AgentConfig(name="Vanilla", role=AgentRole.GENERALIST, spree_enabled=False),
        ),
        ("SPRE", AgentConfig(name="SPRE", role=AgentRole.PLANNER, spree_enabled=True)),
    ]

    test_tasks = [
        "Calculate 123 * 456 and explain the process",
        "Write Python code to sort a list of numbers",
        "Explain the concept of recursion with an example",
    ]

    comparison_results = {}

    for config_name, config in configs:
        print(f"\nTesting {config_name} configuration...")
        agent = ReactAgent(config, tools=tools)

        config_results = []
        total_time = 0

        for task in test_tasks:
            start_time = time.time()
            response = await agent.execute(task)
            exec_time = time.time() - start_time
            total_time += exec_time

            config_results.append(
                {
                    "task": task[:50] + "...",
                    "success": response.success,
                    "time": exec_time,
                    "tokens": response.tokens_used,
                }
            )

        comparison_results[config_name] = {
            "results": config_results,
            "avg_time": total_time / len(test_tasks),
            "total_tokens": sum(r["tokens"] for r in config_results),
        }

    return comparison_results


async def main():
    """Run comprehensive test suite."""
    print("=" * 80)
    print("LLAMAAGENT COMPREHENSIVE TEST SUITE")
    print("=" * 80)

    start_time = time.time()

    # Test basic functionality
    agent, tools = await test_basic_functionality()

    # Test GAIA benchmark
    gaia_report = await test_gaia_benchmark(agent, max_tasks=5)

    # Test mathematical reasoning
    math_results = await test_math_problems(agent)

    # Test code generation
    code_results = await test_code_generation(agent)

    # Compare configurations
    comparison = await compare_configurations()

    total_time = time.time() - start_time

    # Compile final report
    final_report = {
        "test_timestamp": time.time(),
        "total_test_time": total_time,
        "gaia_benchmark": gaia_report,
        "math_reasoning": {
            "total_tasks": len(math_results),
            "successful_tasks": sum(1 for r in math_results if r["success"]),
            "avg_time": sum(r["time"] for r in math_results) / len(math_results),
        },
        "code_generation": {
            "total_tasks": len(code_results),
            "tasks_with_functions": sum(1 for r in code_results if r["has_function"]),
            "avg_time": sum(r["time"] for r in code_results) / len(code_results),
        },
        "configuration_comparison": comparison,
        "datasets_tested": [
            {
                "name": "GAIA",
                "url": "https://huggingface.co/datasets/gaia-benchmark/GAIA",
            },
            {"name": "Custom Math", "description": "Mathematical reasoning tasks"},
            {"name": "Custom Code", "description": "Code generation tasks"},
        ],
    }

    # Save results
    output_path = Path("test_results.json")
    with open(output_path, "w") as f:
        json.dump(final_report, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total test time: {total_time:.1f}s")
    print(
        f"\nGAIA Benchmark: {gaia_report['correct_answers']}/{gaia_report['total_tasks']} correct ({gaia_report['overall_accuracy']:.1%})"
    )
    print(
        f"Math Reasoning: {final_report['math_reasoning']['successful_tasks']}/{final_report['math_reasoning']['total_tasks']} successful"
    )
    print(
        f"Code Generation: {final_report['code_generation']['tasks_with_functions']}/{final_report['code_generation']['total_tasks']} with functions"
    )

    print("\nConfiguration Comparison:")
    for config_name, config_data in comparison.items():
        print(
            f"  {config_name}: {config_data['avg_time']:.2f}s avg, {config_data['total_tokens']} tokens"
        )

    print("\nDatasets tested:")
    for dataset in final_report["datasets_tested"]:
        if "url" in dataset:
            print(f"  - {dataset['name']}: {dataset['url']}")
        else:
            print(f"  - {dataset['name']}: {dataset['description']}")

    print(f"\nResults saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

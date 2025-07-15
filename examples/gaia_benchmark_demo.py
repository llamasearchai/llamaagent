#!/usr/bin/env python3
"""Demonstration of GAIA benchmark usage."""

import asyncio
from pathlib import Path

from llamaagent.agents import AgentConfig
from llamaagent.benchmarks.gaia_benchmark import GAIABenchmark, run_gaia_evaluation
from llamaagent.tools import ToolRegistry


async def demonstrate_gaia_benchmark():
    """Demonstrate GAIA benchmark functionality."""
    print("=== GAIA Benchmark Demonstration ===\n")
    
    # Create a benchmark instance
    benchmark = GAIABenchmark(max_tasks=10)
    
    # Show available tasks
    print(f"Total tasks available: {len(benchmark.tasks)}")
    
    # Get statistics
    stats = benchmark.get_stats()
    print(f"\nBenchmark Statistics:")
    print(f"- Total tasks: {stats['total_tasks']}")
    print(f"- Difficulties: {stats['difficulties']}")
    print(f"- Domains: {stats['domains']}")
    print(f"- Average steps: {stats['avg_steps']:.1f}")
    
    # Show tasks by difficulty
    print("\nTasks by difficulty:")
    for difficulty in ["easy", "medium", "hard"]:
        tasks = benchmark.get_tasks(difficulty=difficulty)
        print(f"- {difficulty}: {len(tasks)} tasks")
        if tasks:
            print(f"  Example: {tasks[0].question[:60]}...")
    
    # Show tasks by domain
    print("\nTasks by domain:")
    for domain in ["mathematics", "logic", "programming", "science"]:
        tasks = benchmark.get_tasks(domain=domain)
        print(f"- {domain}: {len(tasks)} tasks")
    
    # Demonstrate task filtering
    print("\nFiltered tasks (medium difficulty, min 3 steps):")
    filtered = benchmark.get_tasks(difficulty="medium", min_steps=3, limit=3)
    for task in filtered:
        print(f"- {task.task_id}: {task.question[:50]}...")
    
    # Show how to run evaluation (without actually running it)
    print("\nTo run a full evaluation with an agent:")
    print("""
    # Configure agent
    agent_config = AgentConfig(
        name="MyAgent",
        model="gpt-4",
        max_iterations=10
    )
    
    # Create tool registry
    tools = ToolRegistry()
    tools.register_default_tools()
    
    # Run evaluation
    results = await run_gaia_evaluation(
        agent_config=agent_config,
        tools=tools,
        subset="validation",
        max_tasks=20,
        output_dir=Path("results")
    )
    
    # View results
    print(f"Overall accuracy: {results['overall_accuracy']:.2%}")
    """)


async def demonstrate_custom_tasks():
    """Demonstrate creating custom GAIA tasks."""
    print("\n=== Custom GAIA Tasks ===\n")
    
    # Create a custom benchmark with specific tasks
    from llamaagent.benchmarks.gaia_benchmark import GAIATask
    
    custom_tasks = [
        GAIATask(
            task_id="custom_001",
            question="What is the capital of France?",
            expected_answer="Paris",
            difficulty="easy",
            steps_required=1,
            domain="geography"
        ),
        GAIATask(
            task_id="custom_002",
            question="Calculate the area of a circle with radius 5 units (use π = 3.14)",
            expected_answer="78.5",
            difficulty="medium",
            steps_required=2,
            domain="mathematics"
        ),
        GAIATask(
            task_id="custom_003",
            question="Write a SQL query to find all users older than 25 from a 'users' table",
            expected_answer="SELECT * FROM users WHERE age > 25",
            difficulty="medium",
            steps_required=1,
            domain="programming"
        ),
    ]
    
    # Create benchmark with custom tasks
    benchmark = GAIABenchmark()
    benchmark.tasks.extend(custom_tasks)
    
    print("Added custom tasks:")
    for task in custom_tasks:
        print(f"- {task.task_id}: {task.question}")
    
    # Save tasks to file
    benchmark._save_tasks()
    print(f"\nTasks saved to: {benchmark.data_file}")


async def main():
    """Run all demonstrations."""
    await demonstrate_gaia_benchmark()
    await demonstrate_custom_tasks()
    
    print("\nPASS GAIA Benchmark demonstration complete!")


if __name__ == "__main__":
    asyncio.run(main())
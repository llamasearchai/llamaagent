#!/usr/bin/env python3
"""
Working Demo for LlamaAgent
Demonstrates basic functionality that is currently working
"""

import asyncio
import sys

sys.path.insert(0, 'src')

from llamaagent import AgentConfig, ReactAgent
from llamaagent.llm.providers.mock_provider import MockProvider
from llamaagent.tools import CalculatorTool, PythonREPLTool, ToolRegistry


async def main():
    print("=" * 80)
    print("LLAMAAGENT WORKING DEMO")
    print("=" * 80)
    print()

    # 1. Initialize components
    print("1. Initializing Components...")

    # Create LLM provider
    llm_provider = MockProvider()
    print("   ✓ LLM Provider: MockProvider")

    # Create agent configuration
    config = AgentConfig(
        name="DemoAgent", max_iterations=5, enable_tools=True, enable_memory=True
    )
    print(f"   ✓ Agent Config: {config.name}")

    # 2. Setup tools
    print("\n2. Setting up Tools...")
    tool_registry = ToolRegistry()

    calc_tool = CalculatorTool()
    tool_registry.register(calc_tool)
    print("   ✓ Calculator tool registered")

    python_tool = PythonREPLTool()
    tool_registry.register(python_tool)
    print("   ✓ Python REPL tool registered")

    # 3. Create agent
    print("\n3. Creating React Agent...")
    agent = ReactAgent(config, llm_provider=llm_provider, tools=tool_registry)
    print(f"   ✓ Agent created: {agent.config.name}")
    print(f"   ✓ Stats tracking enabled")

    # 4. Execute tasks
    print("\n4. Executing Tasks...")

    tasks = [
        "Calculate 25 * 4 + 10",
        "What is the square root of 144?",
        "Generate a list of prime numbers up to 20",
    ]

    for i, task in enumerate(tasks, 1):
        print(f"\n   Task {i}: {task}")
        response = await agent.execute(task)

        print(f"   Success: {response.success}")
        print(f"   Response: {response.content[:100]}...")
        print(f"   Execution Time: {response.execution_time:.3f}s")
        print(f"   Tokens Used: {response.tokens_used}")

    # 5. Show statistics
    print("\n5. Agent Statistics:")
    stats = agent.stats
    print(f"   Total Executions: {stats.total_executions}")
    print(f"   Successful: {stats.successful_executions}")
    print(f"   Failed: {stats.failed_executions}")
    print(f"   Success Rate: {stats.get_success_rate():.1%}")
    print(f"   Average Execution Time: {stats.average_execution_time:.3f}s")
    print(f"   Total Tokens Used: {stats.total_tokens_used}")

    # 6. Test tool functionality
    print("\n6. Testing Tools Directly...")

    # Test calculator
    calc_result = calc_tool.execute("2 ** 8")
    print(f"   Calculator: 2 ** 8 = {calc_result}")

    # Test Python REPL
    python_result = python_tool.execute("sum(range(1, 11))")
    print(f"   Python REPL: sum(range(1, 11)) = {python_result}")

    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

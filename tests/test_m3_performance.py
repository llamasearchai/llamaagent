import asyncio
import time

from llamaagent.agents import AgentConfig, ReactAgent
from llamaagent.tools import ToolRegistry, get_all_tools


async def performance_test():
    """Test LlamaAgent performance on M3 Max."""
    config = AgentConfig(
        name="M3MaxAgent", spree_enabled=True, max_iterations=15, temperature=0.7
    )

    registry = ToolRegistry()
    for tool in get_all_tools():
        registry.register(tool)

    agent = ReactAgent(config, tools=registry)

    # Complex multi-step task
    task = """
    Perform these calculations and create Python functions:
    1. Calculate compound interest on $50000 at 8.5% for 15 years
    2. Determine monthly payment for a $500000 mortgage at 6.5% for 30 years
    3. Create a portfolio optimization function for 5 stocks
    4. Calculate the present value of an annuity paying $5000/year for 20 years at 7%
    5. Generate a Monte Carlo simulation for retirement planning
    """

    start_time = time.time()
    response = await agent.execute(task)
    end_time = time.time()

    print("M3 Max Performance Test Results:")
    print(f"Success: {response.success}")
    print(f"Execution Time: {end_time - start_time:.2f} seconds")
    print(f"Response Length: {len(response.content)} characters")
    print(f"Tools Used: {len(response.trace)}")

    return response


if __name__ == "__main__":
    asyncio.run(performance_test())

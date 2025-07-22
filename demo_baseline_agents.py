#!/usr/bin/env python3
"""Demo script showing baseline agents functionality."""

import asyncio
from src.llamaagent.benchmarks.baseline_agents import (
    BaselineType,
    BaselineAgentFactory,
    VanillaReactAgent,
    PreActOnlyAgent,
    SEMOnlyAgent
)
from src.llamaagent.llm import create_provider

async def demo_baseline_agents():
    """Demonstrate different baseline agent types."""
    
    print("=== LlamaAgent Baseline Agents Demo ===\n")
    
    # Create a mock LLM provider for demonstration
    llm_provider = create_provider("mock")
    
    # Show all available baseline types
    print("Available baseline agent types:")
    for baseline_type in BaselineAgentFactory.get_all_baseline_types():
        description = BaselineAgentFactory.get_baseline_description(baseline_type)
        print(f"  - {baseline_type}: {description}")
    
    print("\nCreating baseline agents...")
    
    # Create each type of baseline agent
    agents = {}
    for baseline_type in [BaselineType.VANILLA_REACT, BaselineType.PREACT_ONLY, 
                          BaselineType.SEM_ONLY, BaselineType.SPRE_FULL]:
        try:
            agent = BaselineAgentFactory.create_agent(
                baseline_type=baseline_type,
                llm_provider=llm_provider,
                name_suffix="-Demo"
            )
            agents[baseline_type] = agent
            print(f"   Created {baseline_type} agent: {agent.config.name}")
        except Exception as e:
            print(f"   Failed to create {baseline_type} agent: {e}")
    
    # Show agent configurations
    print("\nAgent configurations:")
    for baseline_type, agent in agents.items():
        print(f"\n{baseline_type}:")
        print(f"  - Name: {agent.config.name}")
        print(f"  - Role: {agent.config.role}")
        print(f"  - SPRE enabled: {agent.config.spree_enabled}")
        print(f"  - Max iterations: {agent.config.max_iterations}")
        print(f"  - Agent class: {agent.__class__.__name__}")
    
    # Example task execution (with mock provider)
    print("\n\nExample task execution:")
    task = "Calculate the sum of 15 and 27"
    
    for baseline_type, agent in agents.items():
        print(f"\n{baseline_type} executing task: '{task}'")
        try:
            # Note: With mock provider, this will return a simulated response
            result = await agent.execute(task)
            print(f"  Result: {result}")
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    asyncio.run(demo_baseline_agents())
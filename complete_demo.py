#!/usr/bin/env python3
"""Complete demo of LlamaAgent framework functionality."""

import asyncio
import logging
from typing import Any, Dict

from src.llamaagent.agents import ReactAgent
from src.llamaagent.agents.base import AgentConfig
from src.llamaagent.llm import MockProvider
from src.llamaagent.memory import SimpleMemory
from src.llamaagent.tools import ToolRegistry
from src.llamaagent.tools.calculator import CalculatorTool
from src.llamaagent.tools.python_repl import PythonREPLTool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def demo_basic_agent():
    """Demonstrate basic agent functionality."""
    print("\n=== Basic Agent Demo ===")
    
    # Create agent configuration
    config = AgentConfig(
        name="DemoAgent",
        temperature=0.7
    )
    
    # Create and configure agent
    agent = ReactAgent(config=config)
    
    # Execute a simple task
    result = await agent.execute("Hello, I'm a demo agent!")
    print(f"Agent response: {result}")


async def demo_agent_with_tools():
    """Demonstrate agent with tools."""
    print("\n=== Agent with Tools Demo ===")
    
    # Create configuration
    config = AgentConfig(name="ToolAgent")
    
    # Create tool registry and add tools
    tool_registry = ToolRegistry()
    tool_registry.register(CalculatorTool())
    tool_registry.register(PythonREPLTool())
    
    # Create agent with tools
    agent = ReactAgent(config=config, tools=tool_registry)
    
    # Execute calculations
    tasks = [
        "Calculate 15 * 7",
        "What is the square root of 144?",
        "Evaluate: (10 + 5) * 3"
    ]
    
    for task in tasks:
        result = await agent.execute(task)
        print(f"Task: {task}")
        print(f"Result: {result}\n")


async def demo_agent_with_memory():
    """Demonstrate agent with memory."""
    print("\n=== Agent with Memory Demo ===")
    
    # Create configuration
    config = AgentConfig(name="MemoryAgent")
    
    # Create memory and agent
    memory = SimpleMemory()
    agent = ReactAgent(config=config, memory=memory)
    
    # Add some memories
    await memory.add("User's name is Alice", tags=["user", "identity"])
    await memory.add("User likes Python programming", tags=["user", "preferences"])
    await memory.add("Previous calculation result was 42", tags=["calculation"])
    
    # Execute task that uses memory
    result = await agent.execute("What do you remember about the user?")
    print(f"Agent response: {result}")
    
    # Search memories
    search_results = await memory.search("user", limit=5)
    print(f"\nMemory search results for 'user': {len(search_results)} entries found")
    for entry in search_results:
        print(f"  - {entry['content']}")


async def demo_custom_provider():
    """Demonstrate custom LLM provider."""
    print("\n=== Custom Provider Demo ===")
    
    # Create a custom mock provider with specific responses
    class CustomMockProvider(MockProvider):
        def __init__(self):
            super().__init__()
            self.custom_responses = {
                "weather": "It's sunny and 72°F today!",
                "greeting": "Hello! How can I assist you today?",
                "math": "I can help with calculations using my tools."
            }
        
        async def complete(self, messages: list, **kwargs) -> str:
            # Check for keywords in the last message
            last_message = messages[-1]["content"].lower()
            
            for keyword, response in self.custom_responses.items():
                if keyword in last_message:
                    return response
            
            return super().complete(messages, **kwargs)
    
    # Create agent with custom provider
    config = AgentConfig(name="CustomAgent")
    provider = CustomMockProvider()
    agent = ReactAgent(config=config, llm_provider=provider)
    
    # Test custom responses
    queries = [
        "What's the weather like?",
        "Hello, agent!",
        "Can you help with math?"
    ]
    
    for query in queries:
        result = await agent.execute(query)
        print(f"Query: {query}")
        print(f"Response: {result}\n")


async def demo_advanced_features():
    """Demonstrate advanced features."""
    print("\n=== Advanced Features Demo ===")
    
    # Create configuration with advanced settings
    config = AgentConfig(
        name="AdvancedAgent",
        temperature=0.5,
        max_tokens=150
    )
    
    # Create tools and memory
    tool_registry = ToolRegistry()
    tool_registry.register(CalculatorTool())
    memory = SimpleMemory()
    
    # Create agent
    agent = ReactAgent(
        config=config,
        tools=tool_registry,
        memory=memory
    )
    
    # Store context in memory
    await memory.add("Project budget is $10,000", tags=["project", "finance"])
    await memory.add("Team size is 5 developers", tags=["project", "team"])
    
    # Complex task
    result = await agent.execute(
        "If each developer works 40 hours per week at $50/hour, "
        "how many weeks can the project run with the current budget?"
    )
    print(f"Complex calculation result: {result}")
    
    # Show agent trace
    if hasattr(agent, 'get_trace'):
        trace = agent.get_trace()
        print(f"\nAgent trace steps: {len(trace)}")


async def main():
    """Run all demos."""
    print("=== LlamaAgent Framework Complete Demo ===")
    
    try:
        await demo_basic_agent()
        await demo_agent_with_tools()
        await demo_agent_with_memory()
        await demo_custom_provider()
        await demo_advanced_features()
        
        print("\n=== Demo completed successfully! ===")
        
    except Exception as e:
        logger.error(f"Error during demo: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
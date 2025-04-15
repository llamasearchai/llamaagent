#!/usr/bin/env python3
"""
Simple example of using LlamaAgent
"""
import logging
import os

from llamaagent import Agent, Memory
from llamaagent.tools import CalculatorTool, WebSearchTool

# Set up logging
logging.basicConfig(level=logging.INFO)

# Create memory
memory = Memory(short_term_capacity=10, long_term_enabled=False)

# Create tools
calculator = CalculatorTool()
web_search = WebSearchTool(
    api_key="REDACTED",  # In a real application, use a real API key
    requires_api_key=False,  # For demo purposes only
)

# Create an agent
agent = Agent(
    name="math_assistant",
    description="A helpful assistant that can solve math problems and search for information",
    tools=[calculator, web_search],
    memory=memory,
)

# Example task
result = agent.run("Calculate 23 * 45 and tell me who invented calculus")

# Print the result
print("-" * 50)
print("Task: Calculate 23 * 45 and tell me who invented calculus")
print("-" * 50)
print(f"Output: {result.output}")
print("-" * 50)
print("Thinking:")
print(result.thinking)
print("-" * 50)
print(f"Tools Used: {', '.join(result.tools_used)}")
print(f"Execution Time: {result.execution_time:.2f} seconds")

# Another example
result = agent.run("What is the square root of 144?")

# Print the result
print("\n" + "-" * 50)
print("Task: What is the square root of 144?")
print("-" * 50)
print(f"Output: {result.output}")
print("-" * 50)
print("Thinking:")
print(result.thinking)
print("-" * 50)
print(f"Tools Used: {', '.join(result.tools_used)}")
print(f"Execution Time: {result.execution_time:.2f} seconds")

# Check memory
print("\n" + "-" * 50)
print("Memory Content:")
print(memory.get_context())
print("-" * 50)

# Run a task that references previous context
result = agent.run("Who invented calculus? And what was the first calculation we did?")

# Print the result
print("\n" + "-" * 50)
print("Task: Who invented calculus? And what was the first calculation we did?")
print("-" * 50)
print(f"Output: {result.output}")
print("-" * 50)
print(f"Tools Used: {', '.join(result.tools_used)}")
print(f"Execution Time: {result.execution_time:.2f} seconds")

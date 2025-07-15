"""
Basic Agent Example

This example demonstrates how to create a simple agent with basic tools.
"""

import asyncio

from llamaagent import Agent
from llamaagent.llm import LiteLLMProvider
from llamaagent.tools import Tool


# Define custom tools
@Tool.create
def calculate(expression: str) -> float:
    """
    Safely evaluate a mathematical expression.
    
    Args:
        expression: A mathematical expression like "2 + 2" or "10 * 5"
    
    Returns:
        The result of the calculation
    """
    # Safe evaluation of mathematical expressions
    allowed_chars = "0123456789+-*/()., "
    if all(c in allowed_chars for c in expression):
        try:
            return eval(expression)
        except:
            return "Invalid expression"
    return "Invalid characters in expression"


@Tool.create
def get_time() -> str:
    """Get the current time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@Tool.create
def reverse_string(text: str) -> str:
    """Reverse a given string."""
    return text[::-1]


async def main():
    """Run the basic agent example."""
    
    # Initialize LLM provider
    # You can use any supported model
    llm = LiteLLMProvider(
        model="gpt-3.5-turbo",  # or "claude-3-haiku", "llama3", etc.
        temperature=0.7
    )
    
    # Create the agent
    agent = Agent(
        name="BasicAssistant",
        llm=llm,
        tools=[calculate, get_time, reverse_string],
        system_prompt="""You are a helpful assistant with access to several tools:
        - calculate: For mathematical calculations
        - get_time: To get the current time
        - reverse_string: To reverse text
        
        Use these tools when appropriate to help the user."""
    )
    
    # Example interactions
    print("Basic Agent Example")
    print("=" * 50)
    
    # Example 1: Simple calculation
    print("\n1. Mathematical Calculation:")
    response = await agent.run("What is 25 * 4 + 10?")
    print("User: What is 25 * 4 + 10?")
    print(f"Agent: {response}")
    
    # Example 2: Get current time
    print("\n2. Time Query:")
    response = await agent.run("What time is it right now?")
    print("User: What time is it right now?")
    print(f"Agent: {response}")
    
    # Example 3: String manipulation
    print("\n3. String Manipulation:")
    response = await agent.run("Can you reverse the word 'llamaagent' for me?")
    print("User: Can you reverse the word 'llamaagent' for me?")
    print(f"Agent: {response}")
    
    # Example 4: Complex query using multiple tools
    print("\n4. Complex Query:")
    response = await agent.run(
        "What's 15% of 200? Also, what time is it? "
        "And can you reverse the result of the calculation as a string?"
    )
    print("User: What's 15% of 200? Also, what time is it? And can you reverse the result as a string?")
    print(f"Agent: {response}")
    
    # Example 5: Conversation without tools
    print("\n5. General Conversation:")
    response = await agent.run("Tell me a fun fact about llamas.")
    print("User: Tell me a fun fact about llamas.")
    print(f"Agent: {response}")
    
    # Show agent statistics
    print("\n" + "=" * 50)
    print("Agent Statistics:")
    stats = agent.get_stats()
    print(f"- Total interactions: {stats.get('total_interactions', 0)}")
    print(f"- Tool calls made: {stats.get('tool_calls', 0)}")
    print(f"- Average response time: {stats.get('avg_response_time', 0):.2f}s")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
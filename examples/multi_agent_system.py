#!/usr/bin/env python3
"""
Example of a multi-agent system using LlamaAgent
"""
import logging
import os

from llamaagent import Agent, AgentSystem, Memory
from llamaagent.tools import CalculatorTool, CodeInterpreterTool, WebSearchTool

# Set up logging
logging.basicConfig(level=logging.INFO)

# Create tools
calculator = CalculatorTool()
web_search = WebSearchTool(
    api_key="REDACTED",  # In a real application, use a real API key
    requires_api_key=False,  # For demo purposes only
)
code_interpreter = CodeInterpreterTool()

# Create the researcher agent
researcher = Agent(
    name="researcher",
    description="Specialized in finding information",
    tools=[web_search],
    memory=Memory(short_term_capacity=5),
)

# Create the analyst agent
analyst = Agent(
    name="analyst",
    description="Specialized in analyzing data and performing calculations",
    tools=[calculator, code_interpreter],
    memory=Memory(short_term_capacity=5),
)

# Create the writer agent
writer = Agent(
    name="writer",
    description="Specialized in summarizing information and writing clear text",
    tools=[],
    memory=Memory(short_term_capacity=5),
)

# Create sequential agent system
sequential_system = AgentSystem(
    name="research_team",
    description="A team that researches topics and provides analysis",
    agents=[researcher, analyst, writer],
    coordinator="sequential",
)

# Run the sequential system
print("=" * 70)
print("Running Sequential Agent System (Researcher -> Analyst -> Writer)")
print("=" * 70)
result = sequential_system.run(
    "Provide information about renewable energy sources and calculate the average "
    "efficiency of solar panels versus wind turbines."
)

# Print the result
print("\nFinal Output:")
print("-" * 50)
print(result.output)
print("-" * 50)
print(f"Execution Time: {result.execution_time:.2f} seconds")

# Create round-robin agent system
round_robin_system = AgentSystem(
    name="collaborative_team",
    description="A team that collaboratively works on complex problems",
    agents=[researcher, analyst, writer],
    coordinator="round_robin",
)

# Run the round-robin system
print("\n" + "=" * 70)
print("Running Round-Robin Agent System (agents take turns refining solution)")
print("=" * 70)
result = round_robin_system.run(
    "Compare the economic impact of artificial intelligence in healthcare versus "
    "transportation industries over the next decade."
)

# Print the result
print("\nFinal Output:")
print("-" * 50)
print(result.output)
print("-" * 50)
print(f"Execution Time: {result.execution_time:.2f} seconds")

# Create hierarchical agent system
hierarchical_system = AgentSystem(
    name="managed_team",
    description="A team with a coordinator that delegates tasks",
    agents=[writer, researcher, analyst],  # First agent is the coordinator
    coordinator="hierarchical",
)

# Run the hierarchical system
print("\n" + "=" * 70)
print("Running Hierarchical Agent System (Writer coordinates Researcher and Analyst)")
print("=" * 70)
result = hierarchical_system.run(
    "Write an analysis of climate change effects on agriculture, including data "
    "on crop yields in affected regions and economic projections."
)

# Print the result
print("\nFinal Output:")
print("-" * 50)
print(result.output)
print("-" * 50)
print(f"Execution Time: {result.execution_time:.2f} seconds")

print("\n" + "=" * 70)
print("Compare the three approaches:")
print(
    f"Sequential:   {sequential_system.name} - {sequential_system.agents[0].name} -> {sequential_system.agents[1].name} -> {sequential_system.agents[2].name}"
)
print(
    f"Round-Robin:  {round_robin_system.name} - agents take turns refining the solution"
)
print(
    f"Hierarchical: {hierarchical_system.name} - {hierarchical_system.agents[0].name} delegates to and synthesizes from others"
)
print("=" * 70)

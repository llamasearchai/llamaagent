"""
Advanced Multi-Agent System Example

This example demonstrates a complex multi-agent system where different
specialized agents collaborate to complete tasks.
"""

import asyncio
from typing import Any, Dict

from llamaagent import Agent
from llamaagent.llm import LiteLLMProvider
from llamaagent.memory import VectorMemory
from llamaagent.orchestrator import Orchestrator, WorkflowType
from llamaagent.prompting import ChainOfThoughtStrategy, TreeOfThoughtsStrategy
from llamaagent.tools import CodeExecutionTool, Tool, WebSearchTool


# Custom tools for specialized agents
@Tool.create
async def analyze_code(code: str, language: str = "python") -> Dict[str, Any]:
    """Analyze code for quality, security, and performance issues."""
    # Simplified analysis - in production, use actual code analysis tools
    issues = []
    
    # Basic checks
    if "eval(" in code:
        issues.append({"type": "security", "message": "Use of eval() is dangerous"})
    if "import *" in code:
        issues.append({"type": "style", "message": "Avoid wildcard imports"})
    if not any(line.strip().startswith("#") for line in code.split("\n")):
        issues.append({"type": "documentation", "message": "No comments found"})
    
    return {
        "language": language,
        "lines": len(code.split("\n")),
        "issues": issues,
        "score": max(0, 100 - len(issues) * 20)
    }


@Tool.create
async def generate_tests(code: str, framework: str = "pytest") -> str:
    """Generate unit tests for the given code."""
    # Simplified test generation
    return """
import pytest
from your_module import your_function

def test_your_function():
    # Test case 1
    result = your_function(input1)
    assert result == expected1
    
    # Test case 2
    result = your_function(input2)
    assert result == expected2
    
def test_edge_cases():
    # Test edge cases
    with pytest.raises(ValueError):
        your_function(invalid_input)
"""


class ResearchAgent(Agent):
    """Agent specialized in research and information gathering."""
    
    def __init__(self, llm: LiteLLMProvider):
        super().__init__(
            name="Researcher",
            llm=llm,
            tools=[WebSearchTool()],
            memory=VectorMemory(embedding_model="all-MiniLM-L6-v2"),
            prompting_strategy=ChainOfThoughtStrategy(),
            system_prompt="""You are a research specialist. Your role is to:
            1. Gather accurate, up-to-date information
            2. Verify sources and cross-reference facts
            3. Summarize findings clearly
            4. Identify knowledge gaps
            
            Always cite your sources and indicate confidence levels."""
        )


class CodeAgent(Agent):
    """Agent specialized in code generation and analysis."""
    
    def __init__(self, llm: LiteLLMProvider):
        super().__init__(
            name="Coder",
            llm=llm,
            tools=[CodeExecutionTool(), analyze_code, generate_tests],
            prompting_strategy=TreeOfThoughtsStrategy(num_thoughts=3),
            system_prompt="""You are an expert software engineer. Your role is to:
            1. Write clean, efficient, and secure code
            2. Follow best practices and design patterns
            3. Include proper error handling
            4. Write comprehensive tests
            5. Document your code thoroughly
            
            Consider multiple approaches before implementing."""
        )


class ReviewAgent(Agent):
    """Agent specialized in reviewing and improving work."""
    
    def __init__(self, llm: LiteLLMProvider):
        super().__init__(
            name="Reviewer",
            llm=llm,
            tools=[analyze_code],
            system_prompt="""You are a senior technical reviewer. Your role is to:
            1. Review work for quality and completeness
            2. Identify potential issues and improvements
            3. Suggest optimizations
            4. Ensure requirements are met
            5. Provide constructive feedback
            
            Be thorough but constructive in your reviews."""
        )


class ProjectManagerAgent(Agent):
    """Agent that coordinates other agents."""
    
    def __init__(self, llm: LiteLLMProvider):
        super().__init__(
            name="ProjectManager",
            llm=llm,
            tools=[],
            system_prompt="""You are a project manager coordinating a team. Your role is to:
            1. Break down complex tasks into subtasks
            2. Assign tasks to appropriate team members
            3. Coordinate between team members
            4. Ensure project requirements are met
            5. Summarize results for stakeholders
            
            Team members:
            - Researcher: For information gathering
            - Coder: For implementation
            - Reviewer: For quality assurance"""
        )


async def main():
    """Run the multi-agent example."""
    
    # Initialize LLM for all agents
    llm = LiteLLMProvider(model="gpt-4", temperature=0.7)
    
    # Create specialized agents
    researcher = ResearchAgent(llm)
    coder = CodeAgent(llm)
    reviewer = ReviewAgent(llm)
    manager = ProjectManagerAgent(llm)
    
    # Create orchestrator with hierarchical workflow
    orchestrator = Orchestrator(
        agents=[manager, researcher, coder, reviewer],
        workflow=WorkflowType.HIERARCHICAL,
        manager_agent=manager
    )
    
    print("Multi-Agent System Example")
    print("=" * 50)
    
    # Example 1: Simple collaborative task
    print("\n1. Collaborative Task - Create a Web Scraper:")
    task1 = """
    Create a Python web scraper that:
    1. Scrapes product information from an e-commerce site
    2. Handles pagination
    3. Saves data to CSV
    4. Includes error handling and rate limiting
    """
    
    result = await orchestrator.run(task1)
    print(f"Task: {task1}")
    print(f"\nResult:\n{result}")
    
    # Example 2: Research and implementation task
    print("\n2. Research & Implementation - ML Model:")
    task2 = """
    Research and implement a machine learning model for sentiment analysis:
    1. Research current best practices for sentiment analysis
    2. Choose appropriate model and libraries
    3. Implement the solution with sample data
    4. Include evaluation metrics
    5. Write tests and documentation
    """
    
    result = await orchestrator.run(task2)
    print(f"Task: {task2}")
    print(f"\nResult:\n{result}")
    
    # Example 3: Complex project with multiple iterations
    print("\n3. Complex Project - REST API Development:")
    task3 = """
    Design and implement a REST API for a task management system:
    1. Research best practices for REST API design
    2. Design the API endpoints and data models
    3. Implement using FastAPI
    4. Include authentication and authorization
    5. Write comprehensive tests
    6. Create API documentation
    7. Review and optimize the implementation
    """
    
    # Enable detailed workflow tracking
    orchestrator.enable_tracking = True
    
    result = await orchestrator.run(task3)
    print(f"Task: {task3}")
    print(f"\nResult:\n{result}")
    
    # Show workflow details
    print("\n" + "=" * 50)
    print("Workflow Execution Details:")
    workflow_details = orchestrator.get_workflow_details()
    
    for step in workflow_details["steps"]:
        print(f"\nStep {step['step_number']}: {step['agent']}")
        print(f"Task: {step['task'][:100]}...")
        print(f"Duration: {step['duration']:.2f}s")
        print(f"Status: {step['status']}")
    
    # Show agent collaboration statistics
    print("\n" + "=" * 50)
    print("Agent Collaboration Statistics:")
    stats = orchestrator.get_statistics()
    
    print(f"Total tasks completed: {stats['total_tasks']}")
    print(f"Average completion time: {stats['avg_completion_time']:.2f}s")
    print(f"Success rate: {stats['success_rate']:.1%}")
    
    print("\nAgent Performance:")
    for agent_name, agent_stats in stats['agent_stats'].items():
        print(f"\n{agent_name}:")
        print(f"  - Tasks handled: {agent_stats['tasks_handled']}")
        print(f"  - Average time: {agent_stats['avg_time']:.2f}s")
        print(f"  - Tool calls: {agent_stats['tool_calls']}")
    
    # Example 4: Parallel execution
    print("\n" + "=" * 50)
    print("4. Parallel Execution - Multiple Features:")
    
    # Create parallel orchestrator
    parallel_orchestrator = Orchestrator(
        agents=[researcher, coder, reviewer],
        workflow=WorkflowType.PARALLEL
    )
    
    tasks = [
        "Research and summarize the latest developments in quantum computing",
        "Write a Python function to calculate Fibonacci numbers efficiently",
        "Review this code for security issues: 'user_input = input(); eval(user_input)'"
    ]
    
    print("Executing tasks in parallel...")
    results = await parallel_orchestrator.run_batch(tasks)
    
    for i, (task, result) in enumerate(zip(tasks, results, strict=False)):
        print(f"\nTask {i+1}: {task}")
        print(f"Result: {result[:200]}...")
    
    # Save the session for analysis
    print("\n" + "=" * 50)
    print("Saving session data...")
    
    session_data = {
        "orchestrator_config": orchestrator.get_config(),
        "workflow_history": orchestrator.get_workflow_details(),
        "agent_memories": {}
    }
    
    # Save individual agent memories
    for agent in [researcher, coder, reviewer, manager]:
        if hasattr(agent, 'memory') and agent.memory:
            session_data["agent_memories"][agent.name] = await agent.memory.export()
    
    # Save to file
    import json
    with open("multi_agent_session.json", "w") as f:
        json.dump(session_data, f, indent=2, default=str)
    
    print("Session saved to multi_agent_session.json")


if __name__ == "__main__":
    # Run the multi-agent example
    asyncio.run(main())
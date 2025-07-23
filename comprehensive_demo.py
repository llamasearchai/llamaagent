#!/usr/bin/env python3
"""
Comprehensive LlamaAgent Demo
Demonstrates all major features of the framework
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llamaagent.agents.react import ReactAgent
from llamaagent.agents.base import AgentConfig
from llamaagent.llm import create_provider
from llamaagent.types import TaskInput
from llamaagent.tools import CalculatorTool, PythonREPLTool, ToolRegistry


async def main():
    """Run comprehensive demo of LlamaAgent features"""
    
    print("=" * 80)
    print("LLAMAAGENT COMPREHENSIVE DEMO")
    print("=" * 80)
    print()
    
    # 1. Initialize LLM Provider
    print("1. Initializing LLM Provider...")
    llm_provider = create_provider("mock")
    print(f"   Provider: {llm_provider.__class__.__name__}")
    print(f"   Model: {llm_provider.model_name}")
    print()
    
    # 2. Create Agent Configuration
    print("2. Creating Agent Configuration...")
    config = AgentConfig(
        name="ComprehensiveAgent",
        max_iterations=10,
        enable_tools=True,
        enable_memory=True,
        enable_logging=True,
        debug_mode=False
    )
    print(f"   Name: {config.name}")
    print(f"   Max iterations: {config.max_iterations}")
    print(f"   Tools enabled: {config.enable_tools}")
    print()
    
    # 3. Initialize Tools
    print("3. Setting up Tool Registry...")
    tool_registry = ToolRegistry()
    
    # Add calculator tool
    calc_tool = CalculatorTool()
    tool_registry.register(calc_tool)
    print(f"   Registered: {calc_tool.name}")
    
    # Add Python REPL tool
    repl_tool = PythonREPLTool()
    tool_registry.register(repl_tool)
    print(f"   Registered: {repl_tool.name}")
    print()
    
    # 4. Create Agent
    print("4. Creating React Agent with SPRE methodology...")
    agent = ReactAgent(
        config=config,
        llm_provider=llm_provider,
        tools=tool_registry
    )
    print(f"   Agent ID: {agent.agent_id}")
    print(f"   Agent Name: {agent.name}")
    print()
    
    # 5. Define Test Tasks
    test_tasks = [
        {
            "name": "Simple Math",
            "task": "Calculate the square root of 144",
            "context": {"requires_calculation": True}
        },
        {
            "name": "Complex Reasoning",
            "task": "Explain the difference between machine learning and deep learning, then provide a simple Python example",
            "context": {"requires_explanation": True, "requires_code": True}
        },
        {
            "name": "Multi-step Problem",
            "task": "Calculate the factorial of 5, then explain what a factorial is, and finally write a Python function to calculate factorials",
            "context": {"multi_step": True}
        }
    ]
    
    # 6. Execute Tasks
    print("5. Executing Tasks...")
    print("-" * 80)
    
    for i, task_info in enumerate(test_tasks, 1):
        print(f"\nTask {i}: {task_info['name']}")
        print(f"Description: {task_info['task']}")
        print("-" * 60)
        
        # Create task input
        task = TaskInput(
            id=f"demo-task-{i}",
            task=task_info['task'],
            context=task_info['context']
        )
        
        try:
            # Execute task
            result = await agent.arun(task)
            
            # Display results
            print(f"Status: {result.status.value}")
            print(f"Success: {result.result.success}")
            
            if result.result.data.get("content"):
                content = result.result.data['content']
                # Truncate long responses for display
                if len(content) > 200:
                    content = content[:200] + "..."
                print(f"Response: {content}")
            
            if result.result.error:
                print(f"Error: {result.result.error}")
                
        except Exception as e:
            print(f"Exception occurred: {e}")
    
    print()
    print("-" * 80)
    
    # 7. Display Performance Metrics
    print("\n6. Agent Performance Metrics:")
    print("-" * 60)
    metrics = await agent.get_performance_metrics()
    for key, value in metrics.items():
        print(f"   {key}: {value}")
    
    # 8. Show Reasoning Explanation
    print("\n7. Agent Reasoning Methodology:")
    print("-" * 60)
    reasoning = await agent.explain_reasoning()
    print(reasoning)
    
    # 9. Cleanup
    print("\n8. Cleaning up resources...")
    await agent.cleanup()
    
    print()
    print("=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
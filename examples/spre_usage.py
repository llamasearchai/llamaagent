#!/usr/bin/env python3
"""
SPRE Usage Example - Demonstrating Strategic Planning & Resourceful Execution

This example demonstrates the complete SPRE methodology implementation,
showing how to programmatically invoke the SPRE agent on complex tasks,
print the generated plan, and observe step-by-step execution including
resource assessment decisions.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llamaagent.agents.base import AgentConfig, AgentRole
from llamaagent.agents.react import ReactAgent
from llamaagent.tools import ToolRegistry, get_all_tools
from llamaagent.benchmarks import SPREEvaluator, BaselineAgentFactory


def print_section(title: str, content: str = "") -> None:
    """Print formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    if content:
        print(content)


def print_subsection(title: str) -> None:
    """Print formatted subsection header."""
    print(f"\n{'-'*40}")
    print(f"  {title}")
    print(f"{'-'*40}")


async def demonstrate_spre_agent():
    """Demonstrate SPRE agent on a complex multi-step task."""
    
    print_section("SPRE Agent Demonstration", 
                  "Strategic Planning & Resourceful Execution in Action")
    
    # Create SPRE-enabled agent
    tools = ToolRegistry()
    for tool in get_all_tools():
        tools.register(tool)
    
    config = AgentConfig(
        name="SPRE-Demo-Agent",
        role=AgentRole.PLANNER,
        spree_enabled=True,
        max_iterations=10
    )
    
    agent = ReactAgent(config, tools=tools)
    
    # Complex multi-step task
    task = """Calculate the compound interest on $5000 invested at 8% annual rate for 5 years, 
    then write a Python function that can calculate compound interest for any principal, rate, 
    and time period, and finally determine how much you would need to invest initially at 
    the same rate to have $10,000 after 5 years."""
    
    print_subsection("Task")
    print(task)
    
    print_subsection("Executing SPRE Pipeline...")
    
    # Execute with full tracing
    response = await agent.execute(task)
    
    print_subsection("Execution Results")
    print(f"Success: {response.success}")
    print(f"Execution Time: {response.execution_time:.2f}s")
    print(f"Tokens Used: {response.tokens_used}")
    
    print_subsection("Final Answer")
    print(response.content)
    
    # Analyze the execution trace
    print_subsection("Execution Trace Analysis")
    
    planning_events = [e for e in response.trace if e["type"] in ["planning_start", "plan_generated"]]
    resource_events = [e for e in response.trace if e["type"] == "resource_assessment"]
    tool_events = [e for e in response.trace if e["type"] in ["tool_execution_start", "tool_execution_success"]]
    synthesis_events = [e for e in response.trace if e["type"] == "synthesis_complete"]
    
    print(f"Planning Events: {len(planning_events)}")
    print(f"Resource Assessment Events: {len(resource_events)}")
    print(f"Tool Execution Events: {len(tool_events)}")
    print(f"Synthesis Events: {len(synthesis_events)}")
    
    # Show plan details
    plan_event = next((e for e in response.trace if e["type"] == "plan_generated"), None)
    if plan_event and "plan" in plan_event["data"]:
        plan_data = plan_event["data"]["plan"]
        print_subsection("Generated Plan")
        print(f"Original Task: {plan_data.get('original_task', 'N/A')}")
        print(f"Number of Steps: {plan_data.get('num_steps', len(plan_data.get('steps', [])))}")
        
        steps = plan_data.get("steps", [])
        for i, step in enumerate(steps, 1):
            if hasattr(step, 'description'):
                # PlanStep object
                print(f"\nStep {i}:")
                print(f"  Description: {step.description}")
                print(f"  Required Info: {step.required_information}")
                print(f"  Expected Outcome: {step.expected_outcome}")
            else:
                # Dictionary
                print(f"\nStep {i}:")
                print(f"  Description: {step.get('description', 'N/A')}")
                print(f"  Required Info: {step.get('required_information', 'N/A')}")
                print(f"  Expected Outcome: {step.get('expected_outcome', 'N/A')}")
    
    # Show resource assessment decisions
    print_subsection("Resource Assessment Decisions")
    for event in resource_events:
        data = event["data"]
        step_id = data.get("step_id", "Unknown")
        needs_tool = data.get("needs_tool", False)
        print(f"Step {step_id}: {'TOOL REQUIRED' if needs_tool else 'INTERNAL KNOWLEDGE'}")
    
    return response


async def run_baseline_comparison():
    """Run comparison across all baseline types."""
    
    print_section("Baseline Comparison", 
                  "Comparing SPRE against other agent configurations")
    
    # Simple task for quick comparison
    simple_task = "Calculate 25 * 16 and then find the square root of the result"
    
    results = {}
    
    for baseline_type in BaselineAgentFactory.get_all_baseline_types():
        print_subsection(f"Testing {baseline_type}")
        
        agent = BaselineAgentFactory.create_agent(baseline_type)
        
        import time
        start_time = time.time()
        response = await agent.execute(simple_task)
        execution_time = time.time() - start_time
        
        # Count API calls
        api_calls = len([e for e in response.trace 
                        if e.get("type") in ["planner_response", "resource_assessment_detail", 
                                           "internal_execution", "synthesis_complete"]])
        
        results[baseline_type] = {
            "success": response.success,
            "execution_time": execution_time,
            "api_calls": api_calls,
            "answer": response.content[:100] + "..." if len(response.content) > 100 else response.content
        }
        
        print(f"Success: {response.success}")
        print(f"Time: {execution_time:.2f}s")
        print(f"API Calls: {api_calls}")
        print(f"Answer: {results[baseline_type]['answer']}")
    
    # Summary comparison
    print_subsection("Comparison Summary")
    print(f"{'Agent Type':<15} {'Success':<8} {'Time (s)':<10} {'API Calls':<10}")
    print("-" * 50)
    
    for baseline_type, result in results.items():
        print(f"{baseline_type:<15} {str(result['success']):<8} {result['execution_time']:<10.2f} {result['api_calls']:<10}")
    
    return results


async def run_full_benchmark():
    """Run full benchmark evaluation."""
    
    print_section("Full Benchmark Evaluation", 
                  "Running comprehensive GAIA benchmark evaluation")
    
    evaluator = SPREEvaluator()
    
    # Run evaluation with limited tasks for demo
    results = await evaluator.run_full_evaluation(
        task_filter={"min_steps": 2},
        max_tasks_per_baseline=5
    )
    
    print_subsection("Benchmark Results Summary")
    
    print(f"{'Agent Configuration':<20} {'Success Rate':<12} {'Avg API Calls':<15} {'Efficiency':<10}")
    print("-" * 70)
    
    for baseline_type, result in results.items():
        print(f"{result.agent_name:<20} {result.success_rate:<12.1f}% {result.avg_api_calls:<15.1f} {result.efficiency_ratio:<10.2f}")
    
    # Find the best performer
    best_agent = max(results.values(), key=lambda r: r.efficiency_ratio)
    print(f"\nBest Performing Agent: {best_agent.agent_name}")
    print(f"Efficiency Ratio: {best_agent.efficiency_ratio:.2f}")
    
    return results


async def demonstrate_interactive_planning():
    """Show interactive planning process."""
    
    print_section("Interactive Planning Demonstration",
                  "Step-by-step breakdown of SPRE planning process")
    
    # Create agent with detailed tracing
    config = AgentConfig(
        name="Interactive-SPRE",
        role=AgentRole.PLANNER,
        spree_enabled=True
    )
    
    tools = ToolRegistry()
    for tool in get_all_tools():
        tools.register(tool)
    
    agent = ReactAgent(config, tools=tools)
    
    # Complex reasoning task
    task = """Analyze the efficiency of different sorting algorithms: 
    1) Explain the time complexity of bubble sort, quick sort, and merge sort
    2) Write Python code to implement bubble sort
    3) Calculate how long each would take to sort 10,000 elements (assuming 1 operation = 1 microsecond)
    4) Recommend which algorithm to use for different scenarios"""
    
    print_subsection("Task")
    print(task)
    
    print_subsection("Planning Phase")
    plan = await agent._generate_plan(task)
    
    print(f"Generated {len(plan.steps)} steps:")
    for i, step in enumerate(plan.steps, 1):
        print(f"\nStep {i}: {step.description}")
        print(f"  Required Info: {step.required_information}")
        print(f"  Expected Outcome: {step.expected_outcome}")
    
    print_subsection("Resource Assessment Phase")
    for step in plan.steps:
        needs_tool = await agent._assess_resource_need(step)
        method = "TOOL-BASED" if needs_tool else "INTERNAL KNOWLEDGE"
        print(f"Step {step.step_id}: {method}")
    
    print_subsection("Execution Phase")
    step_results = await agent._execute_plan_with_resource_assessment(plan)
    
    for result in step_results:
        print(f"\nStep {result['step_id']}: {result['description']}")
        print(f"Method: {result['execution_method']}")
        print(f"Result: {result['result'][:200]}...")
    
    print_subsection("Synthesis Phase")
    final_answer = await agent._synthesize_results(plan, step_results)
    print(final_answer)


async def main():
    """Main demonstration function."""
    
    print_section("LlamaAgent SPRE Demonstration Suite",
                  "Comprehensive showcase of Strategic Planning & Resourceful Execution")
    
    try:
        # 1. Basic SPRE demonstration
        await demonstrate_spre_agent()
        
        # 2. Baseline comparison
        await run_baseline_comparison()
        
        # 3. Interactive planning
        await demonstrate_interactive_planning()
        
        # 4. Full benchmark (optional - can be slow)
        response = input("\nRun full benchmark evaluation? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            await run_full_benchmark()
        
        print_section("Demonstration Complete",
                      "SPRE methodology successfully demonstrated across multiple scenarios")
        
    except KeyboardInterrupt:
        print("\n\nDemonstration interrupted by user.")
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 
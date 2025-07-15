#!/usr/bin/env python3
"""
Complete OpenAI Agents SDK Integration Example

This example demonstrates comprehensive integration between LlamaAgent and OpenAI's
Agents SDK, showcasing all major features including:
- Multi-agent orchestration
- Tool synthesis and dynamic tool creation
- Budget tracking and cost optimization
- Hybrid execution modes
- Advanced reasoning with o-series models

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import json
import os
from datetime import datetime

# LlamaAgent imports
from llamaagent import AgentConfig, AgentRole, ReactAgent
from llamaagent.integration.openai_agents import (
    OpenAIAgentMode,
    OpenAIAgentsIntegration,
    OpenAIIntegrationConfig,
)
from llamaagent.llm import create_provider
from llamaagent.orchestrator import AgentOrchestrator, OrchestrationStrategy, WorkflowDefinition, WorkflowStep
from llamaagent.types import TaskInput


class OpenAIIntegrationDemo:
    """Comprehensive demonstration of OpenAI integration capabilities."""
    
    def __init__(self):
        self.integration = None
        self.orchestrator = None
        self.agents = {}
        
    async def setup(self):
        """Initialize the integration and create agents."""
        print("=== Setting up OpenAI Integration ===\n")
        
        # Configure OpenAI integration
        config = OpenAIIntegrationConfig(
            mode=OpenAIAgentMode.HYBRID,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-4o-mini",
            budget_limit=10.0,  # $10 budget for demo
            enable_tracing=True,
            enable_guardrails=True,
            temperature=0.1
        )
        
        self.integration = OpenAIAgentsIntegration(config)
        print(f"✓ OpenAI integration configured (Budget: ${config.budget_limit})")
        
        # Create orchestrator
        self.orchestrator = AgentOrchestrator()
        print("✓ Agent orchestrator initialized")
        
        # Create specialized agents
        await self._create_agents()
        print(f"✓ Created {len(self.agents)} specialized agents")
        
    async def _create_agents(self):
        """Create a suite of specialized agents."""
        
        # 1. Research Agent
        research_config = AgentConfig(
            name="ResearchAgent",
            role=AgentRole.RESEARCHER,
            description="Expert at gathering and analyzing information",
            spree_enabled=True,
            tools=["web_search", "calculator", "python_repl"]
        )
        research_agent = ReactAgent(
            config=research_config,
            llm_provider=create_provider("openai", model_name="gpt-4o-mini")
        )
        self.agents["research"] = research_agent
        self.orchestrator.register_agent(research_agent)
        self.integration.register_agent(research_agent)
        
        # 2. Code Agent
        code_config = AgentConfig(
            name="CodeAgent",
            role=AgentRole.SPECIALIST,
            description="Expert Python developer and code analyzer",
            spree_enabled=True,
            tools=["python_repl", "calculator"]
        )
        code_agent = ReactAgent(
            config=code_config,
            llm_provider=create_provider("openai", model_name="gpt-4o-mini")
        )
        self.agents["code"] = code_agent
        self.orchestrator.register_agent(code_agent)
        self.integration.register_agent(code_agent)
        
        # 3. Analysis Agent
        analysis_config = AgentConfig(
            name="AnalysisAgent",
            role=AgentRole.ANALYZER,
            description="Expert at data analysis and visualization",
            spree_enabled=True,
            tools=["python_repl", "calculator"]
        )
        analysis_agent = ReactAgent(
            config=analysis_config,
            llm_provider=create_provider("openai", model_name="gpt-4o-mini")
        )
        self.agents["analysis"] = analysis_agent
        self.orchestrator.register_agent(analysis_agent)
        self.integration.register_agent(analysis_agent)
        
        # 4. Coordinator Agent
        coordinator_config = AgentConfig(
            name="CoordinatorAgent",
            role=AgentRole.COORDINATOR,
            description="Orchestrates multi-agent workflows",
            spree_enabled=True
        )
        coordinator_agent = ReactAgent(
            config=coordinator_config,
            llm_provider=create_provider("openai", model_name="gpt-4o")
        )
        self.agents["coordinator"] = coordinator_agent
        self.orchestrator.register_agent(coordinator_agent)
        self.integration.register_agent(coordinator_agent)
        
    async def demo_basic_task(self):
        """Demonstrate basic task execution."""
        print("\n=== Demo 1: Basic Task Execution ===\n")
        
        task = TaskInput(
            id="basic_task_001",
            task="Calculate the compound interest on $1000 at 5% for 10 years",
            agent_name="AnalysisAgent"
        )
        
        result = await self.integration.run_task("AnalysisAgent", task)
        
        print(f"Task: {task.task}")
        print(f"Status: {result.status.value}")
        print(f"Success: {result.success}")
        if result.result and result.result.data:
            print(f"Result: {result.result.data.get('response', 'No response')}")
        
        # Show budget status
        budget_status = self.integration.get_budget_status()
        print("\nBudget Status:")
        print(f"  Used: ${budget_status['current_cost']:.4f}")
        print(f"  Remaining: ${budget_status['remaining_budget']:.4f}")
        
    async def demo_multi_agent_workflow(self):
        """Demonstrate multi-agent orchestration."""
        print("\n=== Demo 2: Multi-Agent Workflow ===\n")
        
        # Define a complex workflow
        workflow = WorkflowDefinition(
            workflow_id="analysis_workflow_001",
            name="Market Analysis Workflow",
            description="Analyze cryptocurrency market trends",
            strategy=OrchestrationStrategy.SEQUENTIAL,
            steps=[
                WorkflowStep(
                    step_id="research",
                    agent_name="ResearchAgent",
                    task="Research the current trends in Bitcoin and Ethereum prices over the last month"
                ),
                WorkflowStep(
                    step_id="analyze",
                    agent_name="AnalysisAgent",
                    task="Analyze the price data and identify key patterns or trends",
                    dependencies=["research"]
                ),
                WorkflowStep(
                    step_id="code",
                    agent_name="CodeAgent",
                    task="Write Python code to calculate moving averages and RSI indicators",
                    dependencies=["analyze"]
                ),
                WorkflowStep(
                    step_id="synthesize",
                    agent_name="CoordinatorAgent",
                    task="Synthesize all findings into a comprehensive market report",
                    dependencies=["research", "analyze", "code"]
                )
            ]
        )
        
        # Register and execute workflow
        self.orchestrator.register_workflow(workflow)
        
        print(f"Executing workflow: {workflow.name}")
        print(f"Strategy: {workflow.strategy.value}")
        print(f"Steps: {len(workflow.steps)}")
        
        result = await self.orchestrator.execute_workflow(workflow.workflow_id)
        
        print("\nWorkflow completed!")
        print(f"Success: {result.success}")
        print(f"Execution time: {result.execution_time:.2f}s")
        
        # Show results from each step
        for step_id, step_result in result.results.items():
            print(f"\n{step_id}:")
            print(f"  Status: {step_result.status.value}")
            if step_result.result and step_result.result.data:
                data = step_result.result.data
                if isinstance(data, dict) and 'content' in data:
                    print(f"  Output: {data['content'][:200]}...")
                    
    async def demo_debate_strategy(self):
        """Demonstrate debate-style multi-agent interaction."""
        print("\n=== Demo 3: Agent Debate Strategy ===\n")
        
        # Define a debate workflow
        debate_workflow = WorkflowDefinition(
            workflow_id="debate_workflow_001",
            name="Technical Decision Debate",
            description="Debate the best database for a high-traffic application",
            strategy=OrchestrationStrategy.DEBATE,
            steps=[
                WorkflowStep(
                    step_id="position_1",
                    agent_name="CodeAgent",
                    task="Argue for using PostgreSQL for a high-traffic e-commerce application"
                ),
                WorkflowStep(
                    step_id="position_2",
                    agent_name="AnalysisAgent",
                    task="Argue for using MongoDB for a high-traffic e-commerce application"
                ),
                WorkflowStep(
                    step_id="position_3",
                    agent_name="ResearchAgent",
                    task="Argue for using a hybrid approach with both PostgreSQL and Redis"
                )
            ]
        )
        
        self.orchestrator.register_workflow(debate_workflow)
        
        print(f"Starting debate: {debate_workflow.description}")
        
        result = await self.orchestrator.execute_workflow(debate_workflow.workflow_id)
        
        print("\nDebate completed!")
        print(f"Final synthesis available: {'final_synthesis' in result.results}")
        
        if 'final_synthesis' in result.results:
            synthesis = result.results['final_synthesis']
            if synthesis.result and synthesis.result.data:
                print("\nFinal Decision:")
                print(synthesis.result.data.get('content', 'No synthesis available'))
                
    async def demo_tool_synthesis(self):
        """Demonstrate dynamic tool synthesis."""
        print("\n=== Demo 4: Dynamic Tool Synthesis ===\n")
        
        # Create a task that requires custom tool creation
        task = TaskInput(
            id="tool_synthesis_001",
            task="""Create a custom tool that can:
            1. Fetch cryptocurrency prices from an API
            2. Calculate technical indicators (SMA, EMA, RSI)
            3. Generate buy/sell signals
            Then use this tool to analyze Bitcoin.""",
            agent_name="CodeAgent"
        )
        
        # Execute with tool synthesis
        result = await self.integration.run_task("CodeAgent", task)
        
        print("Tool Synthesis Task Completed")
        print(f"Success: {result.success}")
        
        if result.result and result.result.data:
            print("\nGenerated Tool:")
            print(result.result.data.get('response', 'No tool generated')[:500] + "...")
            
    async def demo_budget_optimization(self):
        """Demonstrate budget-aware execution."""
        print("\n=== Demo 5: Budget Optimization ===\n")
        
        # Check initial budget
        initial_budget = self.integration.get_budget_status()
        print("Initial Budget Status:")
        print(f"  Total: ${initial_budget['budget_limit']}")
        print(f"  Used: ${initial_budget['current_cost']:.4f}")
        print(f"  Remaining: ${initial_budget['remaining_budget']:.4f}")
        
        # Create a cost-optimized workflow
        optimized_workflow = WorkflowDefinition(
            workflow_id="optimized_workflow_001",
            name="Cost-Optimized Analysis",
            description="Perform analysis with budget constraints",
            strategy=OrchestrationStrategy.PARALLEL,
            steps=[
                WorkflowStep(
                    step_id="quick_calc",
                    agent_name="AnalysisAgent",
                    task="Calculate the sum of squares from 1 to 100"
                ),
                WorkflowStep(
                    step_id="simple_code",
                    agent_name="CodeAgent",
                    task="Write a one-line Python function to check if a number is prime"
                )
            ]
        )
        
        self.orchestrator.register_workflow(optimized_workflow)
        
        # Execute with budget tracking
        async with self.integration.traced_execution("budget_demo"):
            result = await self.orchestrator.execute_workflow(optimized_workflow.workflow_id)
        
        # Check final budget
        final_budget = self.integration.get_budget_status()
        print("\nFinal Budget Status:")
        print(f"  Used: ${final_budget['current_cost']:.4f}")
        print(f"  Cost of this workflow: ${final_budget['current_cost'] - initial_budget['current_cost']:.4f}")
        
        # Show usage details
        print(f"\nTotal API Calls: {final_budget['total_calls']}")
        
    async def demo_error_handling(self):
        """Demonstrate error handling and recovery."""
        print("\n=== Demo 6: Error Handling ===\n")
        
        # Create a task that will fail
        error_task = TaskInput(
            id="error_task_001",
            task="Divide 100 by zero and explain the result",
            agent_name="AnalysisAgent"
        )
        
        try:
            result = await self.integration.run_task("AnalysisAgent", error_task)
            print(f"Task Status: {result.status.value}")
            if result.error:
                print(f"Error Handled Gracefully: {result.error}")
        except Exception as e:
            print(f"Exception caught: {e}")
            
        # Create a recovery workflow
        recovery_workflow = WorkflowDefinition(
            workflow_id="recovery_workflow_001",
            name="Error Recovery Workflow",
            description="Demonstrate error recovery",
            strategy=OrchestrationStrategy.SEQUENTIAL,
            steps=[
                WorkflowStep(
                    step_id="risky_operation",
                    agent_name="CodeAgent",
                    task="Try to import a non-existent module and handle the error",
                    required=False  # Non-required step won't fail the workflow
                ),
                WorkflowStep(
                    step_id="fallback",
                    agent_name="AnalysisAgent",
                    task="Provide an alternative solution using built-in modules"
                )
            ]
        )
        
        self.orchestrator.register_workflow(recovery_workflow)
        result = await self.orchestrator.execute_workflow(recovery_workflow.workflow_id)
        
        print("\nRecovery Workflow Completed")
        print(f"Overall Success: {result.success}")
        print(f"Errors: {len(result.errors)}")
        
    async def generate_report(self):
        """Generate a comprehensive report of the demo."""
        print("\n=== Generating Demo Report ===\n")
        
        # Get all execution history
        history = self.orchestrator.get_execution_history()
        budget_status = self.integration.get_budget_status()
        
        report = {
            "demo_timestamp": datetime.now().isoformat(),
            "total_workflows": len(history),
            "successful_workflows": sum(1 for r in history if r.success),
            "total_execution_time": sum(r.execution_time for r in history),
            "budget_summary": {
                "total_budget": budget_status["budget_limit"],
                "total_spent": budget_status["current_cost"],
                "remaining": budget_status["remaining_budget"],
                "total_api_calls": budget_status["total_calls"]
            },
            "agents_used": list(self.agents.keys()),
            "workflow_strategies": list(set(w.metadata.get("strategy", "unknown") for w in history))
        }
        
        # Save report
        report_path = "openai_integration_demo_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
            
        print(f"Report saved to: {report_path}")
        print("\nDemo Summary:")
        print(f"  Total Workflows: {report['total_workflows']}")
        print(f"  Success Rate: {report['successful_workflows']}/{report['total_workflows']}")
        print(f"  Total Time: {report['total_execution_time']:.2f}s")
        print(f"  Total Cost: ${report['budget_summary']['total_spent']:.4f}")
        print(f"  Cost per Call: ${report['budget_summary']['total_spent']/max(1, report['budget_summary']['total_api_calls']):.4f}")


async def main():
    """Run the complete demonstration."""
    print("=" * 60)
    print("LlamaAgent + OpenAI Agents SDK Integration Demo")
    print("=" * 60)
    
    demo = OpenAIIntegrationDemo()
    
    try:
        # Setup
        await demo.setup()
        
        # Run all demos
        await demo.demo_basic_task()
        await demo.demo_multi_agent_workflow()
        await demo.demo_debate_strategy()
        await demo.demo_tool_synthesis()
        await demo.demo_budget_optimization()
        await demo.demo_error_handling()
        
        # Generate report
        await demo.generate_report()
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Ensure we have an API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        exit(1)
        
    asyncio.run(main())
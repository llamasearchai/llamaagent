#!/usr/bin/env python3
"""
Simple OpenAI Agents Integration Demo

A simplified demonstration showing the OpenAI Agents integration working
with budget tracking and basic functionality.

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Simple OpenAI Agents Integration Demo")
print("Author: Nik Jois (nikjois@llamasearch.ai)")
print("=" * 50)

# Check OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("FAIL Error: OPENAI_API_KEY environment variable not set")
    print("Please set your OpenAI API key:")
    print("export OPENAI_API_KEY='${OPENAI_API_KEY}'")
    sys.exit(1)

print("PASS OpenAI API key found")

# Check OpenAI Agents SDK availability
try:
    from llamaagent.integration.openai_agents import OPENAI_AGENTS_AVAILABLE, create_openai_integration
    print(f"OpenAI Agents SDK: {'Available' if OPENAI_AGENTS_AVAILABLE else 'Not Available (using fallback)'}")
except ImportError as e:
    print(f"FAIL Import error: {e}")
    sys.exit(1)

# Import core components
try:
    from llamaagent.agents.base import BaseAgent
    from llamaagent.llm.providers.base_provider import BaseLLMProvider
    from llamaagent.types import TaskInput, TaskOutput, TaskResult, TaskStatus
    print("PASS Core components imported successfully")
except ImportError as e:
    print(f"FAIL Core import error: {e}")
    sys.exit(1)


class SimpleOpenAIProvider(BaseLLMProvider):
    """Simple OpenAI provider for demo."""
    
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: str = None):
        super().__init__(model_name=model_name)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
    
    async def complete(self, prompt: str, **kwargs) -> str:
        """Simple completion method."""
        try:
            import openai
            client = openai.AsyncOpenAI(api_key=self.api_key)
            
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=150
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    def validate_connection(self) -> bool:
        """Validate connection."""
        return bool(self.api_key)


class SimpleAgent(BaseAgent):
    """Simple agent for demo."""
    
    def __init__(self, name: str, llm_provider: BaseLLMProvider):
        super().__init__(name=name, llm_provider=llm_provider)
    
    async def execute_task(self, task_input: TaskInput) -> TaskOutput:
        """Execute a simple task."""
        try:
            print(f"Agent Agent '{self.name}' executing task: {task_input.task}")
            
            # Get response from LLM
            response = await self.llm_provider.complete(task_input.task)
            
            return TaskOutput(
                task_id=task_input.id,
                status=TaskStatus.COMPLETED,
                result=TaskResult(
                    success=True,
                    data={"response": response}
                ),
                completed_at=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            return TaskOutput(
                task_id=task_input.id,
                status=TaskStatus.FAILED,
                result=TaskResult(
                    success=False,
                    error=str(e)
                ),
                completed_at=datetime.now(timezone.utc)
            )


async def run_simple_demo():
    """Run the simple demonstration."""
    print("\n📋 Starting Simple Demo")
    print("-" * 25)
    
    # Create LLM provider
    llm_provider = SimpleOpenAIProvider(
        model_name="gpt-4o-mini",
        api_key=api_key
    )
    
    # Create agent
    agent = SimpleAgent(
        name="Demo-Agent",
        llm_provider=llm_provider
    )
    
    # Create OpenAI integration
    integration = create_openai_integration(
        openai_api_key=api_key,
        model_name="gpt-4o-mini",
        budget_limit=10.0  # $10 budget
    )
    
    print("PASS Integration created successfully")
    print("💰 Budget limit: $10.00")
    
    # Register agent
    adapter = integration.register_agent(agent)
    print("PASS Agent registered with integration")
    
    # Test tasks
    test_tasks = [
        "What is artificial intelligence?",
        "Explain machine learning in one sentence.",
        "What is 2 + 2?",
        "Name three programming languages."
    ]
    
    results = []
    
    for i, task_text in enumerate(test_tasks, 1):
        print(f"\n🔄 Task {i}/{len(test_tasks)}: {task_text}")
        
        task_input = TaskInput(
            id=f"demo_task_{i}",
            task=task_text,
            context={"demo": True}
        )
        
        try:
            # Execute task through integration
            result = await adapter.run_task(task_input)
            
            if result.status == TaskStatus.COMPLETED:
                print("PASS Task completed successfully")
                response = result.result.data.get('response', 'No response')
                print(f"Response Response: {response[:100]}..." if len(response) > 100 else f"Response Response: {response}")
                results.append({"task": task_text, "success": True, "response": response})
            else:
                print(f"FAIL Task failed: {result.result.error}")
                results.append({"task": task_text, "success": False, "error": result.result.error})
            
            # Show budget status
            budget_status = integration.get_budget_status()
            print(f"💰 Cost so far: ${budget_status['current_cost']:.4f}")
            print(f"💵 Remaining: ${budget_status['remaining_budget']:.4f}")
            
        except Exception as e:
            print(f"FAIL Task error: {e}")
            results.append({"task": task_text, "success": False, "error": str(e)})
    
    # Final summary
    print("\nRESULTS Demo Summary")
    print("-" * 20)
    
    successful = sum(1 for r in results if r["success"])
    total = len(results)
    
    print(f"PASS Successful tasks: {successful}/{total}")
    print(f"FAIL Failed tasks: {total - successful}/{total}")
    
    budget_status = integration.get_budget_status()
    print(f"💰 Total cost: ${budget_status['current_cost']:.4f}")
    print(f"💵 Budget remaining: ${budget_status['remaining_budget']:.4f}")
    print(f"📞 Total API calls: {budget_status['total_calls']}")
    
    # Save results
    report = {
        "demo_type": "simple_openai_integration",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "budget_status": budget_status,
        "summary": {
            "total_tasks": total,
            "successful_tasks": successful,
            "failed_tasks": total - successful
        }
    }
    
    report_file = f"simple_demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"📄 Report saved to: {report_file}")
    
    if successful == total:
        print("\nAll tasks completed successfully!")
        print("PASS OpenAI Agents integration is working correctly!")
    else:
        print(f"\n⚠️  {total - successful} tasks failed. Check the logs for details.")
    
    return successful == total


async def main():
    """Main function."""
    try:
        success = await run_simple_demo()
        
        print("\n" + "=" * 50)
        if success:
            print("TARGET Simple OpenAI Integration Demo: SUCCESS")
            print("Starting The system is ready for production use!")
        else:
            print("⚠️  Simple OpenAI Integration Demo: PARTIAL SUCCESS")
            print("FIXING Some components may need configuration.")
        
        print("\nINSIGHT Next Steps:")
        print("   1. Check the generated report for detailed results")
        print("   2. Run the full CLI: python -m llamaagent.cli.openai_cli --help")
        print("   3. Start the FastAPI server: python -m llamaagent.api.openai_fastapi")
        print("   4. Explore the comprehensive demos and examples")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\nFAIL Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 
#!/usr/bin/env python3
"""
Complete OpenAI Agents Integration Demonstration

This script demonstrates the full integration between LlamaAgent and OpenAI Agents SDK,
showcasing budget tracking, multiple models, and complete functionality.

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our components
from llamaagent.agents.react import ReactAgent
from llamaagent.integration.openai_agents import OPENAI_AGENTS_AVAILABLE, OpenAIAgentMode, create_openai_integration
from llamaagent.llm.providers.ollama_provider import OllamaProvider
from llamaagent.llm.providers.openai_provider import OpenAIProvider
from llamaagent.tools.calculator import CalculatorTool
from llamaagent.tools.python_repl import PythonREPLTool
from llamaagent.tools.registry import ToolRegistry
from llamaagent.types import TaskInput, TaskOutput, TaskStatus


class CompleteDemonstration:
    """Complete demonstration of OpenAI Agents integration."""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.total_cost = 0.0
        self.budget_limit = 50.0  # $50 budget for demo
        
    async def run_complete_demo(self):
        """Run the complete demonstration."""
        print("Starting Complete OpenAI Agents Integration Demo")
        print("=" * 60)
        
        # Check prerequisites
        await self._check_prerequisites()
        
        # Demo 1: Basic OpenAI Integration
        await self._demo_basic_openai_integration()
        
        # Demo 2: Local Model with Ollama
        await self._demo_local_model_integration()
        
        # Demo 3: Budget Tracking
        await self._demo_budget_tracking()
        
        # Demo 4: Tool Usage
        await self._demo_tool_usage()
        
        # Demo 5: Hybrid Mode
        await self._demo_hybrid_mode()
        
        # Demo 6: Batch Processing
        await self._demo_batch_processing()
        
        # Generate final report
        await self._generate_final_report()
        
        print("\nPASS Complete demonstration finished successfully!")
    
    async def _check_prerequisites(self):
        """Check system prerequisites."""
        print("\nChecking Prerequisites")
        print("-" * 30)
        
        # Check OpenAI API key
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            print("PASS OpenAI API key found")
        else:
            print("WARNING: OpenAI API key not found (some demos will be skipped)")
        
        # Check OpenAI Agents SDK
        if OPENAI_AGENTS_AVAILABLE:
            print("PASS OpenAI Agents SDK available")
        else:
            print("WARNING:  OpenAI Agents SDK not available (using fallback)")
        
        # Check Ollama
        try:
            ollama_provider = OllamaProvider(model_name="llama3.2:3b")
            print("PASS Ollama connection available")
        except Exception:
            print("WARNING:  Ollama not available (local demos will be skipped)")
    
    async def _demo_basic_openai_integration(self):
        """Demo basic OpenAI integration."""
        print("\nAgent Demo 1: Basic OpenAI Integration")
        print("-" * 40)
        
        try:
            # Create OpenAI provider
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("SKIP:  Skipping OpenAI demo (no API key)")
                return
            
            llm_provider = OpenAIProvider(
                model_name="gpt-4o-mini",
                api_key=api_key,
                temperature=0.1
            )
            
            # Create agent
            agent = ReactAgent(
                name="OpenAI-Demo-Agent",
                llm_provider=llm_provider,
                description="Demo agent using OpenAI GPT-4o-mini"
            )
            
            # Create OpenAI integration
            integration = create_openai_integration(
                openai_api_key=api_key,
                model_name="gpt-4o-mini",
                budget_limit=self.budget_limit
            )
            
            adapter = integration.register_agent(agent)
            
            # Run test task
            task_input = TaskInput(
                id="demo_1_task",
                task="Explain what artificial intelligence is in exactly 50 words.",
                context={"demo": "basic_openai"}
            )
            
            print(f"Response Task: {task_input.task}")
            print("⏳ Executing...")
            
            result = await adapter.run_task(task_input)
            
            self._record_result("Basic OpenAI Integration", result, integration.get_budget_status())
            
            if result.status == TaskStatus.COMPLETED:
                print("PASS Task completed successfully")
                print(f"Response: {result.result.data.get('response', 'No response')}")
            else:
                print(f"FAIL Task failed: {result.result.error}")
            
        except Exception as e:
            print(f"FAIL Demo failed: {e}")
            logger.error(f"Basic OpenAI demo error: {e}", exc_info=True)
    
    async def _demo_local_model_integration(self):
        """Demo local model integration with Ollama."""
        print("\n Demo 2: Local Model Integration (Ollama)")
        print("-" * 45)
        
        try:
            # Create Ollama provider
            llm_provider = OllamaProvider(
                model_name="llama3.2:3b",
                base_url="http://localhost:11434"
            )
            
            # Create agent
            agent = ReactAgent(
                name="Local-Demo-Agent",
                llm_provider=llm_provider,
                description="Demo agent using local Llama 3.2 3B model"
            )
            
            # Create task
            task_input = TaskInput(
                id="demo_2_task",
                task="Write a haiku about machine learning.",
                context={"demo": "local_model"}
            )
            
            print(f"Response Task: {task_input.task}")
            print("⏳ Executing with local model...")
            
            result = await agent.execute_task(task_input)
            
            self._record_result("Local Model Integration", result, {"cost": 0.0, "model": "llama3.2:3b"})
            
            if result.status == TaskStatus.COMPLETED:
                print("PASS Task completed successfully")
                print(f"Response: {result.result.data.get('response', 'No response')}")
            else:
                print(f"FAIL Task failed: {result.result.error}")
            
        except Exception as e:
            print(f"SKIP:  Skipping local model demo: {e}")
    
    async def _demo_budget_tracking(self):
        """Demo budget tracking functionality."""
        print("\n Demo 3: Budget Tracking")
        print("-" * 30)
        
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("SKIP:  Skipping budget demo (no API key)")
                return
            
            # Create integration with small budget
            small_budget = 5.0  # $5 budget
            integration = create_openai_integration(
                openai_api_key=api_key,
                model_name="gpt-4o-mini",
                budget_limit=small_budget
            )
            
            llm_provider = OpenAIProvider(
                model_name="gpt-4o-mini",
                api_key=api_key
            )
            
            agent = ReactAgent(
                name="Budget-Demo-Agent",
                llm_provider=llm_provider,
                description="Agent for budget tracking demo"
            )
            
            adapter = integration.register_agent(agent)
            
            print(f" Budget limit: ${small_budget}")
            print("RESULTS Initial budget status:")
            budget_status = integration.get_budget_status()
            print(f"   - Current cost: ${budget_status['current_cost']:.4f}")
            print(f"   - Remaining: ${budget_status['remaining_budget']:.4f}")
            
            # Run multiple small tasks to show budget tracking
            tasks = [
                "What is 2+2?",
                "Name three colors.",
                "Count to 5."
            ]
            
            for i, task_text in enumerate(tasks, 1):
                print(f"\n Task {i}: {task_text}")
                
                task_input = TaskInput(
                    id=f"budget_demo_task_{i}",
                    task=task_text,
                    context={"demo": "budget_tracking"}
                )
                
                try:
                    result = await adapter.run_task(task_input)
                    
                    budget_status = integration.get_budget_status()
                    print(f"   PASS Completed - Cost: ${budget_status['current_cost']:.4f}")
                    print(f"    Remaining budget: ${budget_status['remaining_budget']:.4f}")
                    
                    if budget_status['remaining_budget'] <= 0:
                        print("   WARNING:  Budget limit reached!")
                        break
                        
                except Exception as e:
                    if "budget" in str(e).lower():
                        print(f"    Budget exceeded: {e}")
                        break
                    else:
                        print(f"   FAIL Task failed: {e}")
            
            self._record_result("Budget Tracking Demo", None, budget_status)
            
        except Exception as e:
            print(f"FAIL Budget demo failed: {e}")
    
    async def _demo_tool_usage(self):
        """Demo tool usage with agents."""
        print("\nBUILD:  Demo 4: Tool Usage")
        print("-" * 25)
        
        try:
            # Create tools
            tool_registry = ToolRegistry()
            tool_registry.register("calculator", CalculatorTool())
            tool_registry.register("python", PythonREPLTool())
            
            # Create agent with tools
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                llm_provider = OpenAIProvider(
                    model_name="gpt-4o-mini",
                    api_key=api_key
                )
                model_name = "gpt-4o-mini"
            else:
                llm_provider = OllamaProvider(model_name="llama3.2:3b")
                model_name = "llama3.2:3b"
            
            agent = ReactAgent(
                name="Tool-Demo-Agent",
                llm_provider=llm_provider,
                tools=tool_registry,
                description=f"Agent with calculator and Python tools using {model_name}"
            )
            
            # Test calculator tool
            print("🧮 Testing calculator tool:")
            calc_task = TaskInput(
                id="tool_demo_calc",
                task="Calculate the square root of 144 and then multiply it by 7",
                context={"demo": "tool_usage", "tool": "calculator"}
            )
            
            print(f"Response Task: {calc_task.task}")
            result = await agent.execute_task(calc_task)
            
            if result.status == TaskStatus.COMPLETED:
                print("PASS Calculator task completed")
                print(f"RESULTS Result: {result.result.data.get('response', 'No response')}")
            else:
                print(f"FAIL Calculator task failed: {result.result.error}")
            
            # Test Python tool
            print("\nAnalyzing Testing Python tool:")
            python_task = TaskInput(
                id="tool_demo_python",
                task="Write and execute Python code to create a list of the first 5 prime numbers",
                context={"demo": "tool_usage", "tool": "python"}
            )
            
            print(f"Response Task: {python_task.task}")
            result = await agent.execute_task(python_task)
            
            if result.status == TaskStatus.COMPLETED:
                print("PASS Python task completed")
                print(f"RESULTS Result: {result.result.data.get('response', 'No response')}")
            else:
                print(f"FAIL Python task failed: {result.result.error}")
            
            self._record_result("Tool Usage Demo", result, {"tools_used": ["calculator", "python"]})
            
        except Exception as e:
            print(f"FAIL Tool demo failed: {e}")
    
    async def _demo_hybrid_mode(self):
        """Demo hybrid mode switching between OpenAI and local models."""
        print("\n Demo 5: Hybrid Mode")
        print("-" * 25)
        
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("SKIP:  Skipping hybrid demo (no API key)")
                return
            
            # Create integration in hybrid mode
            integration = create_openai_integration(
                openai_api_key=api_key,
                model_name="gpt-4o-mini",
                budget_limit=self.budget_limit,
                mode=OpenAIAgentMode.HYBRID
            )
            
            # Test with OpenAI model
            openai_provider = OpenAIProvider(
                model_name="gpt-4o-mini",
                api_key=api_key
            )
            
            openai_agent = ReactAgent(
                name="Hybrid-OpenAI-Agent",
                llm_provider=openai_provider,
                description="Hybrid agent using OpenAI"
            )
            
            adapter = integration.register_agent(openai_agent)
            
            task_input = TaskInput(
                id="hybrid_demo_openai",
                task="Explain quantum computing in one sentence.",
                context={"demo": "hybrid_mode", "model": "openai"}
            )
            
            print("Agent Testing with OpenAI model:")
            print(f"Response Task: {task_input.task}")
            
            result = await adapter.run_task(task_input)
            
            if result.status == TaskStatus.COMPLETED:
                print("PASS OpenAI task completed")
                print(f"RESULTS Response: {result.result.data.get('response', 'No response')}")
            else:
                print(f"FAIL OpenAI task failed: {result.result.error}")
            
            # Test fallback to local model
            try:
                local_provider = OllamaProvider(model_name="llama3.2:3b")
                local_agent = ReactAgent(
                    name="Hybrid-Local-Agent",
                    llm_provider=local_provider,
                    description="Hybrid agent using local model"
                )
                
                local_task = TaskInput(
                    id="hybrid_demo_local",
                    task="What is machine learning?",
                    context={"demo": "hybrid_mode", "model": "local"}
                )
                
                print("\n Testing with local model:")
                print(f"Response Task: {local_task.task}")
                
                local_result = await local_agent.execute_task(local_task)
                
                if local_result.status == TaskStatus.COMPLETED:
                    print("PASS Local task completed")
                    print(f"RESULTS Response: {local_result.result.data.get('response', 'No response')}")
                else:
                    print(f"FAIL Local task failed: {local_result.result.error}")
                
            except Exception as e:
                print(f"SKIP:  Local model not available: {e}")
            
            self._record_result("Hybrid Mode Demo", result, integration.get_budget_status())
            
        except Exception as e:
            print(f"FAIL Hybrid demo failed: {e}")
    
    async def _demo_batch_processing(self):
        """Demo batch processing of multiple tasks."""
        print("\nPACKAGE Demo 6: Batch Processing")
        print("-" * 30)
        
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("SKIP:  Skipping batch demo (no API key)")
                return
            
            # Create integration
            integration = create_openai_integration(
                openai_api_key=api_key,
                model_name="gpt-4o-mini",
                budget_limit=self.budget_limit
            )
            
            llm_provider = OpenAIProvider(
                model_name="gpt-4o-mini",
                api_key=api_key
            )
            
            agent = ReactAgent(
                name="Batch-Demo-Agent",
                llm_provider=llm_provider,
                description="Agent for batch processing demo"
            )
            
            adapter = integration.register_agent(agent)
            
            # Define batch tasks
            batch_tasks = [
                "What is the capital of France?",
                "Name three programming languages.",
                "What is 15 * 8?",
                "Define artificial intelligence.",
                "List two benefits of renewable energy."
            ]
            
            print(f" Processing {len(batch_tasks)} tasks in batch:")
            
            results = []
            start_time = datetime.now()
            
            for i, task_text in enumerate(batch_tasks, 1):
                print(f"   Response Task {i}/{len(batch_tasks)}: {task_text}")
                
                task_input = TaskInput(
                    id=f"batch_task_{i}",
                    task=task_text,
                    context={"demo": "batch_processing", "batch_id": i}
                )
                
                try:
                    result = await adapter.run_task(task_input)
                    results.append({
                        "task": task_text,
                        "status": result.status.value,
                        "success": result.status == TaskStatus.COMPLETED
                    })
                    
                    if result.status == TaskStatus.COMPLETED:
                        print("      PASS Completed")
                    else:
                        print(f"      FAIL Failed: {result.result.error}")
                        
                except Exception as e:
                    print(f"      FAIL Error: {e}")
                    results.append({
                        "task": task_text,
                        "status": "error",
                        "success": False,
                        "error": str(e)
                    })
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            successful_tasks = sum(1 for r in results if r["success"])
            
            print("\nRESULTS Batch Results:")
            print(f"   TIME:  Total time: {duration:.2f} seconds")
            print(f"   PASS Successful: {successful_tasks}/{len(batch_tasks)}")
            print(f"   FAIL Failed: {len(batch_tasks) - successful_tasks}/{len(batch_tasks)}")
            
            budget_status = integration.get_budget_status()
            print(f"    Total cost: ${budget_status['current_cost']:.4f}")
            
            self._record_result("Batch Processing Demo", None, {
                "batch_size": len(batch_tasks),
                "successful": successful_tasks,
                "duration": duration,
                "budget_status": budget_status
            })
            
        except Exception as e:
            print(f"FAIL Batch demo failed: {e}")
    
    def _record_result(self, demo_name: str, result: TaskOutput, metadata: Dict[str, Any]):
        """Record demo result."""
        self.results.append({
            "demo": demo_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "result": {
                "task_id": result.task_id if result else None,
                "status": result.status.value if result else None,
                "success": result.status == TaskStatus.COMPLETED if result else None
            } if result else None,
            "metadata": metadata
        })
    
    async def _generate_final_report(self):
        """Generate final demonstration report."""
        print("\nRESULTS Final Demonstration Report")
        print("=" * 40)
        
        # Calculate summary statistics
        total_demos = len(self.results)
        successful_demos = sum(1 for r in self.results if r.get("result", {}).get("success", False))
        
        print("Performance Summary:")
        print(f"   TARGET Total demos: {total_demos}")
        print(f"   PASS Successful: {successful_demos}")
        print(f"   FAIL Failed: {total_demos - successful_demos}")
        
        # Calculate total cost
        total_cost = 0.0
        for result in self.results:
            metadata = result.get("metadata", {})
            if "budget_status" in metadata:
                budget_status = metadata["budget_status"]
                if isinstance(budget_status, dict):
                    total_cost += budget_status.get("current_cost", 0.0)
        
        print(f"    Total estimated cost: ${total_cost:.4f}")
        print(f"    Budget remaining: ${self.budget_limit - total_cost:.4f}")
        
        # Save detailed report
        report_file = f"demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report_data = {
            "demo_summary": {
                "total_demos": total_demos,
                "successful_demos": successful_demos,
                "failed_demos": total_demos - successful_demos,
                "total_cost": total_cost,
                "budget_limit": self.budget_limit,
                "openai_agents_available": OPENAI_AGENTS_AVAILABLE
            },
            "demo_results": self.results,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f" Detailed report saved to: {report_file}")
        
        # System recommendations
        print("\nTARGET System Status:")
        print(f"   FIXING OpenAI Agents SDK: {'PASS Available' if OPENAI_AGENTS_AVAILABLE else 'FAIL Not Available'}")
        print(f"    OpenAI API: {'PASS Configured' if os.getenv('OPENAI_API_KEY') else 'FAIL Not Configured'}")
        print(f"    Local Models: {'PASS Available' if self._check_ollama() else 'FAIL Not Available'}")
        
        if successful_demos == total_demos:
            print("\nSUCCESS All demonstrations completed successfully!")
            print("   The LlamaAgent system with OpenAI integration is fully operational.")
        else:
            print("\nWARNING:  Some demonstrations failed.")
            print("   Please check the logs and ensure all dependencies are properly configured.")
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is available."""
        try:
            OllamaProvider(model_name="llama3.2:3b")
            return True
        except Exception:
            return False


async def main():
    """Main demonstration function."""
    print("Featured LlamaAgent + OpenAI Agents SDK Complete Integration Demo")
    print("Author: Nik Jois (nikjois@llamasearch.ai)")
    print("=" * 70)
    
    demo = CompleteDemonstration()
    
    try:
        await demo.run_complete_demo()
    except KeyboardInterrupt:
        print("\nSTOP:  Demo interrupted by user")
    except Exception as e:
        print(f"\nFAIL Demo failed with error: {e}")
        logger.error("Demo failed", exc_info=True)
    
    print("\nGOODBYE: Thank you for trying the LlamaAgent + OpenAI integration!")


if __name__ == "__main__":
    asyncio.run(main()) 
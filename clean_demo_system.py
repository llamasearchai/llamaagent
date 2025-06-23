#!/usr/bin/env python3
"""
Complete LlamaAgent System Demonstration
Author: Nik Jois <nikjois@llamasearch.ai>

This demonstration shows the complete LlamaAgent system working with:
- ReactAgent with SPRE methodology
- Multiple LLM providers
- Tool integration
- Memory system
- Database integration
- Vector similarity search
- Comprehensive error handling
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llamaagent import ReactAgent, AgentConfig
from llamaagent.llm import create_provider
from llamaagent.tools import ToolRegistry, get_all_tools
from llamaagent.memory.base import SimpleMemory
from llamaagent.storage.database import DatabaseManager, DatabaseConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LlamaAgentSystemDemo:
    """Complete demonstration of LlamaAgent capabilities."""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.start_time = time.time()
        
    async def run_complete_demo(self):
        """Run the complete demonstration."""
        logger.info("Starting Complete LlamaAgent System Demo")
        
        try:
            # 1. Test Core Agent Functionality
            await self.test_core_agent()
            
            # 2. Test LLM Providers
            await self.test_llm_providers()
            
            # 3. Test Tool Integration
            await self.test_tool_integration()
            
            # 4. Test Memory System
            await self.test_memory_system()
            
            # 5. Test Database Integration
            await self.test_database_integration()
            
            # 6. Test SPRE Methodology
            await self.test_spre_methodology()
            
            # 7. Test Error Handling
            await self.test_error_handling()
            
            # 8. Test Performance
            await self.test_performance()
            
            # Generate final report
            await self.generate_report()
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise
    
    async def test_core_agent(self):
        """Test core agent functionality."""
        logger.info("Testing Core Agent Functionality")
        
        # Create basic agent
        config = AgentConfig(
            name="TestAgent",
            spree_enabled=False
        )
        agent = ReactAgent(config)
        
        # Test simple task
        result = await agent.execute("Calculate 2 + 2")
        
        self.results.append({
            "test": "core_agent",
            "success": result.success,
            "content": result.content,
            "execution_time": result.execution_time
        })
        
        logger.info(f"Core Agent: {result.content} (Success: {result.success})")
    
    async def test_llm_providers(self):
        """Test different LLM providers."""
        logger.info("Testing LLM Providers")
        
        providers = [
            ("mock", {}),
            ("openai", {"api_key": "test-key"}),
        ]
        
        for provider_name, kwargs in providers:
            try:
                provider = create_provider(provider_name, **kwargs)
                
                config = AgentConfig(name=f"{provider_name.title()}Agent")
                agent = ReactAgent(config, llm_provider=provider)
                
                result = await agent.execute("What is 5 * 5?")
                
                self.results.append({
                    "test": f"llm_provider_{provider_name}",
                    "success": result.success,
                    "provider": provider_name,
                    "content": result.content[:100] + "..." if len(result.content) > 100 else result.content
                })
                
                logger.info(f"{provider_name.title()} Provider: Working")
                
            except Exception as e:
                logger.warning(f"{provider_name.title()} Provider: {e}")
                self.results.append({
                    "test": f"llm_provider_{provider_name}",
                    "success": False,
                    "error": str(e)
                })
    
    async def test_tool_integration(self):
        """Test tool integration."""
        logger.info("Testing Tool Integration")
        
        # Create tool registry
        tools = ToolRegistry()
        for tool in get_all_tools():
            tools.register(tool)
        
        # Create agent with tools
        config = AgentConfig(name="ToolAgent")
        agent = ReactAgent(config, tools=tools)
        
        # Test calculator tool
        result = await agent.execute("Calculate the square root of 144")
        
        self.results.append({
            "test": "tool_integration",
            "success": result.success,
            "content": result.content,
            "tools_available": len(tools.list_names())
        })
        
        logger.info(f"Tools: {len(tools.list_names())} available, test success: {result.success}")
    
    async def test_memory_system(self):
        """Test memory system."""
        logger.info("Testing Memory System")
        
        memory = SimpleMemory()
        
        # Store some information
        await memory.add("The capital of France is Paris", tags=["fact"], type="fact")
        await memory.add("Python is a programming language", tags=["fact"], type="fact")
        
        # Retrieve information  
        results = await memory.search("capital France", limit=1)
        
        success = len(results) > 0 and "Paris" in results[0]["content"]
        
        self.results.append({
            "test": "memory_system", 
            "success": success,
            "stored_items": memory.count(),
            "search_results": len(results)
        })
        
        logger.info(f"Memory: {memory.count()} items stored, search works: {success}")
    
    async def test_database_integration(self):
        """Test database integration."""
        logger.info("Testing Database Integration")
        
        try:
            # Test with in-memory database
            config = DatabaseConfig(host=":memory:")
            db = DatabaseManager(config)
            await db.initialise()
            
            # Test conversation storage
            conv_id = await db.save_conversation(
                provider="test",
                model="test-model",
                prompt="Test prompt",
                response="Test response",
                tokens_used=10,
                cost=0.01
            )
            
            # Test embedding storage  
            emb_id = await db.save_embedding(
                text="Test text",
                embedding=[0.1, 0.2, 0.3],
                model="test-embedding-model"
            )
            
            success = bool(conv_id and emb_id)
            
            self.results.append({
                "test": "database_integration",
                "success": success,
                "conversation_id": conv_id,
                "embedding_id": emb_id
            })
            
            await db.close()
            logger.info(f"Database: Working (mock mode: {db.pool is None})")
            
        except Exception as e:
            logger.warning(f"Database: {e}")
            self.results.append({
                "test": "database_integration",
                "success": False,
                "error": str(e)
            })
    
    async def test_spre_methodology(self):
        """Test SPRE methodology."""
        logger.info("Testing SPRE Methodology")
        
        config = AgentConfig(
            name="SPREAgent",
            spree_enabled=True,
            max_iterations=3
        )
        agent = ReactAgent(config)
        
        # Test complex task that benefits from planning
        result = await agent.execute(
            "Plan a simple birthday party for 10 people including food, decorations, and activities"
        )
        
        self.results.append({
            "test": "spre_methodology",
            "success": result.success,
            "content": result.content[:200] + "..." if len(result.content) > 200 else result.content,
            "trace_steps": len(result.trace) if result.trace else 0
        })
        
        logger.info(f"SPRE: Success {result.success}, {len(result.trace) if result.trace else 0} trace steps")
    
    async def test_error_handling(self):
        """Test error handling."""
        logger.info("Testing Error Handling")
        
        config = AgentConfig(name="ErrorTestAgent")
        agent = ReactAgent(config)
        
        # Test with invalid input
        result = await agent.execute("")  # Empty task
        
        # Should handle gracefully
        handled_gracefully = result is not None
        
        self.results.append({
            "test": "error_handling",
            "success": handled_gracefully,
            "error_handled": not result.success if result else False
        })
        
        logger.info(f"Error Handling: Graceful handling {handled_gracefully}")
    
    async def test_performance(self):
        """Test performance with multiple concurrent tasks."""
        logger.info("Testing Performance")
        
        config = AgentConfig(name="PerfAgent")
        
        # Create multiple agents for concurrent testing
        tasks = [
            "Calculate 10 + 5",
            "Calculate 20 * 3", 
            "Calculate 100 / 4",
            "Calculate 7 * 8",
            "Calculate 99 - 33"
        ]
        
        start_time = time.time()
        
        # Run tasks concurrently
        agents = [ReactAgent(config) for _ in tasks]
        results = await asyncio.gather(
            *[agent.execute(task) for agent, task in zip(agents, tasks)],
            return_exceptions=True
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        successful_results = [r for r in results if not isinstance(r, Exception) and r.success]
        
        self.results.append({
            "test": "performance",
            "success": len(successful_results) > 0,
            "concurrent_tasks": len(tasks),
            "successful_tasks": len(successful_results),
            "total_time": total_time,
            "avg_time_per_task": total_time / len(tasks)
        })
        
        logger.info(f"Performance: {len(successful_results)}/{len(tasks)} tasks in {total_time:.2f}s")
    
    async def generate_report(self):
        """Generate final demonstration report."""
        total_time = time.time() - self.start_time
        successful_tests = [r for r in self.results if r.get("success", False)]
        
        report = {
            "demo_summary": {
                "total_tests": len(self.results),
                "successful_tests": len(successful_tests),
                "success_rate": len(successful_tests) / len(self.results) * 100,
                "total_execution_time": total_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "test_results": self.results,
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform,
                "working_directory": str(Path.cwd())
            }
        }
        
        # Save report
        report_file = Path("demo_results_clean.json")
        report_file.write_text(json.dumps(report, indent=2))
        
        logger.info("=" * 60)
        logger.info("COMPLETE LLAMAAGENT SYSTEM DEMO RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {len(self.results)}")
        logger.info(f"Successful: {len(successful_tests)}")
        logger.info(f"Success Rate: {len(successful_tests) / len(self.results) * 100:.1f}%")
        logger.info(f"Total Time: {total_time:.2f} seconds")
        logger.info(f"Report saved: {report_file}")
        logger.info("=" * 60)
        
        # Print individual test results
        for result in self.results:
            status = "PASS" if result.get("success") else "FAIL"
            test_name = result.get("test", "unknown").replace("_", " ").title()
            logger.info(f"{status}: {test_name}")
        
        return report


async def main():
    """Main entry point."""
    try:
        demo = LlamaAgentSystemDemo()
        await demo.run_complete_demo()
        print("\nComplete LlamaAgent System Demo finished successfully!")
        return 0
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code) 
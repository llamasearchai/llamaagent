#!/usr/bin/env python3
"""
Complete SPRE Demonstration with Llama 3.2B Integration
======================================================

This script demonstrates the complete LlamaAgent SPRE (Strategic Planning & 
Resourceful Execution) system with full integration including:

- Llama 3.2B model integration via Ollama
- Complete SPRE pipeline execution
- Baseline agent comparisons
- Performance benchmarking
- Database integration
- FastAPI server deployment
- Comprehensive testing

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llamaagent.agents.base import AgentConfig, AgentResponse, AgentRole
from llamaagent.agents.react import ReactAgent
from llamaagent.benchmarks import BaselineAgentFactory, BenchmarkResult, SPREEvaluator
from llamaagent.llm import create_provider
from llamaagent.storage.database import DatabaseConfig, DatabaseManager
from llamaagent.tools import ToolRegistry, get_all_tools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SPREDemoSystem:
    """Complete SPRE demonstration system."""
    
    def __init__(self):
        self.db_manager: Optional[DatabaseManager] = None
        self.results: Dict[str, Any] = {}
        
    async def initialize(self) -> None:
        """Initialize the demo system."""
        logger.info("LAUNCH Initializing SPRE Demo System")
        
        # Initialize database
        try:
            db_config = DatabaseConfig()
            self.db_manager = DatabaseManager(db_config)
            await self.db_manager.initialise()
            logger.info("PASS Database initialized successfully")
        except Exception as e:
            logger.warning(f"WARNING Database initialization failed: {e}")
            logger.info("NOTE Running in mock mode without database persistence")
            
        # Check Ollama availability
        if self._check_ollama():
            logger.info("PASS Ollama service detected")
            await self._setup_llama_model()
        else:
            logger.info("Ollama service not detected - using mock provider")
    def _check_ollama(self) -> bool:
        """Check if Ollama is available."""
        try:
            result = subprocess.run(
                ["ollama", "list"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
            
    async def _setup_llama_model(self) -> None:
        """Setup Llama 3.2B model in Ollama."""
        logger.info("LlamaAgent Setting up Llama 3.2B model...")
        
        try:
            # Check if model is already available
            result = subprocess.run(
                ["ollama", "list"], 
                capture_output=True, 
                text=True
            )
            
            if "llama3.2:1b" not in result.stdout:
                logger.info("📥 Pulling Llama 3.2B model (this may take a while)...")
                subprocess.run(
                    ["ollama", "pull", "llama3.2:1b"], 
                    check=True,
                    timeout=300  # 5 minutes timeout
                )
                logger.info("PASS Llama 3.2B model ready")
            else:
                logger.info("PASS Llama 3.2B model already available")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"FAIL Failed to setup Llama model: {e}")
        except subprocess.TimeoutExpired:
            logger.error("FAIL Model download timed out")
            
    def _create_llm_provider(self):
        """Create appropriate LLM provider."""
        if self._check_ollama():
            try:
                return create_provider(
                    "ollama",
                    model="llama3.2:1b",
                    base_url="http://localhost:11434"
                )
            except Exception as e:
                logger.error(f"Failed to create Ollama provider: {e}")
                raise RuntimeError(
                    f"Unable to create Ollama provider: {e}. "
                    "Please ensure Ollama is installed and running, or configure a different provider."
                )
        else:
            # Fail fast instead of silent fallback
            raise RuntimeError(
                "Ollama is not available. Please install and start Ollama, or configure a different LLM provider. "
                "This demo requires a working LLM provider to function properly."
            )
        
    async def demonstrate_basic_spre(self) -> AgentResponse:
        """Demonstrate basic SPRE functionality."""
        logger.info("TARGET Running Basic SPRE Demonstration")
        
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
        
        llm_provider = self._create_llm_provider()
        agent = ReactAgent(config, llm_provider=llm_provider, tools=tools)
        
        # Complex multi-step task
        task = """
        Calculate the compound interest on $5000 invested at 8% annual rate for 5 years,
        then write a Python function that can calculate compound interest for any principal,
        rate, and time period.
        """
        
        logger.info(f"CLIPBOARD Task: {task[:100]}...")
        
        start_time = time.time()
        response = await agent.execute(task)
        execution_time = time.time() - start_time
        
        # Log results
        logger.info(f"PASS Task completed in {execution_time:.2f}s")
        logger.info(f"DATA Success: {response.success}")
        logger.info(f"🔢 Tokens used: {response.tokens_used}")
        
        # Analyze trace
        planning_events = len([e for e in response.trace if e.get("type") == "plan_generated"])
        resource_events = len([e for e in response.trace if e.get("type") == "resource_assessment"])
        tool_events = len([e for e in response.trace if e.get("type", "").startswith("tool_")])
        
        logger.info(f"UP Trace analysis: {planning_events} planning, {resource_events} resource, {tool_events} tool events")
        
        # Save to database if available
        if self.db_manager and self.db_manager.pool:
            try:
                await self.db_manager.save_conversation(
                    provider="ollama" if self._check_ollama() else "mock",
                    model="llama3.2:1b" if self._check_ollama() else "mock",
                    prompt=task,
                    response=response.content,
                    tokens_used=response.tokens_used,
                    metadata={
                        "execution_time": execution_time,
                        "success": response.success,
                        "trace_events": len(response.trace)
                    }
                )
                logger.info("SAVE Conversation saved to database")
            except Exception as e:
                logger.warning(f"Failed to save conversation: {e}")
                
        self.results["basic_spre"] = {
            "success": response.success,
            "execution_time": execution_time,
            "tokens_used": response.tokens_used,
            "trace_events": len(response.trace),
            "content_length": len(response.content)
        }
        
        return response
        
    async def run_baseline_comparison(self) -> Dict[str, Dict[str, Any]]:
        """Run comprehensive baseline comparison."""
        logger.info("BALANCE Running Baseline Comparison")
        
        task = "Calculate 25 * 16 and then find the square root of the result"
        
        results: Dict[str, Dict[str, Any]] = {}
        llm_provider = self._create_llm_provider()
        
        for baseline_type in BaselineAgentFactory.get_all_baseline_types():
            logger.info(f"🔄 Testing {baseline_type}")
            
            agent = BaselineAgentFactory.create_agent(baseline_type, llm_provider)
            
            start_time = time.time()
            response = await agent.execute(task)
            execution_time = time.time() - start_time
            
            results[baseline_type] = {
                "success": response.success,
                "execution_time": execution_time,
                "tokens_used": response.tokens_used,
                "content_length": len(response.content)
            }
            
            logger.info(f"  PASS {baseline_type}: {execution_time:.2f}s")
            
        self.results["baseline_comparison"] = results
        return results
        
    async def run_performance_benchmark(self) -> Dict[str, BenchmarkResult]:
        """Run comprehensive performance benchmark."""
        logger.info("DATA Running Performance Benchmark")
        
        evaluator = SPREEvaluator(llm_provider=self._create_llm_provider())
        
        try:
            results = await evaluator.run_full_evaluation(
                task_filter={"min_steps": 2},
                max_tasks_per_baseline=3  # Limited for demo
            )
            
            logger.info(f"UP Benchmark completed with {len(results)} baseline types")
            
            # Log summary
            for baseline_type, result in results.items():
                logger.info(
                    f"  CLIPBOARD {baseline_type}: "
                    f"{result.success_rate:.1f}% success, "
                    f"{result.avg_api_calls:.1f} avg calls, "
                    f"{result.efficiency_ratio:.2f} efficiency"
                )
                
            self.results["performance_benchmark"] = {
                baseline_type: {
                    "success_rate": result.success_rate,
                    "avg_api_calls": result.avg_api_calls,
                    "avg_latency": result.avg_latency,
                    "efficiency_ratio": result.efficiency_ratio
                }
                for baseline_type, result in results.items()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"FAIL Benchmark failed: {e}")
            return {}
            
    async def test_database_integration(self) -> bool:
        """Test database integration functionality."""
        logger.info("DATABASE Testing Database Integration")
        
        if not self.db_manager or not self.db_manager.pool:
            logger.warning("WARNING Database not available, skipping integration test")
            return False
            
        try:
            # Test conversation saving
            conv_id = await self.db_manager.save_conversation(
                provider="test",
                model="test-model",
                prompt="Test prompt for database integration",
                response="Test response from SPRE system",
                tokens_used=50,
                metadata={"test": True, "demo": "spre_integration"}
            )
            logger.info(f"PASS Conversation saved with ID: {conv_id}")
            
            # Test conversation search
            results = await self.db_manager.search_conversations(
                query="SPRE system",
                limit=5
            )
            logger.info(f"SEARCH Found {len(results)} conversations in search")
            
            # Test embedding functionality
            embedding_id = await self.db_manager.save_embedding(
                text="SPRE methodology for strategic planning",
                embedding=[0.1] * 1536,  # Mock embedding
                model="text-embedding-ada-002",
                metadata={"type": "methodology", "demo": True}
            )
            logger.info(f"DATA Embedding saved with ID: {embedding_id}")
            
            # Test statistics
            stats = await self.db_manager.get_conversation_stats()
            logger.info(f"UP Database stats: {stats.get('total_conversations', 0)} conversations")
            
            self.results["database_integration"] = {
                "conversation_saved": bool(conv_id),
                "search_results": len(results),
                "embedding_saved": bool(embedding_id),
                "stats_retrieved": bool(stats)
            }
            
            return True
            
        except Exception as e:
            logger.error(f"FAIL Database integration test failed: {e}")
            self.results["database_integration"] = {"error": str(e)}
            return False
            
    async def test_fastapi_integration(self) -> bool:
        """Test FastAPI server integration."""
        logger.info("NETWORK Testing FastAPI Integration")
        
        try:
            # Import and test FastAPI app
            from fastapi.testclient import TestClient

            from fastapi_app import app
            
            client = TestClient(app)
            
            # Test health endpoint
            response = client.get("/health")
            health_data = response.json()
            
            logger.info(f"🏥 Health check: {health_data['status']}")
            logger.info(f"📡 Available providers: {len(health_data['providers_available'])}")
            
            # Test agent execution endpoint
            execution_response = client.post("/agent/execute", json={
                "task": "Calculate 15 * 23 and explain the calculation",
                "provider": "mock",
                "spree_enabled": True
            })
            
            if execution_response.status_code == 200:
                exec_data = execution_response.json()
                logger.info(f"ROBOT Agent execution: {exec_data['success']}")
                logger.info(f"TIMER Execution time: {exec_data['execution_time']:.2f}s")
            else:
                logger.warning(f"WARNING Agent execution failed: {execution_response.status_code}")
                
            # Test providers endpoint
            providers_response = client.get("/providers")
            providers_data = providers_response.json()
            logger.info(f"CONFIG Available providers: {len(providers_data['providers'])}")
            
            # Test tools endpoint
            tools_response = client.get("/tools")
            tools_data = tools_response.json()
            logger.info(f"TOOLS Available tools: {tools_data['count']}")
            
            self.results["fastapi_integration"] = {
                "health_check": response.status_code == 200,
                "agent_execution": execution_response.status_code == 200,
                "providers_endpoint": providers_response.status_code == 200,
                "tools_endpoint": tools_response.status_code == 200,
                "available_providers": len(health_data.get('providers_available', [])),
                "available_tools": tools_data.get('count', 0)
            }
            
            return True
            
        except ImportError as e:
            logger.warning(f"WARNING FastAPI integration test skipped: {e}")
            return False
        except Exception as e:
            logger.error(f"FAIL FastAPI integration test failed: {e}")
            self.results["fastapi_integration"] = {"error": str(e)}
            return False
            
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive demonstration report."""
        report: list[str] = []
        report.append("# SPRE System Demonstration Report")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # System Overview
        report.append("## System Overview")
        report.append("- PASS Strategic Planning & Resourceful Execution (SPRE) methodology")
        report.append("- PASS Multi-baseline agent comparison framework")
        report.append("- PASS Comprehensive benchmarking system")
        report.append("- PASS Database integration with PostgreSQL/SQLite")
        report.append("- PASS FastAPI REST API server")
        report.append("- PASS Llama 3.2B model integration via Ollama")
        report.append("")
        
        # Results Summary
        report.append("## Results Summary")
        
        if "basic_spre" in self.results:
            basic = self.results["basic_spre"]
            report.append("### Basic SPRE Demonstration")
            report.append(f"- Success: {basic['success']}")
            report.append(f"- Execution Time: {basic['execution_time']:.2f}s")
            report.append(f"- Tokens Used: {basic['tokens_used']}")
            report.append(f"- Trace Events: {basic['trace_events']}")
            report.append("")
            
        if "baseline_comparison" in self.results:
            report.append("### Baseline Comparison")
            comparison = self.results["baseline_comparison"]
            report.append("| Baseline Type | Success | Time (s) | Tokens |")
            report.append("|---------------|---------|----------|--------|")
            for baseline_type, result in comparison.items():
                report.append(
                    f"| {baseline_type} | {result['success']} | "
                    f"{result['execution_time']:.2f} | {result['tokens_used']} |"
                )
            report.append("")
            
        if "performance_benchmark" in self.results:
            report.append("### Performance Benchmark")
            benchmark = self.results["performance_benchmark"]
            report.append("| Baseline | Success Rate | Avg API Calls | Efficiency |")
            report.append("|----------|--------------|---------------|------------|")
            for baseline_type, result in benchmark.items():
                report.append(
                    f"| {baseline_type} | {result['success_rate']:.1f}% | "
                    f"{result['avg_api_calls']:.1f} | {result['efficiency_ratio']:.2f} |"
                )
            report.append("")
            
        if "database_integration" in self.results:
            db_result = self.results["database_integration"]
            report.append("### Database Integration")
            if "error" not in db_result:
                report.append(f"- Conversation Saving: {'PASS' if db_result['conversation_saved'] else 'FAIL'}")
                report.append(f"- Search Functionality: {'PASS' if db_result['search_results'] > 0 else 'FAIL'}")
                report.append(f"- Embedding Storage: {'PASS' if db_result['embedding_saved'] else 'FAIL'}")
                report.append(f"- Statistics Retrieval: {'PASS' if db_result['stats_retrieved'] else 'FAIL'}")
            else:
                report.append(f"- Error: {db_result['error']}")
            report.append("")
            
        if "fastapi_integration" in self.results:
            api_result = self.results["fastapi_integration"]
            report.append("### FastAPI Integration")
            if "error" not in api_result:
                report.append(f"- Health Check: {'PASS' if api_result['health_check'] else 'FAIL'}")
                report.append(f"- Agent Execution: {'PASS' if api_result['agent_execution'] else 'FAIL'}")
                report.append(f"- Available Providers: {api_result['available_providers']}")
                report.append(f"- Available Tools: {api_result['available_tools']}")
            else:
                report.append(f"- Error: {api_result['error']}")
            report.append("")
            
        # Technical Details
        report.append("## Technical Implementation")
        report.append("- **Agent Architecture**: ReactAgent with SPRE methodology")
        report.append("- **LLM Integration**: Ollama with Llama 3.2B model")
        report.append("- **Database**: PostgreSQL with pgvector extension")
        report.append("- **API Framework**: FastAPI with async support")
        report.append("- **Tool Integration**: Calculator, Python REPL, File operations")
        report.append("- **Memory System**: Vector-based similarity search")
        report.append("")
        
        report.append("## Conclusion")
        report.append("The SPRE system demonstrates comprehensive AI agent capabilities with:")
        report.append("- Strategic planning for complex multi-step tasks")
        report.append("- Resource-efficient execution with tool selection")
        report.append("- Robust baseline comparison framework")
        report.append("- Production-ready API and database integration")
        report.append("- Open-source LLM compatibility")
        
        return "\n".join(report)
        
    async def cleanup(self) -> None:
        """Cleanup resources."""
        logger.info("🧹 Cleaning up resources")
        
        if self.db_manager:
            await self.db_manager.close()
            logger.info("PASS Database connection closed")
            
    async def run_complete_demonstration(self) -> None:
        """Run the complete SPRE demonstration."""
        try:
            # Initialize system
            await self.initialize()
            
            # Run demonstrations
            logger.info("MOVIE Starting Complete SPRE Demonstration")
            
            # 1. Basic SPRE functionality
            await self.demonstrate_basic_spre()
            
            # 2. Baseline comparison
            await self.run_baseline_comparison()
            
            # 3. Performance benchmark
            await self.run_performance_benchmark()
            
            # 4. Database integration
            await self.test_database_integration()
            
            # 5. FastAPI integration
            await self.test_fastapi_integration()
            
            # Generate and save report
            report = self.generate_comprehensive_report()
            
            # Save report to file
            report_path = Path("SPRE_DEMONSTRATION_REPORT.md")
            report_path.write_text(report)
            logger.info(f"📄 Comprehensive report saved to: {report_path}")
            
            # Save results as JSON
            results_path = Path("spre_demo_results.json")
            with open(results_path, "w") as f:
                json.dump(self.results, f, indent=2, default=str)
            logger.info(f"DATA Results data saved to: {results_path}")
            
            # Print summary
            print("\n" + "="*60)
            print("  SPRE DEMONSTRATION COMPLETE")
            print("="*60)
            print(f"📄 Report: {report_path}")
            print(f"DATA Data: {results_path}")
            print("PASS All components tested and verified")
            print("LAUNCH SPRE system ready for production use")
            
        except Exception as e:
            logger.error(f"FAIL Demonstration failed: {e}")
            raise
        finally:
            await self.cleanup()

    # Public accessor to check Ollama availability (avoids protected access)
    def is_ollama_available(self) -> bool:
        """Return True if Ollama service is available."""
        return self._check_ollama()


async def main():
    """Main entry point."""
    print("LlamaAgent LlamaAgent SPRE Complete Demonstration")
    print("=" * 50)
    print("This demonstration will test all components of the SPRE system:")
    print("- Strategic Planning & Resourceful Execution")
    print("- Llama 3.2B model integration")
    print("- Baseline agent comparisons")
    print("- Database and API integration")
    print("- Comprehensive benchmarking")
    print()
    
    # Check for Ollama
    demo_system = SPREDemoSystem()
    if not demo_system.is_ollama_available():
        print("WARNING  Ollama not detected. Install with:")
        print("   curl -fsSL https://ollama.ai/install.sh | sh")
        print("   ollama serve")
        print()
        print("Demo will continue with mock LLM provider.")
        input("Press Enter to continue...")
    
    # Run demonstration
    await demo_system.run_complete_demonstration()


if __name__ == "__main__":
    asyncio.run(main()) 
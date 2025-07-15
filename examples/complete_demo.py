#!/usr/bin/env python3
"""
Complete demonstration of LlamaAgent framework capabilities.
Shows all major features with working examples.
"""

import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Core imports
from src.llamaagent import ReactAgent
from src.llamaagent.llm import MockProvider
from src.llamaagent.tools import Tool, ToolRegistry
from src.llamaagent.tools.calculator import CalculatorTool
from src.llamaagent.tools.python_repl import PythonREPLTool
from src.llamaagent.memory import BaseMemory
from src.llamaagent.types import AgentResponse

# Data generation
from src.llamaagent.data_generation import SPREGenerator, GDTGenerator

# Benchmarking
from src.llamaagent.benchmarks import SPREEvaluator, GAIABenchmark

# Visualization
from src.llamaagent.visualization import (
    create_agent_performance_plot,
    create_benchmark_comparison_plot
)

# Storage and caching
from src.llamaagent.storage import DatabaseManager, VectorMemory
from src.llamaagent.cache import CacheManager

# API
from src.llamaagent.api import create_app


class DemoRunner:
    """Main demo runner class."""
    
    def __init__(self):
        self.results = []
        self.agents = {}
        
    async def setup(self):
        """Set up demo components."""
        print("=== LlamaAgent Framework Demo ===\n")
        print("Setting up components...")
        
        # Initialize cache
        self.cache = CacheManager()
        
        # Initialize storage (in-memory for demo)
        self.db = DatabaseManager("sqlite:///:memory:")
        await self.db.initialize()
        
        # Initialize vector memory
        self.vector_memory = VectorMemory()
        
        print("✓ Storage and cache initialized")
        
    async def demo_basic_agent(self):
        """Demonstrate basic agent functionality."""
        print("\n1. Basic Agent Demo")
        print("-" * 50)
        
        # Create agent with mock provider
        agent = ReactAgent(
            name="DemoAgent",
            provider=MockProvider(),
            tools=[CalculatorTool(), PythonREPLTool()],
            enable_memory=True
        )
        
        # Test queries
        queries = [
            "What is 25 * 4?",
            "Generate a list of the first 10 fibonacci numbers",
            "What is the capital of France?"
        ]
        
        for query in queries:
            print(f"\nQuery: {query}")
            response = await agent.run(query)
            print(f"Response: {response.response}")
            print(f"Success: {response.success}")
            
            self.results.append({
                "agent": "DemoAgent",
                "query": query,
                "response": response.response,
                "success": response.success,
                "timestamp": datetime.now().isoformat()
            })
        
        # Show memory
        print(f"\nAgent memory entries: {len(agent.memory.entries)}")
        
        self.agents["demo"] = agent
        
    async def demo_custom_tools(self):
        """Demonstrate custom tool creation."""
        print("\n2. Custom Tools Demo")
        print("-" * 50)
        
        # Create custom tool
        @Tool(
            name="weather",
            description="Get weather information for a city"
        )
        async def weather_tool(city: str) -> str:
            # Mock weather data
            weather_data = {
                "New York": "Sunny, 72°F",
                "London": "Cloudy, 59°F",
                "Tokyo": "Rainy, 68°F",
                "Paris": "Partly cloudy, 64°F"
            }
            return weather_data.get(city, f"No weather data for {city}")
        
        @Tool(
            name="stock_price",
            description="Get stock price for a symbol"
        )
        async def stock_price_tool(symbol: str) -> str:
            # Mock stock data
            stock_data = {
                "AAPL": "$150.25",
                "GOOGL": "$2,750.50",
                "MSFT": "$305.75",
                "AMZN": "$3,300.00"
            }
            return stock_data.get(symbol, f"No data for {symbol}")
        
        # Create agent with custom tools
        agent = ReactAgent(
            name="CustomToolAgent",
            provider=MockProvider(),
            tools=[weather_tool, stock_price_tool]
        )
        
        # Test custom tools
        queries = [
            "What's the weather in New York?",
            "What's the current price of AAPL stock?",
            "Compare weather in London and Tokyo"
        ]
        
        for query in queries:
            print(f"\nQuery: {query}")
            response = await agent.run(query)
            print(f"Response: {response.response}")
        
        self.agents["custom"] = agent
        
    async def demo_data_generation(self):
        """Demonstrate data generation capabilities."""
        print("\n3. Data Generation Demo")
        print("-" * 50)
        
        # SPRE Generator
        print("\nGenerating SPRE data...")
        spre_gen = SPREGenerator()
        
        # Generate sample tasks
        topics = ["mathematics", "science", "history"]
        spre_data = []
        
        for topic in topics:
            task = await spre_gen.generate_task(topic)
            spre_data.append(task)
            print(f"\nTopic: {topic}")
            print(f"Question: {task['question']}")
            print(f"Answer: {task['answer']}")
            print(f"Steps: {len(task['reasoning_steps'])}")
        
        # GDT Generator
        print("\n\nGenerating debate data...")
        gdt_gen = GDTGenerator()
        
        debate_topic = await gdt_gen.generate_topic()
        print(f"Debate topic: {debate_topic}")
        
        # Generate mini debate
        debate = await gdt_gen.generate_debate(
            debate_topic,
            num_rounds=2
        )
        
        print(f"Positions: {debate['positions']}")
        print(f"Arguments generated: {len(debate['arguments'])}")
        
        self.results.append({
            "type": "data_generation",
            "spre_tasks": len(spre_data),
            "debate_topic": debate_topic
        })
        
    async def demo_benchmarking(self):
        """Demonstrate benchmarking capabilities."""
        print("\n4. Benchmarking Demo")
        print("-" * 50)
        
        # Create test agent
        agent = self.agents.get("demo", ReactAgent(
            name="BenchmarkAgent",
            provider=MockProvider()
        ))
        
        # SPRE Evaluation
        print("\nRunning SPRE evaluation...")
        spre_eval = SPREEvaluator()
        
        # Create sample tasks
        test_tasks = [
            {
                "question": "What is 10 + 15?",
                "answer": "25",
                "reasoning_steps": ["Add 10 and 15 to get 25"]
            },
            {
                "question": "What is the capital of Japan?",
                "answer": "Tokyo",
                "reasoning_steps": ["Japan's capital is Tokyo"]
            }
        ]
        
        results = []
        for task in test_tasks:
            result = await spre_eval.evaluate_task(agent, task)
            results.append(result)
            print(f"Task: {task['question']}")
            print(f"Correct: {result.get('correct', False)}")
        
        # Calculate metrics
        correct = sum(1 for r in results if r.get('correct', False))
        accuracy = correct / len(results) if results else 0
        
        print(f"\nSPRE Accuracy: {accuracy:.2%}")
        
        self.results.append({
            "type": "benchmark",
            "spre_accuracy": accuracy,
            "tasks_evaluated": len(results)
        })
        
    async def demo_caching(self):
        """Demonstrate caching functionality."""
        print("\n5. Caching Demo")
        print("-" * 50)
        
        # Cache some results
        print("Caching agent responses...")
        
        for i in range(3):
            key = f"demo_response_{i}"
            value = f"Cached response {i}"
            await self.cache.set(key, value, ttl=300)
            print(f"Cached: {key}")
        
        # Retrieve cached values
        print("\nRetrieving cached values...")
        for i in range(3):
            key = f"demo_response_{i}"
            value = await self.cache.get(key)
            print(f"{key}: {value}")
        
        # Test cache miss
        missing = await self.cache.get("non_existent_key")
        print(f"\nCache miss returns: {missing}")
        
    async def demo_vector_memory(self):
        """Demonstrate vector memory functionality."""
        print("\n6. Vector Memory Demo")
        print("-" * 50)
        
        print("Adding documents to vector memory...")
        
        # Add sample documents
        documents = [
            ("doc1", "Python is a programming language", [0.1, 0.2, 0.3]),
            ("doc2", "Machine learning uses algorithms", [0.4, 0.5, 0.6]),
            ("doc3", "Python is great for machine learning", [0.25, 0.35, 0.45])
        ]
        
        for doc_id, text, embedding in documents:
            await self.vector_memory.add_embedding(
                doc_id, embedding, {"text": text}
            )
            print(f"Added: {doc_id} - {text}")
        
        # Search similar documents
        print("\nSearching for similar documents...")
        query_embedding = [0.2, 0.3, 0.4]
        
        results = await self.vector_memory.search_similar(
            query_embedding, k=2
        )
        
        print(f"Top {len(results)} similar documents:")
        for result in results:
            print(f"- {result['id']}: {result['metadata']['text']}")
            print(f"  Similarity: {result['similarity']:.3f}")
        
    async def demo_persistence(self):
        """Demonstrate data persistence."""
        print("\n7. Data Persistence Demo")
        print("-" * 50)
        
        # Create user
        print("Creating user in database...")
        user_id = await self.db.create_user(
            username="demo_user",
            email="demo@example.com",
            metadata={"created_by": "demo"}
        )
        print(f"Created user with ID: {user_id}")
        
        # Create session
        print("\nCreating chat session...")
        session_id = await self.db.create_session(
            user_id=user_id,
            metadata={
                "model": "mock",
                "purpose": "demonstration"
            }
        )
        print(f"Created session with ID: {session_id}")
        
        # Log messages
        print("\nLogging chat messages...")
        messages = [
            ("user", "Hello, how are you?"),
            ("assistant", "I'm doing well, thank you!"),
            ("user", "What can you help me with?"),
            ("assistant", "I can help with various tasks!")
        ]
        
        for role, content in messages:
            msg_id = await self.db.log_message(
                session_id=session_id,
                role=role,
                content=content,
                metadata={"demo": True}
            )
            print(f"Logged message {msg_id}: [{role}] {content[:30]}...")
        
        # Retrieve session
        print("\nRetrieving session history...")
        history = await self.db.get_session_messages(session_id)
        print(f"Retrieved {len(history)} messages")
        
    def generate_report(self):
        """Generate final demo report."""
        print("\n8. Demo Summary Report")
        print("=" * 50)
        
        report = {
            "demo_completed": datetime.now().isoformat(),
            "components_tested": [
                "Basic Agent",
                "Custom Tools",
                "Data Generation",
                "Benchmarking",
                "Caching",
                "Vector Memory",
                "Persistence"
            ],
            "agents_created": len(self.agents),
            "results_collected": len(self.results),
            "status": "SUCCESS"
        }
        
        print(json.dumps(report, indent=2))
        
        # Save results
        with open("demo_results.json", "w") as f:
            json.dump({
                "report": report,
                "results": self.results
            }, f, indent=2)
        
        print("\nResults saved to demo_results.json")
        
        # Create visualizations
        print("\nGenerating visualizations...")
        
        # Agent performance plot
        if self.results:
            agent_results = [r for r in self.results if "agent" in r]
            if agent_results:
                create_agent_performance_plot(agent_results)
                print("✓ Agent performance plot created")
        
        return report
        
    async def run_demo(self):
        """Run complete demo."""
        try:
            await self.setup()
            
            # Run all demos
            await self.demo_basic_agent()
            await self.demo_custom_tools()
            await self.demo_data_generation()
            await self.demo_benchmarking()
            await self.demo_caching()
            await self.demo_vector_memory()
            await self.demo_persistence()
            
            # Generate report
            report = self.generate_report()
            
            print("\n✓ Demo completed successfully!")
            print("\nKey Features Demonstrated:")
            print("- Multi-provider LLM support with MockProvider")
            print("- Tool creation and execution")
            print("- Memory and context management")
            print("- Data generation (SPRE and GDT)")
            print("- Benchmarking and evaluation")
            print("- Caching for performance")
            print("- Vector memory for semantic search")
            print("- Database persistence")
            print("- Comprehensive error handling")
            
        except Exception as e:
            print(f"\nFAIL Demo error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup
            if hasattr(self, 'db'):
                await self.db.close()
            print("\nDemo cleanup completed.")


async def main():
    """Main entry point."""
    demo = DemoRunner()
    await demo.run_demo()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
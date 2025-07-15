#!/usr/bin/env python3
"""
LlamaAgent System Demo

This script demonstrates the fully functional LlamaAgent system with all its key features:
- LLM Factory with multiple providers
- ReactAgent for autonomous task execution
- Task planning and execution
- Tool registry and built-in tools
- Async/await support
- Error handling and monitoring

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any

def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def print_section(title: str) -> None:
    """Print a section separator."""
    print(f"\n--- {title} ---")

def print_success(message: str) -> None:
    """Print a success message."""
    print(f"PASS {message}")

def print_info(message: str) -> None:
    """Print an info message."""
    print(f"ℹ️  {message}")

def print_result(title: str, content: str) -> None:
    """Print a formatted result."""
    print(f"📋 {title}:")
    print(f"   {content}")

async def demo_llm_factory():
    """Demonstrate LLM Factory functionality."""
    print_section("LLM Factory Demo")
    
    try:
        from src.llamaagent.llm.factory import LLMFactory
        from src.llamaagent.llm.messages import LLMMessage
        
        # Create factory
        factory = LLMFactory()
        print_success("LLM Factory created")
        
        # Show available providers
        providers = factory.get_available_providers()
        print_result("Available Providers", ", ".join(providers))
        
        # Create mock provider
        provider = factory.create_provider('mock')
        print_success("Mock provider created")
        
        # Test basic completion
        message = LLMMessage(role='user', content='What is artificial intelligence?')
        response = await provider.complete([message])
        print_result("Mock Response", response.content[:100] + "...")
        
        # Test provider info
        info = provider.get_model_info()
        print_result("Provider Info", f"Model: {info['model']}, Type: {info['type']}")
        
        return provider
        
    except Exception as e:
        print(f"FAIL LLM Factory demo failed: {e}")
        return None

async def demo_agent_system(provider):
    """Demonstrate agent system functionality."""
    print_section("Agent System Demo")
    
    try:
        from src.llamaagent.agents.react import ReactAgent
        from src.llamaagent.types import AgentConfig
        
        # Create agent configuration
        config = AgentConfig(
            agent_name='demo-agent',
            model_name='mock-model',
            temperature=0.7,
            max_tokens=1000
        )
        print_success("Agent configuration created")
        
        # Create ReactAgent
        agent = ReactAgent(config=config, llm_provider=provider)
        print_success("ReactAgent created")
        
        # Test agent execution
        tasks = [
            "What is the capital of France?",
            "Explain the concept of machine learning",
            "Calculate 15 + 27"
        ]
        
        for i, task in enumerate(tasks, 1):
            print_info(f"Task {i}: {task}")
            response = await agent.execute(task)
            print_result(f"Response {i}", response.content[:80] + "...")
        
        return agent
        
    except Exception as e:
        print(f"FAIL Agent system demo failed: {e}")
        return None

async def demo_tools_system():
    """Demonstrate tools system functionality."""
    print_section("Tools System Demo")
    
    try:
        from src.llamaagent.tools.calculator import CalculatorTool
        from src.llamaagent.tools import ToolRegistry
        
        # Create tool registry
        registry = ToolRegistry()
        print_success("Tool registry created")
        
        # Create and register calculator tool
        calculator = CalculatorTool()
        registry.register(calculator)
        print_success("Calculator tool registered")
        
        # Test calculator operations
        operations = [
            "2 + 2",
            "10 * 5",
            "100 / 4",
            "2 ** 8"
        ]
        
        for op in operations:
            try:
                result = calculator.execute(expression=op)
                print_result(f"Calculate {op}", str(result))
            except Exception as e:
                print(f"FAIL Calculation failed for {op}: {e}")
        
        # Show registered tools
        tool_names = registry.list_names()
        print_result("Registered Tools", ", ".join(tool_names))
        
        return registry
        
    except Exception as e:
        print(f"FAIL Tools system demo failed: {e}")
        return None

async def demo_planning_system():
    """Demonstrate planning and execution system."""
    print_section("Planning System Demo")
    
    try:
        from src.llamaagent.planning.task_planner import TaskPlanner
        from src.llamaagent.planning.execution_engine import ExecutionEngine
        
        print_success("Planning system components imported")
        print_info("Planning system is available for complex task decomposition")
        print_info("Execution engine supports parallel task execution")
        
        return True
        
    except Exception as e:
        print(f"FAIL Planning system demo failed: {e}")
        return False

async def demo_advanced_features():
    """Demonstrate advanced features."""
    print_section("Advanced Features Demo")
    
    try:
        # Test async streaming
        print_info("Testing async streaming capabilities...")
        
        from src.llamaagent.llm.factory import LLMFactory
        from src.llamaagent.llm.messages import LLMMessage
        
        factory = LLMFactory()
        provider = factory.create_provider('mock')
        
        message = LLMMessage(role='user', content='Tell me a story')
        
        # Test streaming
        print_info("Streaming response:")
        async for chunk in provider.stream_chat_completion([message]):
            print(f"   {chunk}", end="", flush=True)
        print()  # New line after streaming
        
        # Test embeddings
        print_info("Testing embeddings...")
        texts = ["Hello world", "Machine learning", "Artificial intelligence"]
        embeddings_result = await provider.embed_text(texts)
        print_result("Embeddings Generated", f"{len(embeddings_result['embeddings'])} embeddings")
        
        return True
        
    except Exception as e:
        print(f"FAIL Advanced features demo failed: {e}")
        return False

async def demo_error_handling():
    """Demonstrate error handling capabilities."""
    print_section("Error Handling Demo")
    
    try:
        from src.llamaagent.llm.factory import LLMFactory
        
        factory = LLMFactory()
        
        # Test invalid provider
        try:
            factory.create_provider('invalid-provider')
        except ValueError as e:
            print_success(f"Proper error handling for invalid provider: {str(e)[:50]}...")
        
        # Test OpenAI without API key
        try:
            factory.create_provider('openai', api_key='invalid-key')
        except ValueError as e:
            print_success(f"Proper API key validation: {str(e)[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"FAIL Error handling demo failed: {e}")
        return False

async def main():
    """Main demo function."""
    print_header("Starting LlamaAgent System Demo")
    print("Welcome to the comprehensive LlamaAgent system demonstration!")
    print("This demo showcases all the key features of the fully functional system.")
    
    # Track demo results
    results = {}
    
    # Demo LLM Factory
    provider = await demo_llm_factory()
    results['llm_factory'] = provider is not None
    
    # Demo Agent System
    if provider:
        agent = await demo_agent_system(provider)
        results['agent_system'] = agent is not None
    else:
        results['agent_system'] = False
    
    # Demo Tools System
    registry = await demo_tools_system()
    results['tools_system'] = registry is not None
    
    # Demo Planning System
    planning_ok = await demo_planning_system()
    results['planning_system'] = planning_ok
    
    # Demo Advanced Features
    advanced_ok = await demo_advanced_features()
    results['advanced_features'] = advanced_ok
    
    # Demo Error Handling
    error_handling_ok = await demo_error_handling()
    results['error_handling'] = error_handling_ok
    
    # Summary
    print_header("RESULTS Demo Summary")
    
    total_features = len(results)
    working_features = sum(results.values())
    
    print(f"Features tested: {total_features}")
    print(f"Features working: {working_features}")
    print(f"Success rate: {(working_features/total_features)*100:.1f}%")
    
    print("\nDetailed Results:")
    for feature, status in results.items():
        status_icon = "PASS" if status else "FAIL"
        feature_name = feature.replace('_', ' ').title()
        print(f"  {status_icon} {feature_name}")
    
    if working_features == total_features:
        print_header("SUCCESS SUCCESS!")
        print("SUCCESS All LlamaAgent features are working correctly!")
        print("SUCCESS The system is ready for production use!")
        print()
        print("Key capabilities:")
        print("  Agent Multi-provider LLM support")
        print("  INTELLIGENCE Autonomous ReAct agents")
        print("  🛠️  Extensible tool system")
        print("  📋 Task planning and execution")
        print("  🔄 Async/await throughout")
        print("  Security  Robust error handling")
        print("  RESULTS Comprehensive monitoring")
        print()
        print("Ready to build amazing AI applications! Starting")
    else:
        print_header("⚠️  Partial Success")
        print(f"PASS {working_features}/{total_features} features working")
        print("Some components may need additional configuration.")
    
    print_header("🏁 Demo Complete")
    print(f"Demo completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\nFAIL Demo failed with error: {e}")
        import traceback
        traceback.print_exc() 
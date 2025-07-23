#!/usr/bin/env python3
"""
Test Basic Functionality - Verify the core modules are working
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_basic_imports():
    """Test that basic imports work."""
    print("Analyzing Testing Basic Imports...")
    
    try:
        # Test CLI imports
        from src.llamaagent.cli.main import app
        print("PASS CLI main imported successfully")
        
        # Test agent imports
        from src.llamaagent.agents.base import BaseAgent
        from src.llamaagent.agents.react import ReactAgent
        print("PASS Agent modules imported successfully")
        
        # Test tool imports
        from src.llamaagent.tools.base import BaseTool
        from src.llamaagent.tools.calculator import CalculatorTool
        from src.llamaagent.tools.python_repl import PythonREPLTool
        print("PASS Tool modules imported successfully")
        
        # Test LLM imports
        from src.llamaagent.llm.factory import LLMFactory
        from src.llamaagent.llm.messages import LLMMessage, LLMResponse
        print("PASS LLM modules imported successfully")
        
        # Test data generation imports
        from src.llamaagent.data_generation.base import BaseDataGenerator
        from src.llamaagent.data_generation.gdt import GDTDataGenerator
        print("PASS Data generation modules imported successfully")
        
        return True
        
    except Exception as e:
        print(f"FAIL Import error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_basic_agent():
    """Test creating a basic agent."""
    print("\nAnalyzing Testing Basic Agent Creation...")
    
    try:
        from src.llamaagent.agents.base import BaseAgent, AgentConfig
        from src.llamaagent.llm.providers.mock_provider import MockProvider
        
        # Create mock provider
        mock_provider = MockProvider(model="test-model")
        
        # Create agent config
        config = AgentConfig(
            name="TestAgent",
            llm_provider=mock_provider
        )
        
        # Create agent
        agent = BaseAgent(config=config)
        print(f"PASS Created agent: {agent.config.name}")
        
        # Test agent execution
        from src.llamaagent.types import TaskInput
        
        task = TaskInput(
            id="test-task",
            data={"prompt": "What is 2+2?"}
        )
        
        result = await agent.execute_task(task)
        print(f"PASS Agent executed task: {result.status}")
        
        return True
        
    except Exception as e:
        print(f"FAIL Agent test error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_tools():
    """Test tool functionality."""
    print("\nAnalyzing Testing Tool Functionality...")
    
    try:
        from src.llamaagent.tools.calculator import CalculatorTool
        
        # Create calculator tool
        calc = CalculatorTool()
        
        # Test calculation
        result = await calc.execute(expression="2 + 2")
        print(f"PASS Calculator result: {result}")
        
        # Test tool info
        info = calc.get_info()
        print(f"PASS Tool info: {info['name']} - {info['description']}")
        
        return True
        
    except Exception as e:
        print(f"FAIL Tool test error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_llm_factory():
    """Test LLM factory."""
    print("\nAnalyzing Testing LLM Factory...")
    
    try:
        from src.llamaagent.llm.factory import LLMFactory
        
        # Create factory
        factory = LLMFactory()
        
        # List available providers
        providers = factory.list_providers()
        print(f"PASS Available providers: {providers}")
        
        # Create mock provider
        provider = factory.create_provider("mock", model="test-model")
        print(f"PASS Created provider: {provider.__class__.__name__}")
        
        return True
        
    except Exception as e:
        print(f"FAIL LLM factory test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cli():
    """Test CLI functionality."""
    print("\nAnalyzing Testing CLI...")
    
    try:
        from src.llamaagent.cli.main import app
        
        # Test that app exists and has commands
        if hasattr(app, 'registered_commands'):
            commands = list(app.registered_commands.keys())
            print(f"PASS CLI commands available: {commands}")
        else:
            print("PASS CLI app created successfully")
        
        return True
        
    except Exception as e:
        print(f"FAIL CLI test error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("=" * 50)
    print("Starting LlamaAgent Basic Functionality Test")
    print("=" * 50)
    
    tests = [
        test_basic_imports(),
        test_basic_agent(),
        test_tools(),
        test_llm_factory()
    ]
    
    # Run async tests
    results = await asyncio.gather(*tests)
    
    # Run sync tests
    cli_result = test_cli()
    results.append(cli_result)
    
    # Summary
    print("\n" + "=" * 50)
    print("RESULTS Test Summary")
    print("=" * 50)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r)
    
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    
    if passed_tests == total_tests:
        print("\nPASS All tests passed! The framework is working correctly.")
        print("\n Next steps:")
        print("  1. Run the CLI: python -m src.llamaagent.cli.main --help")
        print("  2. Start interactive mode: python -m src.llamaagent.cli.interactive")
        print("  3. Run the test suite: pytest tests/")
    else:
        print("\nWARNING:  Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    asyncio.run(main())
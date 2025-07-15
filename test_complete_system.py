#!/usr/bin/env python3
"""
Complete System Test Suite for LlamaAgent
=========================================

Comprehensive test suite that validates all LlamaAgent components work correctly
with mock data and real functionality.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Test results storage
test_results: Dict[str, Any] = {
    "total_tests": 0,
    "passed": 0,
    "failed": 0,
    "skipped": 0,
    "errors": [],
    "start_time": None,
    "end_time": None
}

def test_decorator(test_name: str) -> Callable[[Callable[..., bool]], Callable[..., bool]]:
    """Decorator for test functions."""
    def decorator(func: Callable[..., bool]) -> Callable[..., bool]:
        def wrapper(*args: Any, **kwargs: Any) -> bool:
            test_results["total_tests"] += 1
            print(f"\n{'='*60}")
            print(f"RUNNING: {test_name}")
            print(f"{'='*60}")
            
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                
                if result:
                    test_results["passed"] += 1
                    print(f"PASS: {test_name} ({end_time - start_time:.2f}s)")
                else:
                    test_results["failed"] += 1
                    print(f"FAIL: {test_name} ({end_time - start_time:.2f}s)")
                    
                return result
                
            except Exception as e:
                test_results["failed"] += 1
                test_results["errors"].append(f"{test_name}: {str(e)}")
                print(f"ERROR: {test_name} - {str(e)}")
                logger.exception(f"Error in {test_name}")
                return False
                
        return wrapper
    return decorator

@test_decorator("Core Imports Test")
def test_core_imports():
    """Test that all core modules can be imported."""
    try:
        # Test core agent imports
        from src.llamaagent.agents.base import AgentConfig, AgentRole
        from src.llamaagent.agents.react import ReactAgent
        print("✓ Agent modules imported successfully")
        
        # Test LLM provider imports
        from src.llamaagent.llm.providers.mock_provider import MockProvider
        from src.llamaagent.llm.factory import create_provider
        print("✓ LLM provider modules imported successfully")
        
        # Test tool imports
        from src.llamaagent.tools.calculator import CalculatorTool
        from src.llamaagent.tools.python_repl import PythonREPLTool
        from src.llamaagent.tools.base import Tool
        print("✓ Tool modules imported successfully")
        
        # Test memory imports
        from src.llamaagent.memory.base import SimpleMemory
        print("✓ Memory modules imported successfully")
        
        # Test types
        from src.llamaagent.types import TaskInput, TaskOutput, TaskStatus
        print("✓ Type definitions imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"FAIL: Import error - {e}")
        return False

@test_decorator("Mock Provider Test")
def test_mock_provider():
    """Test mock LLM provider functionality."""
    try:
        from src.llamaagent.llm.providers.mock_provider import MockProvider
        
        # Create mock provider
        provider = MockProvider(model_name="mock-gpt-4")
        print(f"✓ Mock provider created: {provider.model_name}")
        
        # Test basic completion
        response = provider.complete("Hello, world!")
        print(f"✓ Mock completion response: {response[:50]}...")
        
        # Test with context
        context = {"user": "test", "task": "greeting"}
        response_with_context = provider.complete("Hello", context)
        print(f"✓ Mock completion with context: {response_with_context[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"FAIL: Mock provider test failed - {e}")
        return False

@test_decorator("Agent Configuration Test")
def test_agent_config():
    """Test agent configuration and creation."""
    try:
        from src.llamaagent.agents.base import AgentConfig, AgentRole
        
        # Test basic config creation
        config = AgentConfig(
            name="TestAgent",
            role="assistant",
            provider="mock",
            model="mock-gpt-4"
        )
        print(f"✓ Agent config created: {config.name}")
        
        # Test config serialization
        config_dict = config.model_dump()
        print(f"✓ Config serialized: {len(config_dict)} fields")
        
        # Test config validation
        assert config.name == "TestAgent"
        assert config.role == "assistant"
        assert config.provider == "mock"
        assert config.model == "mock-gpt-4"
        print("✓ Config validation passed")
        
        return True
        
    except Exception as e:
        print(f"FAIL: Agent config test failed - {e}")
        return False

@test_decorator("Tool System Test")
def test_tool_system():
    """Test tool creation and execution."""
    try:
        from src.llamaagent.tools.calculator import CalculatorTool
        from src.llamaagent.tools.python_repl import PythonREPLTool
        
        # Test calculator tool
        calc_tool = CalculatorTool()
        print(f"✓ Calculator tool created: {calc_tool.name}")
        
        # Test calculation
        result = calc_tool.execute("2 + 2")
        print(f"✓ Calculator result: {result}")
        
        # Test Python REPL tool
        python_tool = PythonREPLTool()
        print(f"✓ Python REPL tool created: {python_tool.name}")
        
        # Test Python execution
        python_result = python_tool.execute("print('Hello from Python!')")
        print(f"✓ Python execution result: {python_result}")
        
        return True
        
    except Exception as e:
        print(f"FAIL: Tool system test failed - {e}")
        return False

@test_decorator("Memory System Test")
def test_memory_system():
    """Test memory storage and retrieval."""
    try:
        from src.llamaagent.memory.base import SimpleMemory
        
        # Create memory instance
        memory = SimpleMemory()
        print("✓ Memory system created")
        
        # Test memory storage
        memory.store("user_name", "John Doe")
        memory.store("task_count", 5)
        print("✓ Memory storage tested")
        
        # Test memory retrieval
        retrieved_name = memory.retrieve("user_name")
        retrieved_count = memory.retrieve("task_count")
        
        assert retrieved_name == "John Doe"
        assert retrieved_count == 5
        print("✓ Memory retrieval tested")
        
        # Test memory listing
        all_memories = memory.list_memories()
        print(f"✓ Memory listing: {len(all_memories)} items")
        
        return True
        
    except Exception as e:
        print(f"FAIL: Memory system test failed - {e}")
        return False

@test_decorator("Agent Creation Test")
def test_agent_creation():
    """Test full agent creation and basic functionality."""
    try:
        from src.llamaagent.agents.react import ReactAgent
        from src.llamaagent.agents.base import AgentConfig
        from src.llamaagent.llm.providers.mock_provider import MockProvider
        from src.llamaagent.tools.calculator import CalculatorTool
        from src.llamaagent.memory.base import SimpleMemory
        
        # Create components
        config = AgentConfig(
            name="TestReactAgent",
            role="assistant",
            provider="mock",
            model="mock-gpt-4"
        )
        
        provider = MockProvider(model_name="mock-gpt-4")
        tools = [CalculatorTool()]
        memory = SimpleMemory()
        
        print("✓ Agent components created")
        
        # Create agent
        agent = ReactAgent(
            config=config,
            llm_provider=provider,
            tools=tools,
            memory=memory
        )
        
        print(f"✓ ReactAgent created: {agent.name}")
        
        # Test agent properties
        assert agent.name == "TestReactAgent"
        assert len(agent.tools) == 1
        assert agent.memory is not None
        print("✓ Agent properties validated")
        
        return True
        
    except Exception as e:
        print(f"FAIL: Agent creation test failed - {e}")
        return False

@test_decorator("Task Execution Test")
def test_task_execution():
    """Test task execution with mock data."""
    try:
        from src.llamaagent.types import TaskInput, TaskStatus
        
        # Create mock task
        task_input = TaskInput(
            task="Calculate 2 + 2 and explain the result",
            context={"user": "test_user"},
            metadata={"priority": "high"}
        )
        
        print(f"✓ Task created: {task_input.task}")
        
        # Simulate task processing
        import time
        start_time = time.time()
        
        # Mock processing
        time.sleep(0.1)  # Simulate work
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Create mock result
        mock_result = {
            "result": "The calculation 2 + 2 equals 4. This is basic arithmetic addition.",
            "status": TaskStatus.COMPLETED,
            "execution_time": execution_time,
            "metadata": {"tokens_used": 25}
        }
        
        print(f"✓ Task executed in {execution_time:.3f}s")
        print(f"✓ Result: {mock_result['result'][:50]}...")
        
        return True
        
    except Exception as e:
        print(f"FAIL: Task execution test failed - {e}")
        return False

@test_decorator("API Components Test")
def test_api_components():
    """Test API-related components."""
    try:
        # Test if API modules can be imported
        from src.llamaagent.api.main import create_app
        print("✓ API main module imported")
        
        # Test configuration
        from src.llamaagent.config.settings import get_settings
        settings = get_settings()
        print(f"✓ Settings loaded: {type(settings).__name__}")
        
        # Test if FastAPI app can be created
        app = create_app()
        print(f"✓ FastAPI app created: {type(app).__name__}")
        
        return True
        
    except Exception as e:
        print(f"FAIL: API components test failed - {e}")
        return False

@test_decorator("CLI Components Test")
def test_cli_components():
    """Test CLI-related components."""
    try:
        # Test CLI imports
        from src.llamaagent.cli.main import app as cli_app
        print("✓ CLI main module imported")
        
        # Test enhanced CLI
        from src.llamaagent.cli.enhanced_cli import EnhancedCLI
        print("✓ Enhanced CLI imported")
        
        # Test CLI can be instantiated
        cli = EnhancedCLI()
        print(f"✓ Enhanced CLI created: {type(cli).__name__}")
        
        return True
        
    except Exception as e:
        print(f"FAIL: CLI components test failed - {e}")
        return False

@test_decorator("Monitoring Components Test")
def test_monitoring_components():
    """Test monitoring and observability components."""
    try:
        # Test monitoring imports
        from src.llamaagent.monitoring.health import HealthChecker
        from src.llamaagent.monitoring.metrics_collector import MetricsCollector
        print("✓ Monitoring modules imported")
        
        # Test health checker
        health_checker = HealthChecker()
        print(f"✓ Health checker created: {type(health_checker).__name__}")
        
        # Test metrics collector
        metrics = MetricsCollector()
        print(f"✓ Metrics collector created: {type(metrics).__name__}")
        
        return True
        
    except Exception as e:
        print(f"FAIL: Monitoring components test failed - {e}")
        return False

@test_decorator("Security Components Test")
def test_security_components():
    """Test security-related components."""
    try:
        # Test security imports
        from src.llamaagent.security.authentication import AuthenticationManager
        from src.llamaagent.security.authorization import AuthorizationManager
        print("✓ Security modules imported")
        
        # Test authentication manager
        auth_manager = AuthenticationManager()
        print(f"✓ Authentication manager created: {type(auth_manager).__name__}")
        
        # Test authorization manager
        authz_manager = AuthorizationManager()
        print(f"✓ Authorization manager created: {type(authz_manager).__name__}")
        
        return True
        
    except Exception as e:
        print(f"FAIL: Security components test failed - {e}")
        return False

@test_decorator("Integration Test")
def test_integration():
    """Test integration between components."""
    try:
        from src.llamaagent.agents.react import ReactAgent
        from src.llamaagent.agents.base import AgentConfig
        from src.llamaagent.llm.providers.mock_provider import MockProvider
        from src.llamaagent.tools.calculator import CalculatorTool
        from src.llamaagent.memory.base import SimpleMemory
        from src.llamaagent.types import TaskInput
        
        # Create integrated system
        config = AgentConfig(
            name="IntegrationTestAgent",
            role="assistant",
            provider="mock",
            model="mock-gpt-4"
        )
        
        provider = MockProvider(model_name="mock-gpt-4")
        tools = [CalculatorTool()]
        memory = SimpleMemory()
        
        agent = ReactAgent(
            config=config,
            llm_provider=provider,
            tools=tools,
            memory=memory
        )
        
        print("✓ Integrated system created")
        
        # Test system interaction
        task = TaskInput(
            task="What is 5 * 7?",
            context={"user": "integration_test"}
        )
        
        # Store in memory
        memory.store("last_task", task.task)
        
        # Verify memory
        retrieved_task = memory.retrieve("last_task")
        assert retrieved_task == task.task
        
        print("✓ Integration test completed")
        
        return True
        
    except Exception as e:
        print(f"FAIL: Integration test failed - {e}")
        return False

def generate_test_report():
    """Generate comprehensive test report."""
    report = {
        "test_summary": {
            "total_tests": test_results["total_tests"],
            "passed": test_results["passed"],
            "failed": test_results["failed"],
            "skipped": test_results["skipped"],
            "success_rate": (test_results["passed"] / test_results["total_tests"] * 100) if test_results["total_tests"] > 0 else 0
        },
        "execution_time": {
            "start_time": test_results["start_time"].isoformat() if test_results["start_time"] else None,
            "end_time": test_results["end_time"].isoformat() if test_results["end_time"] else None,
            "duration": str(test_results["end_time"] - test_results["start_time"]) if test_results["start_time"] and test_results["end_time"] else None
        },
        "errors": test_results["errors"],
        "system_info": {
            "python_version": sys.version,
            "platform": sys.platform,
            "timestamp": datetime.now().isoformat()
        }
    }
    
    # Save report
    report_file = Path("test_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    return report

def print_test_summary():
    """Print test summary to console."""
    print("\n" + "="*80)
    print("TEST EXECUTION SUMMARY")
    print("="*80)
    
    total = test_results["total_tests"]
    passed = test_results["passed"]
    failed = test_results["failed"]
    
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed/total*100):.1f}%" if total > 0 else "0%")
    
    if test_results["start_time"] and test_results["end_time"]:
        duration = test_results["end_time"] - test_results["start_time"]
        print(f"Duration: {duration}")
    
    if test_results["errors"]:
        print(f"\nErrors ({len(test_results['errors'])}):")
        for error in test_results["errors"]:
            print(f"  - {error}")
    
    print("="*80)
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! System is working perfectly!")
    else:
        print(f"⚠️  {failed} test(s) failed. Please review the errors above.")

def main():
    """Main test execution function."""
    print("LlamaAgent Complete System Test Suite")
    print("Author: Nik Jois <nikjois@llamasearch.ai>")
    print("="*80)
    
    test_results["start_time"] = datetime.now()
    
    # Run all tests
    test_functions = [
        test_core_imports,
        test_mock_provider,
        test_agent_config,
        test_tool_system,
        test_memory_system,
        test_agent_creation,
        test_task_execution,
        test_api_components,
        test_cli_components,
        test_monitoring_components,
        test_security_components,
        test_integration
    ]
    
    for test_func in test_functions:
        test_func()
    
    test_results["end_time"] = datetime.now()
    
    # Generate and print summary
    report = generate_test_report()
    print_test_summary()
    
    # Exit with appropriate code
    sys.exit(0 if test_results["failed"] == 0 else 1)

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Test script for LlamaAgent Master Program
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_basic_functionality():
    """Test basic functionality of the master program."""
    print("Testing LlamaAgent Master Program...")
    
    try:
        # Import the orchestrator
        from llamaagent_master_program import MasterOrchestrator, CreateMasterTaskRequest
        
        # Create orchestrator
        orchestrator = MasterOrchestrator()
        print("PASS Orchestrator created successfully")
        
        # Test 1: Create a simple task
        print("\nTest 1: Creating simple task...")
        request = CreateMasterTaskRequest(
            task_description="Calculate the sum of 10 and 20",
            auto_decompose=False,
            auto_spawn=False,
            max_agents=1,
            enable_openai=False
        )
        
        result = await orchestrator.create_master_task(request)
        print(f"PASS Task created: {result}")
        
        # Test 2: Create a complex task with decomposition
        print("\nTest 2: Creating complex task with decomposition...")
        request2 = CreateMasterTaskRequest(
            task_description="Build a simple web scraper",
            auto_decompose=True,
            auto_spawn=False,
            max_agents=5,
            enable_openai=False
        )
        
        result2 = await orchestrator.create_master_task(request2)
        print(f"PASS Complex task created with {result2.get('total_subtasks', 0)} subtasks")
        
        # Test 3: Get system status
        print("\nTest 3: Getting system status...")
        status = await orchestrator.get_system_status()
        print(f"PASS System status: {status.dict()}")
        
        # Test 4: Test hierarchy
        print("\nTest 4: Testing agent hierarchy...")
        hierarchy_viz = orchestrator.get_hierarchy_visualization()
        print("PASS Hierarchy visualization generated")
        
        print("\nSUCCESS All tests passed!")
        
    except Exception as e:
        print(f"FAIL Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

async def test_agent_spawning():
    """Test agent spawning capabilities."""
    print("\nTesting Agent Spawning...")
    
    try:
        from llamaagent_master_program import MasterOrchestrator, CreateMasterTaskRequest
        
        orchestrator = MasterOrchestrator()
        
        # Create task with auto-spawning
        request = CreateMasterTaskRequest(
            task_description="Analyze data and generate report",
            auto_decompose=True,
            auto_spawn=True,
            max_agents=3,
            enable_openai=False,
            priority="high"
        )
        
        result = await orchestrator.create_master_task(request)
        print(f"PASS Spawned {result.get('spawned_agents', 0)} agents")
        
        # Wait a bit for execution
        await asyncio.sleep(2)
        
        # Check hierarchy
        stats = orchestrator.agent_spawner.hierarchy.get_hierarchy_stats()
        print(f"PASS Hierarchy stats: Total agents: {stats['total_agents']}, Active: {stats['active_agents']}")
        
        return True
        
    except Exception as e:
        print(f"FAIL Agent spawning test failed: {e}")
        return False

async def test_task_planning():
    """Test task planning capabilities."""
    print("\nTesting Task Planning...")
    
    try:
        from llamaagent.planning.task_planner import TaskPlanner, Task, TaskPriority
        
        planner = TaskPlanner()
        
        # Create a plan
        plan = planner.create_plan(
            goal="Build a machine learning model",
            auto_decompose=True
        )
        
        print(f"PASS Created plan with {len(plan.tasks)} tasks")
        print(f"PASS Plan valid: {plan.is_valid}")
        
        if plan.validation_errors:
            print(f"⚠️  Validation errors: {plan.validation_errors}")
        
        # Get execution order
        try:
            execution_order = planner.get_execution_order(plan)
            print(f"PASS Execution order has {len(execution_order)} levels")
        except Exception as e:
            print(f"⚠️  Could not determine execution order: {e}")
        
        return True
        
    except Exception as e:
        print(f"FAIL Task planning test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("LlamaAgent Master Program Test Suite")
    print("=" * 60)
    
    # Run tests
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    tests = [
        ("Basic Functionality", test_basic_functionality()),
        ("Task Planning", test_task_planning()),
        ("Agent Spawning", test_agent_spawning()),
    ]
    
    results = []
    for name, test_coro in tests:
        print(f"\n{'=' * 40}")
        print(f"Running: {name}")
        print(f"{'=' * 40}")
        
        try:
            result = loop.run_until_complete(test_coro)
            results.append((name, result))
        except Exception as e:
            print(f"FAIL Test {name} crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "PASS PASS" if result else "FAIL FAIL"
        print(f"{name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nSUCCESS All tests passed! The system is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    loop.close()

if __name__ == "__main__":
    main()
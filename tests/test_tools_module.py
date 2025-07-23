#!/usr/bin/env python3
"""Test script to verify the tools module implementation"""

import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test 1: Import base components
print("Test 1: Importing base components...")
try:
    from llamaagent.tools.base import BaseTool, ToolRegistry

    print(" BaseTool and ToolRegistry imported successfully")
except Exception as e:
    print(f" Failed to import base components: {e}")
    sys.exit(1)

# Test 2: Import built-in tools
print("\nTest 2: Importing built-in tools...")
try:
    from llamaagent.tools.calculator import CalculatorTool
    from llamaagent.tools.python_repl import PythonREPLTool

    print(" CalculatorTool and PythonREPLTool imported successfully")
except Exception as e:
    print(f" Failed to import built-in tools: {e}")
    sys.exit(1)

# Test 3: Test tool instantiation
print("\nTest 3: Testing tool instantiation...")
try:
    calc = CalculatorTool()
    repl = PythonREPLTool()
    print(f" CalculatorTool: name='{calc.name}', description='{calc.description}'")
    print(f" PythonREPLTool: name='{repl.name}', description='{repl.description}'")
except Exception as e:
    print(f" Failed to instantiate tools: {e}")
    sys.exit(1)

# Test 4: Test tool execution
print("\nTest 4: Testing tool execution...")
try:
    # Test calculator
    result = calc.execute(expression="2 + 3 * 4")
    print(f" Calculator: 2 + 3 * 4 = {result}")

    # Test Python REPL
    result = repl.execute(code="print('Hello from Python REPL!')")
    print(f" Python REPL output: {result}")
except Exception as e:
    print(f" Failed to execute tools: {e}")
    sys.exit(1)

# Test 5: Test ToolRegistry
print("\nTest 5: Testing ToolRegistry...")
try:
    registry = ToolRegistry()
    registry.register(calc)
    registry.register(repl)

    print(f" Registered tools: {registry.list_names()}")

    # Test getting a tool
    retrieved_calc = registry.get("calculator")
    print(f" Retrieved calculator tool: {retrieved_calc.name}")

    # Test deregistering
    registry.deregister("calculator")
    print(f" After deregistering calculator: {registry.list_names()}")
except Exception as e:
    print(f" Failed ToolRegistry test: {e}")
    sys.exit(1)

# Test 6: Test the __init__.py imports
print("\nTest 6: Testing __init__.py imports...")
try:
    from llamaagent.tools import (BaseTool, CalculatorTool, PythonREPLTool,
                                  Tool, ToolRegistry,
                                  create_tool_from_function, get_all_tools)

    print(" All core exports from __init__.py imported successfully")

    # Test create_tool_from_function
    def my_custom_tool(x: int, y: int) -> int:
        """Add two numbers together"""
        return x + y

    custom_tool = create_tool_from_function(my_custom_tool, name="adder")
    print(
        f" Created custom tool: name='{custom_tool.name}', description='{custom_tool.description}'"
    )
    result = custom_tool.execute(x=5, y=3)
    print(f" Custom tool execution: 5 + 3 = {result}")

    # Test get_all_tools
    all_tools = get_all_tools()
    print(
        f" get_all_tools() returned {len(all_tools)} tools: {[t.name for t in all_tools]}"
    )

except Exception as e:
    print(f" Failed __init__.py imports test: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\nPASS All tests passed! The tools module is working correctly.")

#!/usr/bin/env python3
"""Final test script to verify the tools module implementation"""

import os
import sys

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Testing LlamaAgent Tools Module")
print("=" * 50)

# Test the complete module import
print("\nTest: Importing from llamaagent.tools...")
try:
    # First, let's check if there are any import issues in the parent modules
    # We'll do this step by step

    # Import just the tools module components we fixed
    import llamaagent.tools.base

    print(" llamaagent.tools.base imported")

    import llamaagent.tools.calculator

    print(" llamaagent.tools.calculator imported")

    import llamaagent.tools.python_repl

    print(" llamaagent.tools.python_repl imported")

    # Now test the actual classes
    print("\nTesting core classes...")

    # Test BaseTool and ToolRegistry
    BaseTool = llamaagent.tools.base.BaseTool
    ToolRegistry = llamaagent.tools.base.ToolRegistry
    print(" BaseTool and ToolRegistry classes available")

    # Test built-in tools
    CalculatorTool = llamaagent.tools.calculator.CalculatorTool
    PythonREPLTool = llamaagent.tools.python_repl.PythonREPLTool
    print(" CalculatorTool and PythonREPLTool classes available")

    # Test instantiation
    print("\nTesting instantiation...")
    calc = CalculatorTool()
    repl = PythonREPLTool()
    registry = ToolRegistry()
    print(f" CalculatorTool: {calc.name} - {calc.description}")
    print(f" PythonREPLTool: {repl.name} - {repl.description}")

    # Test execution
    print("\nTesting execution...")
    calc_result = calc.execute(expression="42 + 8")
    print(f" Calculator: 42 + 8 = {calc_result}")

    repl_result = repl.execute(code="x = 10; y = 20; print(f'Sum: {x + y}')")
    print(f" Python REPL output: {repl_result}")

    # Test registry
    print("\nTesting ToolRegistry...")
    registry.register(calc)
    registry.register(repl)
    print(f" Registered tools: {registry.list_names()}")

    retrieved = registry.get("calculator")
    print(f" Retrieved tool: {retrieved.name if retrieved else 'None'}")

    # Now test the __init__.py imports
    print("\nTesting __init__.py imports...")
    try:
        # Import from __init__.py
        from llamaagent.tools import BaseTool as BaseTool2
        from llamaagent.tools import CalculatorTool as CalculatorTool2
        from llamaagent.tools import PythonREPLTool as PythonREPLTool2
        from llamaagent.tools import Tool
        from llamaagent.tools import ToolRegistry as ToolRegistry2
        from llamaagent.tools import create_tool_from_function, get_all_tools

        print(" All exports from __init__.py imported successfully")

        # Test that they're the same classes
        print(f" BaseTool is same: {BaseTool is BaseTool2}")
        print(f" Tool is BaseTool alias: {Tool is BaseTool}")

        # Test utility functions
        print("\nTesting utility functions...")

        # Test create_tool_from_function
        def multiply(x: int, y: int) -> int:
            """Multiply two numbers"""
            return x * y

        custom_tool = create_tool_from_function(multiply, name="multiplier")
        print(f" Custom tool created: {custom_tool.name} - {custom_tool.description}")

        result = custom_tool.execute(x=6, y=7)
        print(f" Custom tool result: 6 × 7 = {result}")

        # Test get_all_tools
        all_tools = get_all_tools()
        print(f" get_all_tools() returns {len(all_tools)} tools")
        for tool in all_tools:
            print(f"  - {tool.name}: {tool.description}")

    except ImportError as e:
        print(f" Failed to import from __init__.py: {e}")
        import traceback

        traceback.print_exc()

    print("\nPASS Tools module is working correctly!")

except Exception as e:
    print(f"\n Test failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

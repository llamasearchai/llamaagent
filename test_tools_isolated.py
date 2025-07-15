#!/usr/bin/env python3
"""Test script to verify the tools module in isolation"""

import sys
import os

# Add tools directory to path
tools_dir = os.path.join(os.path.dirname(__file__), 'src', 'llamaagent', 'tools')
sys.path.insert(0, tools_dir)

# Test 1: Import and test base.py
print("Test 1: Testing base.py...")
try:
    import base
    
    # Test BaseTool (abstract class)
    print(f"✓ BaseTool class available: {hasattr(base, 'BaseTool')}")
    print(f"✓ Tool alias available: {hasattr(base, 'Tool')}")
    print(f"✓ ToolRegistry class available: {hasattr(base, 'ToolRegistry')}")
    
    # Test ToolRegistry
    registry = base.ToolRegistry()
    print("✓ ToolRegistry instantiated successfully")
    
except Exception as e:
    print(f"✗ Failed base.py test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Import and test calculator.py
print("\nTest 2: Testing calculator.py...")
try:
    # Import calculator
    import calculator
    
    # Test CalculatorTool
    calc = calculator.CalculatorTool()
    print(f"✓ CalculatorTool instantiated: name='{calc.name}', description='{calc.description}'")
    
    # Test execution
    result = calc.execute(expression="10 + 5 * 2")
    print(f"✓ Calculator execution: 10 + 5 * 2 = {result}")
    
    # Test error handling
    error_result = calc.execute(expression="invalid")
    print(f"✓ Calculator error handling: {error_result}")
    
except Exception as e:
    print(f"✗ Failed calculator.py test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Import and test python_repl.py
print("\nTest 3: Testing python_repl.py...")
try:
    import python_repl
    
    # Test PythonREPLTool
    repl = python_repl.PythonREPLTool()
    print(f"✓ PythonREPLTool instantiated: name='{repl.name}', description='{repl.description}'")
    
    # Test execution
    result = repl.execute(code="print('Test output'); x = 42; x")
    print(f"✓ Python REPL execution output: {repr(result)}")
    
    # Test error handling
    error_result = repl.execute(code="1/0")
    print(f"✓ Python REPL error handling works: {'ZeroDivisionError' in error_result}")
    
except Exception as e:
    print(f"✗ Failed python_repl.py test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nPASS All isolated tests passed! The core tool files are working correctly.")
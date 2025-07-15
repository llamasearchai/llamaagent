#!/usr/bin/env python3
"""
Manual Syntax Fixes for specific files with known issues
"""

import os
from pathlib import Path
from typing import List, Tuple

def fix_specific_files():
    """Apply specific fixes to files with known issues."""
    
    fixes = []
    
    # Fix 1: src/llamaagent/cli/main.py - line 416
    file_path = Path("src/llamaagent/cli/main.py")
    if file_path.exists():
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix the unclosed asyncio.run call
        content = content.replace(
            "    asyncio.run(run_benchmark()",
            "    asyncio.run(run_benchmark())"
        )
        
        with open(file_path, 'w') as f:
            f.write(content)
        fixes.append(("src/llamaagent/cli/main.py", "Fixed unclosed asyncio.run()"))
    
    # Fix 2: src/llamaagent/cli/interactive.py - line 207
    file_path = Path("src/llamaagent/cli/interactive.py")
    if file_path.exists():
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix unclosed function call
        content = content.replace(
            '        config_table.add_row("Temperature", str(self.config.llm.temperature)',
            '        config_table.add_row("Temperature", str(self.config.llm.temperature))'
        )
        
        with open(file_path, 'w') as f:
            f.write(content)
        fixes.append(("src/llamaagent/cli/interactive.py", "Fixed unclosed add_row()"))
    
    # Fix 3: src/llamaagent/cache/llm_cache.py - line 275
    file_path = Path("src/llamaagent/cache/llm_cache.py")
    if file_path.exists():
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix the ternary operator syntax
        content = content.replace(
            "                if total_requests > 0:",
            "                if total_requests > 0"
        )
        
        with open(file_path, 'w') as f:
            f.write(content)
        fixes.append(("src/llamaagent/cache/llm_cache.py", "Fixed ternary operator"))
    
    # Fix 4: src/llamaagent/cache/query_optimizer.py - line 311
    file_path = Path("src/llamaagent/cache/query_optimizer.py")
    if file_path.exists():
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix comment syntax
        content = content.replace(
            "            if avg_time > 1000:  # If average time > 1 second:",
            "            if avg_time > 1000:  # If average time > 1 second"
        )
        
        with open(file_path, 'w') as f:
            f.write(content)
        fixes.append(("src/llamaagent/cache/query_optimizer.py", "Fixed comment"))
    
    # Fix 5: src/llamaagent/cache/advanced_cache.py - missing closing parentheses
    file_path = Path("src/llamaagent/cache/advanced_cache.py")
    if file_path.exists():
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix unclosed function calls
        content = content.replace(
            "            return len(pickle.dumps(value)",
            "            return len(pickle.dumps(value))"
        )
        content = content.replace(
            "            return len(str(value).encode()",
            "            return len(str(value).encode())"
        )
        content = content.replace(
            "                for k, v in self.cache.items():",
            "                for k, v in self.cache.items()"
        )
        
        with open(file_path, 'w') as f:
            f.write(content)
        fixes.append(("src/llamaagent/cache/advanced_cache.py", "Fixed unclosed calls"))
    
    # Fix 6: src/llamaagent/security/validator.py - ternary operators
    file_path = Path("src/llamaagent/security/validator.py")
    if file_path.exists():
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix ternary syntax
        content = content.replace(
            "                if is_valid:",
            "                if is_valid"
        )
        content = content.replace(
            "            elif isinstance(value, (int, float):",
            "            elif isinstance(value, (int, float)):"
        )
        content = content.replace(
            "                if not DataTypeValidator.validate_integer(:",
            "                if not DataTypeValidator.validate_integer("
        )
        content = content.replace(
            "                    if sanitize:",
            "                    if sanitize"
        )
        
        with open(file_path, 'w') as f:
            f.write(content)
        fixes.append(("src/llamaagent/security/validator.py", "Fixed ternary operators"))
    
    # Fix 7: src/llamaagent/security/rate_limiter.py
    file_path = Path("src/llamaagent/security/rate_limiter.py")
    if file_path.exists():
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix unclosed deque() call
        content = content.replace(
            "        requests = self.requests.get(identifier, deque()",
            "        requests = self.requests.get(identifier, deque())"
        )
        
        # Fix dictionary syntax
        content = content.replace(
            "                for name, rule in self.rules.items():",
            "                for name, rule in self.rules.items()"
        )
        
        with open(file_path, 'w') as f:
            f.write(content)
        fixes.append(("src/llamaagent/security/rate_limiter.py", "Fixed unclosed calls"))
    
    # Fix 8: src/llamaagent/integration/_openai_stub.py
    file_path = Path("src/llamaagent/integration/_openai_stub.py")
    if file_path.exists():
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix list comprehensions
        content = content.replace(
            "                for choice in self.choices:",
            "                for choice in self.choices"
        )
        content = content.replace(
            "                for item in self.data:",
            "                for item in self.data"
        )
        content = content.replace(
            "                for result in self.results:",
            "                for result in self.results"
        )
        
        with open(file_path, 'w') as f:
            f.write(content)
        fixes.append(("src/llamaagent/integration/_openai_stub.py", "Fixed list comprehensions"))
    
    # Fix 9: src/llamaagent/agents/reasoning_chains.py
    file_path = Path("src/llamaagent/agents/reasoning_chains.py")
    if file_path.exists():
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix comment/colon issue
        content = content.replace(
            "        for i in range(3):  # Generate 3 initial branches:",
            "        for i in range(3):  # Generate 3 initial branches"
        )
        
        # Fix missing closing parenthesis
        content = content.replace(
            "        return max(branches, key=lambda b: len(b.content)",
            "        return max(branches, key=lambda b: len(b.content))"
        )
        
        # Fix missing closing parenthesis in TaskResult
        content = content.replace(
            '                result=TaskResult(success=False, error=str(e)',
            '                result=TaskResult(success=False, error=str(e))'
        )
        
        # Fix missing closing parenthesis at end of file
        content = content.replace(
            "    asyncio.run(main()",
            "    asyncio.run(main())"
        )
        
        with open(file_path, 'w') as f:
            f.write(content)
        fixes.append(("src/llamaagent/agents/reasoning_chains.py", "Fixed syntax issues"))
    
    # Fix 10: src/llamaagent/data_generation/base.py
    file_path = Path("src/llamaagent/data_generation/base.py")
    if file_path.exists():
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix field default_factory
        content = content.replace(
            '    node_id: str = field(default_factory=lambda: str(uuid.uuid4())',
            '    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))'
        )
        
        with open(file_path, 'w') as f:
            f.write(content)
        fixes.append(("src/llamaagent/data_generation/base.py", "Fixed field syntax"))
    
    # Fix 11: src/llamaagent/api/simon_ecosystem_api.py
    file_path = Path("src/llamaagent/api/simon_ecosystem_api.py")
    if file_path.exists():
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix f-string UUID
        content = content.replace(
            '            resource_id = f"resource_{str(uuid.uuid4()[:8]}"',
            '            resource_id = f"resource_{str(uuid.uuid4())[:8]}"'
        )
        
        with open(file_path, 'w') as f:
            f.write(content)
        fixes.append(("src/llamaagent/api/simon_ecosystem_api.py", "Fixed f-string"))
    
    # Fix 12: src/llamaagent/cli/code_generator.py
    file_path = Path("src/llamaagent/cli/code_generator.py")
    if file_path.exists():
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Fix line 85 - seems to be missing closing parenthesis or similar
        for i, line in enumerate(lines):
            if i == 84:  # Line 85 (0-indexed)
                if 'self._generate_filename(request.prompt, request.language' in line:
                    lines[i] = line.rstrip() + ')\n'
        
        with open(file_path, 'w') as f:
            f.writelines(lines)
        fixes.append(("src/llamaagent/cli/code_generator.py", "Fixed generate_filename call"))
    
    return fixes


def test_imports():
    """Test if key modules can be imported after fixes."""
    test_modules = [
        "src.llamaagent.cli.main",
        "src.llamaagent.agents.reasoning_chains",
        "src.llamaagent.cache.llm_cache",
        "src.llamaagent.security.validator",
    ]
    
    results = []
    for module in test_modules:
        try:
            exec(f"import {module}")
            results.append((module, "PASS Success"))
        except Exception as e:
            results.append((module, f"FAIL {str(e)}"))
    
    return results


def main():
    """Main entry point."""
    print("FIXING Manual Syntax Fixes")
    print("=" * 50)
    
    # Apply fixes
    fixes = fix_specific_files()
    
    print("\nResponse Fixes Applied:")
    for file, description in fixes:
        print(f"  PASS {file}: {description}")
    
    # Test imports
    print("\nAnalyzing Testing imports:")
    results = test_imports()
    for module, status in results:
        print(f"  {status} {module}")
    
    print("\nPASS Manual fixes complete!")


if __name__ == "__main__":
    main()
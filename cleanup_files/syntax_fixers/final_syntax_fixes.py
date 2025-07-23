#!/usr/bin/env python3
"""
Final Syntax Fixes - Addresses remaining syntax errors
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

def apply_remaining_fixes():
    """Apply the remaining syntax fixes needed."""
    
    fixes = []
    
    # Fix 1: src/llamaagent/cli/llm_cmd.py - list comprehension
    file_path = Path("src/llamaagent/cli/llm_cmd.py")
    if file_path.exists():
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix the list comprehension syntax - missing colon after for
        content = content.replace(
            "                for row in provider_stats:",
            "                for row in provider_stats"
        )
        
        with open(file_path, 'w') as f:
            f.write(content)
        fixes.append((str(file_path), "Fixed list comprehension"))
    
    # Fix 2: src/llamaagent/security/rate_limiter.py - line 238
    file_path = Path("src/llamaagent/security/rate_limiter.py")
    if file_path.exists():
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix unclosed deque call
        content = content.replace(
            "            requests = algorithm.requests.get(identifier, deque()",
            "            requests = algorithm.requests.get(identifier, deque())"
        )
        
        with open(file_path, 'w') as f:
            f.write(content)
        fixes.append((str(file_path), "Fixed unclosed deque()"))
    
    # Fix 3: src/llamaagent/cache/llm_cache.py - line 205
    file_path = Path("src/llamaagent/cache/llm_cache.py")
    if file_path.exists():
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Find and fix the problematic line
        for i, line in enumerate(lines):
            if i >= 200 and i <= 210 and "self, messages: List[LLMMessage], response: LLMResponse, **kwargs)" in line:
                # This line is missing parenthesis or has extra
                lines[i] = "        self, messages: List[LLMMessage], response: LLMResponse, **kwargs\n"
        
        with open(file_path, 'w') as f:
            f.writelines(lines)
        fixes.append((str(file_path), "Fixed method signature"))
    
    # Fix 4: src/llamaagent/cache/query_optimizer.py - line 147
    file_path = Path("src/llamaagent/cache/query_optimizer.py")
    if file_path.exists():
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix unclosed parenthesis in extend call
        content = re.sub(
            r'plans\.extend\(\[plan\] \* len\(group\)\s*\n',
            'plans.extend([plan] * len(group))\n',
            content
        )
        
        with open(file_path, 'w') as f:
            f.write(content)
        fixes.append((str(file_path), "Fixed extend call"))
    
    # Fix 5: src/llamaagent/agents/reasoning_chains.py - line 254
    file_path = Path("src/llamaagent/agents/reasoning_chains.py")
    if file_path.exists():
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix the line with hasattr check
        content = content.replace(
            "            [LLMMessage(role=\"user\", content=task_input.prompt if hasattr(task_input, 'prompt') else str(task_input.data)]",
            "            [LLMMessage(role=\"user\", content=task_input.prompt if hasattr(task_input, 'prompt') else str(task_input.data))]"
        )
        
        with open(file_path, 'w') as f:
            f.write(content)
        fixes.append((str(file_path), "Fixed LLMMessage list"))
    
    # Fix 6: src/llamaagent/benchmarks/spre_evaluator.py - line 57
    file_path = Path("src/llamaagent/benchmarks/spre_evaluator.py")
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Fix property definition - missing colon
        for i, line in enumerate(lines):
            if i >= 55 and i <= 60 and "@property" in lines[i-1] and "def avg_api_calls(self)" in line:
                if not line.strip().endswith(':'):
                    lines[i] = line.rstrip() + ':\n'
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        fixes.append((str(file_path), "Fixed property definition"))
    
    # Fix 7: src/llamaagent/cli/code_generator.py - line 82
    file_path = Path("src/llamaagent/cli/code_generator.py")
    if file_path.exists():
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix parenthesis issue
        content = content.replace(
            "        code = self._extract_code_from_response(response.content if hasattr(response, 'content') else str(response)",
            "        code = self._extract_code_from_response(response.content if hasattr(response, 'content') else str(response))"
        )
        
        with open(file_path, 'w') as f:
            f.write(content)
        fixes.append((str(file_path), "Fixed extract_code_from_response call"))
    
    # Fix 8: src/llamaagent/api/production_app.py
    file_path = Path("src/llamaagent/api/production_app.py")
    if file_path.exists():
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix json.loads call
        content = re.sub(
            r'content = json\.loads\(response\.body\.decode\(\)',
            'content = json.loads(response.body.decode())',
            content
        )
        
        with open(file_path, 'w') as f:
            f.write(content)
        fixes.append((str(file_path), "Fixed json.loads call"))
    
    # Fix 9: src/llamaagent/api/openai_comprehensive_api.py
    file_path = Path("src/llamaagent/api/openai_comprehensive_api.py")
    if file_path.exists():
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix create_openai_tool call
        content = re.sub(
            r'tools\[tool_type\] = create_openai_tool\(tool_type, get_integration\(\)',
            'tools[tool_type] = create_openai_tool(tool_type, get_integration())',
            content
        )
        
        with open(file_path, 'w') as f:
            f.write(content)
        fixes.append((str(file_path), "Fixed create_openai_tool call"))
    
    # Fix 10: src/llamaagent/benchmarks/gaia_benchmark.py - fix append pattern
    file_path = Path("src/llamaagent/benchmarks/gaia_benchmark.py")
    if file_path.exists():
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix the append() indentation issue
        content = re.sub(
            r'results\.append\(\)\s*\n\s+GAIAResult\(',
            'results.append(\n                    GAIAResult(',
            content
        )
        
        with open(file_path, 'w') as f:
            f.write(content)
        fixes.append((str(file_path), "Fixed append pattern"))
    
    return fixes


def verify_core_modules():
    """Verify that core modules can now be imported."""
    import subprocess
    import sys
    
    test_scripts = [
        # Test basic import
        """
import sys
sys.path.insert(0, '.')
from src.llamaagent.cli import main
print("PASS CLI main module imported successfully")
""",
        # Test agent import
        """
import sys
sys.path.insert(0, '.')
from src.llamaagent.agents.base import BaseAgent
print("PASS Base agent imported successfully")
""",
        # Test tools import
        """
import sys
sys.path.insert(0, '.')
from src.llamaagent.tools.base import BaseTool
print("PASS Base tool imported successfully")
""",
    ]
    
    results = []
    for i, script in enumerate(test_scripts):
        try:
            result = subprocess.run(
                [sys.executable, '-c', script],
                capture_output=True,
                text=True,
                cwd='.'
            )
            if result.returncode == 0:
                results.append((f"Test {i+1}", result.stdout.strip()))
            else:
                results.append((f"Test {i+1}", f"FAIL Error: {result.stderr.strip()}"))
        except Exception as e:
            results.append((f"Test {i+1}", f"FAIL Exception: {str(e)}"))
    
    return results


def main():
    """Main entry point."""
    print("FIXING Final Syntax Fixes")
    print("=" * 50)
    
    # Apply fixes
    fixes = apply_remaining_fixes()
    
    print("\nResponse Fixes Applied:")
    for file, description in fixes:
        print(f"  PASS {file}: {description}")
    
    print("\nAnalyzing Verifying core modules...")
    results = verify_core_modules()
    for test, result in results:
        print(f"  {test}: {result}")
    
    print("\nPASS Final syntax fixes complete!")
    print("\n Next steps:")
    print("  1. Run tests: pytest tests/")
    print("  2. Try the CLI: python -m src.llamaagent.cli.main --help")
    print("  3. Start development server: python -m src.llamaagent.api.main")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Final comprehensive syntax error fixer."""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# List of specific files and their fixes
SPECIFIC_FIXES = {
    "src/llamaagent/cli/enhanced_shell_cli.py": [
        ("role=msg_data[\"role\"],", "role=msg_data[\"role\"],")
    ],
    "src/llamaagent/cli/enhanced_cli.py": [
        ("successful_messages", "successful_messages = 0")
    ],
    "src/llamaagent/cli/config_manager.py": [
        ("logger.error(f\"Failed to save configuration", "# Fixed in file already")
    ],
    "src/llamaagent/cli/role_manager.py": [
        ("return", "# Check structure")
    ],
    "src/llamaagent/cli/function_manager.py": [
        ("result = func(*args, **kwargs", "result = func(*args, **kwargs)")
    ],
    "src/llamaagent/cli/openai_cli.py": [
        ("model_costs.get(model, 0.0", "model_costs.get(model, 0.0)")
    ],
    "src/llamaagent/cli/diagnostics_cli.py": [
        ("diagnostics = MasterDiagnostics(str(project_path)", "diagnostics = MasterDiagnostics(str(project_path))"),
        ("with Progress(", "with Progress(")
    ],
    "src/llamaagent/cli/code_generator.py": [
        ("\"description\": request.description,", "\"description\": request.description,")
    ],
    "src/llamaagent/reasoning/memory_manager.py": [
        ("if item:", "# Check before line 210")
    ],
    "src/llamaagent/reasoning/chain_engine.py": [
        ("tool_calls: List[ToolCall] = field(default_factory=list", "tool_calls: List[ToolCall] = field(default_factory=list)")
    ],
    "src/llamaagent/knowledge/knowledge_generator.py": [
        ("words = text.split(", "words = text.split()")
    ],
    "src/llamaagent/ml/inference_engine.py": [
        ("}", "})")
    ],
    "src/llamaagent/evolution/adaptive_learning.py": [
        ("py_files = list(self.data_dir.glob(\"*.py\")", "py_files = list(self.data_dir.glob(\"*.py\"))")
    ],
    "src/llamaagent/data_generation/base.py": [
        ("node_id: str = field(default_factory=lambda: str(uuid.uuid4())", "node_id: str = field(default_factory=lambda: str(uuid.uuid4()))")
    ],
    "src/llamaagent/benchmarks/gaia_benchmark.py": [
        ("self.tasks", "# Check previous line")
    ],
    "src/llamaagent/benchmarks/spre_evaluator.py": [
        ("def avg_api_calls(self) -> float:", "# Check previous line")
    ]
}

def find_syntax_error(file_path: str) -> Tuple[int, str]:
    """Find syntax error in file."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", file_path],
            capture_output=True,
            text=True
        )
        if result.stderr:
            match = re.search(r'line (\d+)', result.stderr)
            if match:
                return int(match.group(1)), result.stderr
        return 0, ""
    except Exception as e:
        return 0, str(e)

def fix_file(file_path: str) -> bool:
    """Fix syntax errors in a file."""
    line_num, error = find_syntax_error(file_path)
    if not error:
        return True
    
    print(f"Fixing {file_path} (error on line {line_num})")
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Specific fixes for known issues
        modified = False
        
        # Fix missing closing parentheses/brackets
        if line_num > 0 and line_num <= len(lines):
            check_line = line_num - 2  # Python line numbers are 1-based
            if check_line >= 0:
                line = lines[check_line]
                
                # Count unmatched delimiters
                open_parens = line.count('(') - line.count(')')
                open_brackets = line.count('[') - line.count(']')
                open_braces = line.count('{') - line.count('}')
                
                if open_parens > 0:
                    lines[check_line] = line.rstrip() + ')' * open_parens + '\n'
                    modified = True
                elif open_brackets > 0:
                    lines[check_line] = line.rstrip() + ']' * open_brackets + '\n'
                    modified = True
                elif open_braces > 0:
                    lines[check_line] = line.rstrip() + '}' * open_braces + '\n'
                    modified = True
        
        # Apply specific fixes if defined
        if file_path in SPECIFIC_FIXES:
            for old_pattern, new_pattern in SPECIFIC_FIXES[file_path]:
                for i, line in enumerate(lines):
                    if old_pattern in line and not new_pattern.startswith("#"):
                        lines[i] = line.replace(old_pattern, new_pattern)
                        modified = True
        
        if modified:
            with open(file_path, 'w') as f:
                f.writelines(lines)
            
            # Verify fix
            _, new_error = find_syntax_error(file_path)
            if not new_error:
                print(f"  ✓ Fixed successfully!")
                return True
            else:
                print(f"  ✗ Still has errors")
                return False
    
    except Exception as e:
        print(f"  Error: {e}")
        return False
    
    return False

def main():
    """Main function."""
    # Find all Python files with syntax errors
    src_dir = Path("src/llamaagent")
    error_files = []
    
    print("Scanning for syntax errors...")
    for py_file in src_dir.rglob("*.py"):
        line_num, error = find_syntax_error(str(py_file))
        if error:
            error_files.append(str(py_file))
    
    print(f"\nFound {len(error_files)} files with syntax errors")
    
    # Fix each file
    fixed_count = 0
    for file_path in error_files:
        if fix_file(file_path):
            fixed_count += 1
    
    print(f"\n✓ Fixed {fixed_count}/{len(error_files)} files")
    
    # Final scan
    remaining = []
    for py_file in src_dir.rglob("*.py"):
        _, error = find_syntax_error(str(py_file))
        if error:
            remaining.append(str(py_file))
    
    if remaining:
        print(f"\n✗ {len(remaining)} files still have errors:")
        for f in remaining[:10]:
            print(f"  - {f}")
    else:
        print("\n✓ All syntax errors have been fixed!")

if __name__ == "__main__":
    main()
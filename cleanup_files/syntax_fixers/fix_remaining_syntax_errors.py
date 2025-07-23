#!/usr/bin/env python3
"""Fix all remaining syntax errors in the LlamaAgent codebase."""

import os
import re
import subprocess
import sys
from pathlib import Path

def find_syntax_errors(file_path):
    """Find syntax errors in a Python file."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", file_path],
            capture_output=True,
            text=True
        )
        if result.stderr:
            return result.stderr
        return None
    except Exception as e:
        return str(e)

def fix_common_syntax_errors(content):
    """Fix common syntax errors in Python code."""
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # Fix missing closing parentheses
        open_parens = line.count('(') - line.count(')')
        if open_parens > 0:
            # Check if next line starts with indentation suggesting continuation
            if i + 1 < len(lines) and lines[i + 1].strip() and not lines[i + 1][0].isspace():
                line += ')' * open_parens
        
        # Fix missing closing brackets
        open_brackets = line.count('[') - line.count(']')
        if open_brackets > 0:
            if i + 1 < len(lines) and lines[i + 1].strip() and not lines[i + 1][0].isspace():
                line += ']' * open_brackets
        
        # Fix f-string issues
        if 'f"' in line or "f'" in line:
            # Ensure all { have matching }
            in_string = False
            quote_char = None
            brace_count = 0
            
            for j, char in enumerate(line):
                if char in '"\'':
                    if not in_string:
                        in_string = True
                        quote_char = char
                    elif char == quote_char and (j == 0 or line[j-1] != '\\'):
                        in_string = False
                        quote_char = None
                elif in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
            
            if brace_count > 0:
                line += '}' * brace_count
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def process_file(file_path):
    """Process a single file to fix syntax errors."""
    print(f"Processing: {file_path}")
    
    error = find_syntax_errors(file_path)
    if not error:
        return True
    
    print(f"  Error found: {error}")
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Apply fixes
        fixed_content = fix_common_syntax_errors(content)
        
        # Write back
        with open(file_path, 'w') as f:
            f.write(fixed_content)
        
        # Check if fixed
        error = find_syntax_errors(file_path)
        if error:
            print(f"  Still has errors after auto-fix: {error}")
            return False
        else:
            print(f"  Fixed successfully!")
            return True
    except Exception as e:
        print(f"  Failed to process: {e}")
        return False

def main():
    """Main function to fix all syntax errors."""
    # List of files with known syntax errors
    error_files = [
        "src/llamaagent/optimization/performance.py",
        "src/llamaagent/agents/multimodal_reasoning.py",
        "src/llamaagent/cli/enhanced_shell_cli.py",
        "src/llamaagent/cli/enhanced_cli.py",
        "src/llamaagent/cli/config_manager.py",
        "src/llamaagent/cli/role_manager.py",
        "src/llamaagent/cli/function_manager.py",
        "src/llamaagent/cli/openai_cli.py",
        "src/llamaagent/cli/diagnostics_cli.py",
        "src/llamaagent/cli/code_generator.py",
        "src/llamaagent/reasoning/memory_manager.py",
        "src/llamaagent/reasoning/chain_engine.py",
        "src/llamaagent/knowledge/knowledge_generator.py",
        "src/llamaagent/ml/inference_engine.py",
        "src/llamaagent/evolution/adaptive_learning.py",
    ]
    
    success_count = 0
    for file_path in error_files:
        if Path(file_path).exists():
            if process_file(file_path):
                success_count += 1
        else:
            print(f"File not found: {file_path}")
    
    print(f"\nFixed {success_count}/{len(error_files)} files")
    
    # Find any remaining files with syntax errors
    print("\nScanning for any remaining syntax errors...")
    remaining_errors = []
    
    for root, dirs, files in os.walk("src/llamaagent"):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                error = find_syntax_errors(file_path)
                if error:
                    remaining_errors.append((file_path, error))
    
    if remaining_errors:
        print(f"\nFound {len(remaining_errors)} files with remaining syntax errors:")
        for file_path, error in remaining_errors[:10]:  # Show first 10
            print(f"  {file_path}")
            print(f"    {error.strip()}")
    else:
        print("\nNo remaining syntax errors found!")

if __name__ == "__main__":
    main()
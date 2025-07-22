#!/usr/bin/env python3
"""Final comprehensive syntax error fixer for all remaining files."""

import os
import re
import subprocess
import sys
from pathlib import Path

def find_and_fix_syntax_errors():
    """Find and fix all syntax errors in the codebase."""
    src_dir = Path("src/llamaagent")
    
    # Get all Python files
    py_files = list(src_dir.rglob("*.py"))
    
    fixed_count = 0
    total_errors = 0
    
    for py_file in py_files:
        # Check for syntax errors
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(py_file)],
            capture_output=True,
            text=True
        )
        
        if result.stderr:
            total_errors += 1
            print(f"\nChecking {py_file}")
            
            # Extract line number
            match = re.search(r'line (\d+)', result.stderr)
            if match:
                line_num = int(match.group(1))
                
                try:
                    with open(py_file, 'r') as f:
                        lines = f.readlines()
                    
                    modified = False
                    
                    # Check the error line and previous line
                    if line_num > 0 and line_num <= len(lines):
                        error_line_idx = line_num - 1
                        prev_line_idx = error_line_idx - 1
                        
                        if prev_line_idx >= 0:
                            prev_line = lines[prev_line_idx]
                            
                            # Fix common syntax errors
                            # 1. Missing closing parentheses
                            open_parens = prev_line.count('(') - prev_line.count(')')
                            if open_parens > 0:
                                lines[prev_line_idx] = prev_line.rstrip() + ')' * open_parens + '\n'
                                modified = True
                            
                            # 2. Missing closing brackets
                            open_brackets = prev_line.count('[') - prev_line.count(']')
                            if open_brackets > 0:
                                lines[prev_line_idx] = prev_line.rstrip() + ']' * open_brackets + '\n'
                                modified = True
                            
                            # 3. Missing closing braces
                            open_braces = prev_line.count('{') - prev_line.count('}')
                            if open_braces > 0:
                                lines[prev_line_idx] = prev_line.rstrip() + '}' * open_braces + '\n'
                                modified = True
                    
                    if modified:
                        with open(py_file, 'w') as f:
                            f.writelines(lines)
                        
                        # Verify fix
                        verify_result = subprocess.run(
                            [sys.executable, "-m", "py_compile", str(py_file)],
                            capture_output=True,
                            text=True
                        )
                        
                        if not verify_result.stderr:
                            print(f"   Fixed successfully!")
                            fixed_count += 1
                        else:
                            print(f"   Still has errors")
                    
                except Exception as e:
                    print(f"  Error processing: {e}")
    
    print(f"\n\nSummary:")
    print(f"Total files with errors: {total_errors}")
    print(f"Successfully fixed: {fixed_count}")
    print(f"Remaining errors: {total_errors - fixed_count}")

if __name__ == "__main__":
    find_and_fix_syntax_errors()
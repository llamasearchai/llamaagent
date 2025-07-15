#!/usr/bin/env python3
"""
Systematic syntax error fixer for the LlamaAgent codebase.
"""

import ast
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def find_python_files(root_dir: str) -> List[Path]:
    """Find all Python files in the directory."""
    python_files = []
    for root, dirs, files in os.walk(root_dir):
        # Skip __pycache__ directories
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        
        for file in files:
            if file.endswith(".py"):
                python_files.append(Path(root) / file)
    
    return python_files


def check_syntax(file_path: Path) -> Tuple[bool, str]:
    """Check if a Python file has syntax errors."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to parse the file
        ast.parse(content)
        return True, ""
    except SyntaxError as e:
        return False, f"{e.msg} at line {e.lineno}"
    except Exception as e:
        return False, str(e)


def compile_check(file_path: Path) -> Tuple[bool, str]:
    """Use py_compile to check for syntax errors."""
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(file_path)],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        return False, result.stderr
    return True, ""


def main():
    """Main function to find and report syntax errors."""
    root_dir = "src/llamaagent"
    
    print(f"Scanning {root_dir} for Python files...")
    python_files = find_python_files(root_dir)
    print(f"Found {len(python_files)} Python files")
    
    errors = []
    
    for file_path in python_files:
        # First try AST parsing
        ast_ok, ast_error = check_syntax(file_path)
        
        if not ast_ok:
            # Double-check with py_compile
            compile_ok, compile_error = compile_check(file_path)
            
            if not compile_ok:
                errors.append({
                    'file': str(file_path),
                    'ast_error': ast_error,
                    'compile_error': compile_error
                })
                print(f"FAIL {file_path}")
                print(f"   AST Error: {ast_error}")
                if compile_error:
                    # Extract just the error message
                    error_lines = compile_error.strip().split('\n')
                    for line in error_lines:
                        if 'SyntaxError:' in line:
                            print(f"   Compile Error: {line.strip()}")
            else:
                print(f"✓ {file_path}")
        else:
            print(f"✓ {file_path}")
    
    print(f"\n\nSummary:")
    print(f"Total files: {len(python_files)}")
    print(f"Files with errors: {len(errors)}")
    print(f"Files without errors: {len(python_files) - len(errors)}")
    
    if errors:
        print(f"\n\nFiles with syntax errors:")
        for error in errors:
            print(f"  - {error['file']}")
        
        # Write errors to a file for further processing
        with open('syntax_errors.txt', 'w') as f:
            for error in errors:
                f.write(f"{error['file']}|{error['ast_error']}|{error['compile_error']}\n")
        
        print(f"\nError details written to syntax_errors.txt")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
import os
import ast
import sys
from pathlib import Path

def check_syntax(file_path):
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)

def find_python_files(directory):
    """Find all Python files in a directory."""
    path = Path(directory)
    return list(path.rglob('*.py'))

def main():
    src_dir = Path('src')
    if not src_dir.exists():
        print("Error: src directory not found")
        return
    
    python_files = find_python_files(src_dir)
    syntax_errors = []
    
    print(f"Checking {len(python_files)} Python files in src/...")
    print("-" * 80)
    
    for file_path in sorted(python_files):
        is_valid, error = check_syntax(file_path)
        if not is_valid:
            syntax_errors.append((str(file_path), error))
            print(f"FAIL SYNTAX ERROR: {file_path}")
            print(f"   Error: {error}")
            print()
    
    print("-" * 80)
    print(f"\nSummary: Found {len(syntax_errors)} files with syntax errors out of {len(python_files)} total files")
    
    if syntax_errors:
        print("\nFiles with syntax errors:")
        for file_path, error in syntax_errors:
            print(f"  - {file_path}")

if __name__ == "__main__":
    main()
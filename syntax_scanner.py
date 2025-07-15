#!/usr/bin/env python3
"""
Syntax error scanner for Python files.
Scans all Python files in a directory and reports syntax errors.
"""

import ast
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

def scan_file(filepath: Path) -> Dict[str, Any]:
    """Scan a single file for syntax errors."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        try:
            ast.parse(content)
            return {
                'path': str(filepath),
                'has_error': False,
                'error': None
            }
        except SyntaxError as e:
            return {
                'path': str(filepath),
                'has_error': True,
                'error': {
                    'msg': str(e.msg),
                    'line': e.lineno,
                    'offset': e.offset,
                    'text': e.text
                }
            }
    except Exception as e:
        return {
            'path': str(filepath),
            'has_error': True,
            'error': {
                'msg': f"Failed to read file: {str(e)}",
                'line': None,
                'offset': None,
                'text': None
            }
        }

def scan_directory(directory: str) -> List[Dict[str, Any]]:
    """Scan all Python files in a directory for syntax errors."""
    errors = []
    total_files = 0
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = Path(root) / file
                total_files += 1
                result = scan_file(filepath)
                if result['has_error']:
                    errors.append(result)
    
    return errors, total_files

def main():
    """Main function to scan for syntax errors."""
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = 'src'
    
    print(f"Scanning {directory} for syntax errors...")
    errors, total_files = scan_directory(directory)
    
    print(f"\nScanned {total_files} Python files")
    print(f"Found {len(errors)} files with syntax errors\n")
    
    if errors:
        for error in errors:
            print(f"File: {error['path']}")
            if error['error']['line']:
                print(f"  Line {error['error']['line']}: {error['error']['msg']}")
                if error['error']['text']:
                    print(f"  Code: {error['error']['text'].strip()}")
            else:
                print(f"  Error: {error['error']['msg']}")
            print()

if __name__ == "__main__":
    main()
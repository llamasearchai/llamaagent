#!/usr/bin/env python3
"""
Comprehensive Fix for Malformed .append() Calls

This script systematically fixes all malformed .append() calls across the codebase
where the parentheses are incorrectly structured.

Common patterns to fix:
- list.append()\n    item\n)
- list.append()\n        item\n    )

Author: Comprehensive System Fix
"""

import os
import re
import glob
from pathlib import Path

def fix_malformed_append_calls():
    """Fix all malformed .append() calls in Python files."""
    
    # Find all Python files in the src directory
    python_files = glob.glob("src/**/*.py", recursive=True)
    
    fixed_files = []
    total_fixes = 0
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Pattern 1: .append()\n    item\n)
            # Replace with: .append(\n    item\n)
            pattern1 = r'\.append\(\)\s*\n(\s+)([^)]+)\n\s*\)'
            content = re.sub(pattern1, r'.append(\n\1\2\n\1)', content, flags=re.MULTILINE)
            
            # Pattern 2: .append()\n        item\n    )
            # Replace with: .append(\n        item\n    )
            pattern2 = r'\.append\(\)\s*\n(\s+)([^)]+)\n(\s*)\)'
            content = re.sub(pattern2, r'.append(\n\1\2\n\3)', content, flags=re.MULTILINE)
            
            # Pattern 3: More complex multi-line patterns
            # Handle cases where there are multiple lines between append() and )
            pattern3 = r'\.append\(\)\s*\n((?:\s+[^)]+\n)*)\s*\)'
            content = re.sub(pattern3, r'.append(\n\1)', content, flags=re.MULTILINE)
            
            # Check if any changes were made
            if content != original_content:
                # Write back the fixed content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                # Count the number of fixes made
                fixes_made = original_content.count('.append()') - content.count('.append()')
                total_fixes += fixes_made
                fixed_files.append((file_path, fixes_made))
                
                print(f"PASS Fixed {fixes_made} malformed .append() calls in {file_path}")
            
        except Exception as e:
            print(f"FAIL Error processing {file_path}: {e}")
    
    # Summary
    print(f"\nTARGET COMPREHENSIVE APPEND FIX SUMMARY")
    print(f"Files processed: {len(python_files)}")
    print(f"Files fixed: {len(fixed_files)}")
    print(f"Total fixes applied: {total_fixes}")
    
    if fixed_files:
        print(f"\nDetailed fixes:")
        for file_path, fixes in fixed_files:
            print(f"  - {file_path}: {fixes} fixes")
    
    return len(fixed_files), total_fixes

if __name__ == "__main__":
    print("Starting Comprehensive .append() Fix...")
    print("=" * 60)
    
    fixed_files, total_fixes = fix_malformed_append_calls()
    
    print("=" * 60)
    print(f"PASS Comprehensive fix completed!")
    print(f"   Fixed {total_fixes} malformed .append() calls in {fixed_files} files") 
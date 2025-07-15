#!/usr/bin/env python3
"""
Targeted syntax fixer that reads the error report and fixes specific issues.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Load the error report
def load_error_report(report_path: str = "syntax_fix_report.json") -> Dict:
    """Load the error report."""
    with open(report_path, 'r') as f:
        return json.load(f)

def fix_specific_error(filepath: Path, line_num: int, error_msg: str) -> bool:
    """Fix a specific error in a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if line_num > len(lines) or line_num < 1:
            return False
        
        # Adjust for 0-based indexing
        idx = line_num - 1
        original_line = lines[idx]
        fixed_line = original_line
        
        # Fix based on error type
        if "closing parenthesis '}' does not match opening parenthesis '('" in error_msg:
            # F-string error - missing closing parenthesis
            # Count parentheses and add missing ones
            open_parens = original_line.count('(')
            close_parens = original_line.count(')')
            if open_parens > close_parens:
                # Add missing closing parentheses before the f-string closing brace
                fixed_line = re.sub(r'(\{[^}]*)\}', lambda m: m.group(1) + ')' * (open_parens - close_parens) + '}', original_line)
        
        elif "unmatched ')'" in error_msg:
            # Too many closing parentheses
            # Remove extra closing parentheses
            fixed_line = re.sub(r'\)+(\s*[,;:\n])', r')\1', original_line)
        
        elif "invalid syntax" in error_msg:
            # Check for common patterns
            # Extra comma before closing parenthesis
            if re.search(r',\s*\)', original_line):
                fixed_line = re.sub(r',\s*\)', ')', original_line)
            
            # Missing colon after control structures
            elif re.match(r'\s*(if|for|while|def|class|elif|else|try|except|finally)\s+.*[^:]$', original_line):
                fixed_line = original_line.rstrip() + ':\n'
            
            # For loop in wrong position (inside dict/list)
            elif 'for ' in original_line and '{' in original_line:
                # Try to fix dict comprehension syntax
                fixed_line = re.sub(r'\{([^:]+):\s*([^}]+)\s+for\s+', r'{{\1: \2 for ', original_line)
        
        elif "unexpected indent" in error_msg:
            # Fix indentation - assume 4 spaces
            stripped = original_line.lstrip()
            if stripped:
                # Guess correct indentation based on previous non-empty line
                correct_indent = 0
                for i in range(idx - 1, -1, -1):
                    if lines[i].strip():
                        prev_indent = len(lines[i]) - len(lines[i].lstrip())
                        if lines[i].rstrip().endswith(':'):
                            correct_indent = prev_indent + 4
                        else:
                            correct_indent = prev_indent
                        break
                fixed_line = ' ' * correct_indent + stripped
        
        # Write back if changed
        if fixed_line != original_line:
            lines[idx] = fixed_line
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            print(f"  PASS Fixed line {line_num}: {error_msg}")
            return True
        
        return False
        
    except Exception as e:
        print(f"  FAIL Error fixing line {line_num}: {str(e)}")
        return False

def main():
    """Main function to fix all reported errors."""
    report = load_error_report()
    
    failed_files = report.get("failed_files", [])
    file_errors = report.get("file_errors", {})
    
    print(f"Found {len(failed_files)} files with syntax errors")
    print("=" * 60)
    
    fixed_count = 0
    total_errors = 0
    
    for filename in failed_files:
        filepath = Path(filename)
        if not filepath.exists():
            print(f"FAIL File not found: {filepath}")
            continue
        
        print(f"\nProcessing: {filepath}")
        
        # Get errors for this file
        file_key = filepath.name
        if file_key in file_errors:
            errors = file_errors[file_key]
            for error in errors:
                total_errors += 1
                line_num = error.get("line", 0)
                error_msg = error.get("error", "")
                
                if fix_specific_error(filepath, line_num, error_msg):
                    fixed_count += 1
    
    print("\n" + "=" * 60)
    print(f"Fixed {fixed_count} out of {total_errors} errors")
    
    # Now check which files still have errors
    import ast
    still_failing = []
    
    print("\nValidating fixes...")
    for filename in failed_files:
        filepath = Path(filename)
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    ast.parse(f.read())
                print(f"  PASS {filepath.name} - syntax valid")
            except SyntaxError as e:
                still_failing.append((filepath, str(e)))
                print(f"  FAIL {filepath.name} - still has errors: {e}")
    
    if still_failing:
        print(f"\n{len(still_failing)} files still have syntax errors")
        with open("remaining_errors.txt", "w") as f:
            for filepath, error in still_failing:
                f.write(f"{filepath}: {error}\n")
        print("Remaining errors saved to: remaining_errors.txt")
    else:
        print("\nSUCCESS All syntax errors fixed!")

if __name__ == "__main__":
    main()
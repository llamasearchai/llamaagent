#!/usr/bin/env python3
"""Fix remaining syntax errors efficiently."""

import re
import subprocess
import sys
from pathlib import Path

def fix_syntax_error(file_path, line_num, error_msg):
    """Attempt to fix a syntax error based on the error message."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        if line_num <= 0 or line_num > len(lines):
            return False
        
        idx = line_num - 1
        line = lines[idx]
        original = line
        fixed = False
        
        # Pattern 1: Missing closing parenthesis/bracket
        if "unexpected EOF" in error_msg or "invalid syntax" in error_msg:
            open_count = line.count('(') + line.count('[') + line.count('{')
            close_count = line.count(')') + line.count(']') + line.count('}')
            
            if open_count > close_count:
                # Add missing closing delimiters
                missing = open_count - close_count
                line = line.rstrip()
                
                # Determine which delimiter to add
                if line.count('(') > line.count(')'):
                    line += ')' * (line.count('(') - line.count(')'))
                if line.count('[') > line.count(']'):
                    line += ']' * (line.count('[') - line.count(']'))
                if line.count('{') > line.count('}'):
                    line += '}' * (line.count('{') - line.count('}'))
                    
                line += '\n'
                fixed = True
        
        # Pattern 2: Extra comma before closing delimiter
        if re.search(r',\s*[\)\]\}]', line):
            line = re.sub(r',(\s*[\)\]\}])', r'\1', line)
            fixed = True
        
        # Pattern 3: Colon at end of list comprehension
        if 'for ' in line and line.strip().endswith(':') and '[' in ''.join(lines[max(0, idx-5):idx]):
            line = line.rstrip()[:-1] + '\n'
            fixed = True
        
        # Pattern 4: Missing comma in dict/list
        if idx > 0 and not lines[idx-1].rstrip().endswith(',') and line.strip().startswith('"'):
            if ':' in line:  # Likely a dict entry
                lines[idx-1] = lines[idx-1].rstrip() + ',\n'
                fixed = True
        
        if fixed and line != original:
            lines[idx] = line
            with open(file_path, 'w') as f:
                f.writelines(lines)
            return True
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return False

def main():
    """Main function to fix all syntax errors."""
    src_dir = Path("/Users/o2/Desktop/llamaagent/src")
    fixed_count = 0
    
    # Get all Python files with syntax errors
    for py_file in src_dir.rglob("*.py"):
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(py_file)],
            capture_output=True,
            text=True
        )
        
        if result.stderr and "SyntaxError" in result.stderr:
            # Parse error
            lines = result.stderr.strip().split('\n')
            for line in lines:
                if "line" in line and "File" in line:
                    try:
                        # Extract line number
                        match = re.search(r'line (\d+)', line)
                        if match:
                            line_num = int(match.group(1))
                            
                            print(f"Attempting to fix {py_file}:{line_num}")
                            if fix_syntax_error(py_file, line_num, result.stderr):
                                fixed_count += 1
                                print(f"   Fixed!")
                            else:
                                print(f"   Could not auto-fix")
                            break
                    except Exception as e:
                        print(f"  Error: {e}")
    
    print(f"\nFixed {fixed_count} syntax errors")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Fix syntax errors in ReactAgent file.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import re

def fix_react_syntax():
    """Fix syntax errors in the ReactAgent file."""
    
    file_path = "src/llamaagent/agents/react.py"
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix missing closing parenthesis
    content = re.sub(
        r'result_str = str\(int\(result\)(?!\))',
        'result_str = str(int(result))',
        content
    )
    
    # Fix missing closing parenthesis in add_val
    content = re.sub(
        r'add_val = float\(add_match\.group\(1\)(?!\))',
        'add_val = float(add_match.group(1))',
        content
    )
    
    # Fix the return statement
    content = re.sub(
        r'return str\(int\(result\) if result\.is_integer\(\) else str\(result\)',
        'return str(int(result) if result.is_integer() else result)',
        content
    )
    
    # Fix the sum expression
    content = re.sub(
        r'total_chars = sum\(len\(str\(item\.get\("data", ""\)\) for item in self\.trace\)',
        'total_chars = sum(len(str(item.get("data", "")) for item in self.trace)',
        content
    )
    
    # Fix any remaining unmatched parentheses
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Check for common patterns that need fixing
        if 'float(add_match.group(1)' in line and not line.endswith(')'):
            line = line.rstrip() + ')'
        elif 'str(int(result)' in line and line.count('(') > line.count(')'):
            line = line.rstrip() + ')'
        elif 'sum(len(str(item.get("data", "")' in line and not line.endswith(')'):
            line = line.rstrip() + ')'
        
        fixed_lines.append(line)
    
    content = '\n'.join(fixed_lines)
    
    # Write the fixed content back
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("PASS Fixed syntax errors in ReactAgent file")

if __name__ == "__main__":
    fix_react_syntax() 
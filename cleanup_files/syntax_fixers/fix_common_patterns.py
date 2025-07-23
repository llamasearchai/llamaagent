#!/usr/bin/env python3
"""Fix common syntax error patterns."""

import re
from pathlib import Path


def fix_file(file_path, line_num):
    """Fix syntax errors in a specific file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    if line_num <= 0 or line_num > len(lines):
        return False

    # Get the problematic line (0-indexed)
    idx = line_num - 1
    line = lines[idx]
    original_line = line

    # Common fixes
    fixed = False

    # Fix extra comma before closing parenthesis/bracket/brace
    if re.search(r',\s*[\)\]\}]', line):
        line = re.sub(r',(\s*[\)\]\}])', r'\1', line)
        fixed = True

    # Fix missing closing parenthesis
    open_parens = line.count('(')
    close_parens = line.count(')')
    if open_parens > close_parens and not line.strip().endswith(':'):
        line = line.rstrip() + ')' * (open_parens - close_parens) + '\n'
        fixed = True

    # Fix extra closing parenthesis
    if close_parens > open_parens:
        # Remove extra closing parens from the end
        diff = close_parens - open_parens
        line = line.rstrip()
        for _ in range(diff):
            if line.endswith(')'):
                line = line[:-1]
        line += '\n'
        fixed = True

    # Fix colon at end of list comprehension
    if (
        'for ' in line
        and line.strip().endswith(':')
        and '[' in ''.join(lines[max(0, idx - 5) : idx])
    ):
        line = line.rstrip()[:-1] + '\n'
        fixed = True

    # Fix missing comma in dictionary/list
    if (
        idx > 0
        and not lines[idx - 1].rstrip().endswith(',')
        and not lines[idx - 1].rstrip().endswith('{')
        and not lines[idx - 1].rstrip().endswith('[')
    ):
        if line.strip().startswith('"') and ':' in line:
            lines[idx - 1] = lines[idx - 1].rstrip() + ',\n'
            fixed = True

    if fixed and line != original_line:
        lines[idx] = line
        with open(file_path, 'w') as f:
            f.writelines(lines)
        return True

    return False


# Process specific files
files_to_fix = [
    ("/Users/o2/Desktop/llamaagent/src/llamaagent/api/production_app.py", 154),
    ("/Users/o2/Desktop/llamaagent/src/llamaagent/api/shell_endpoints.py", 486),
    ("/Users/o2/Desktop/llamaagent/src/llamaagent/api/simon_ecosystem_api.py", 355),
    ("/Users/o2/Desktop/llamaagent/src/llamaagent/benchmarks/gaia_benchmark.py", 127),
    ("/Users/o2/Desktop/llamaagent/src/llamaagent/benchmarks/spre_evaluator.py", 57),
]

for file_path, line_num in files_to_fix:
    print(f"Attempting to fix {file_path}:{line_num}")
    if fix_file(file_path, line_num):
        print(f"  Fixed!")
    else:
        print(f"  Could not auto-fix, manual intervention needed")

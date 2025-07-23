#!/usr/bin/env python3
"""Batch syntax error fixer."""

import subprocess
import sys
from pathlib import Path


def get_syntax_errors():
    """Get all Python files with syntax errors."""
    src_dir = Path("/Users/o2/Desktop/llamaagent/src")
    errors = []

    for py_file in src_dir.rglob("*.py"):
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(py_file)],
            capture_output=True,
            text=True,
        )
        if result.stderr:
            # Parse the error to get file and line number
            if "SyntaxError" in result.stderr:
                lines = result.stderr.strip().split('\n')
                for i, line in enumerate(lines):
                    if "File" in line and ".py" in line:
                        # Extract file path
                        file_path = line.split('"')[1]
                        # Look for line number
                        if "line" in line:
                            try:
                                line_part = line.split("line")[1].strip()
                                line_num = int(line_part.split()[0].rstrip(','))
                                errors.append((file_path, line_num, result.stderr))
                            except:
                                # If we can't parse line number, still include the file
                                errors.append((file_path, 0, result.stderr))
                        break

    return sorted(errors)


if __name__ == "__main__":
    errors = get_syntax_errors()
    print(f"Found {len(errors)} files with syntax errors:")
    for file_path, line_num, error in errors:
        print(f"{file_path}:{line_num}")
        # Print a snippet of the error
        error_lines = error.split('\n')
        for line in error_lines:
            if "SyntaxError:" in line:
                print(f"  {line.strip()}")
                break

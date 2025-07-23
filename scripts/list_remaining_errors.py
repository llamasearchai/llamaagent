#!/usr/bin/env python3
"""List all remaining syntax errors with details."""

import subprocess
import sys
from pathlib import Path

src_dir = Path("/Users/o2/Desktop/llamaagent/src")
errors = []

for py_file in src_dir.rglob("*.py"):
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(py_file)],
        capture_output=True,
        text=True,
    )

    if result.stderr and "SyntaxError" in result.stderr:
        print(f"\n{'='*60}")
        print(f"File: {py_file}")
        print(f"Error: {result.stderr.strip()}")
        print('=' * 60)

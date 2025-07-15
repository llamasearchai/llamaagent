#!/usr/bin/env python3
"""
Priority-based syntax fixer for critical LlamaAgent files.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# Priority files to fix first (most important for the framework)
PRIORITY_FILES = [
    # Core agent files
    "src/llamaagent/agents/react.py",
    "src/llamaagent/agents/base.py",
    "src/llamaagent/agents/reasoning_chains.py",
    "src/llamaagent/agents/multimodal_reasoning.py",
    
    # CLI files
    "src/llamaagent/cli/main.py",
    "src/llamaagent/cli/interactive.py",
    "src/llamaagent/cli/enhanced_cli.py",
    
    # Cache files
    "src/llamaagent/cache/result_cache.py",
    "src/llamaagent/cache/cache_manager.py",
    "src/llamaagent/cache/llm_cache.py",
    
    # Integration files
    "src/llamaagent/integration/_openai_stub.py",
    "src/llamaagent/integration/simon_tools.py",
    
    # Security files
    "src/llamaagent/security/rate_limiter.py",
    "src/llamaagent/security/validator.py",
]


def check_file_syntax(file_path: Path) -> Tuple[bool, str]:
    """Check if a file has syntax errors."""
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(file_path)],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        return False, result.stderr
    return True, ""


def main():
    """Main function to check priority files."""
    print("Checking priority files for syntax errors...\n")
    
    files_with_errors = []
    files_without_errors = []
    
    for file_path in PRIORITY_FILES:
        path = Path(file_path)
        if not path.exists():
            print(f"⚠️  {file_path} - File not found")
            continue
            
        is_valid, error = check_file_syntax(path)
        
        if is_valid:
            print(f"PASS {file_path}")
            files_without_errors.append(file_path)
        else:
            print(f"FAIL {file_path}")
            # Extract error line
            for line in error.split('\n'):
                if 'line' in line:
                    print(f"   {line.strip()}")
            files_with_errors.append(file_path)
    
    print(f"\n\nSummary:")
    print(f"Total priority files: {len(PRIORITY_FILES)}")
    print(f"Files without errors: {len(files_without_errors)}")
    print(f"Files with errors: {len(files_with_errors)}")
    
    if files_with_errors:
        print(f"\nFiles that need fixing:")
        for file in files_with_errors:
            print(f"  - {file}")
    
    return 0 if not files_with_errors else 1


if __name__ == "__main__":
    sys.exit(main())
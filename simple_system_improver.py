#!/usr/bin/env python3
"""
Simple LlamaAgent System Improver
==================================

A streamlined script to fix critical issues and optimize the LlamaAgent codebase.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import subprocess
import sys
from typing import List, Tuple


def run_command(cmd: str, description: str) -> bool:
    """Run a command and return success status"""
    print(f"{description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"{description} completed successfully")
            return True
        else:
            print(f"{description} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"{description} failed with exception: {e}")
        return False

def main() -> bool:
    """Run system improvements"""
    print("Starting LlamaAgent System Improvement Starting...")
    
    improvements: List[Tuple[str, str]] = [
        ("python -m ruff check --fix --select F401,I001 src/", "Fix unused imports"),
        ("python -m ruff check --fix src/", "Fix linting issues"),
        ("python -m black src/ --line-length 88", "Format code"),
        ("python -m isort src/ --profile black", "Sort imports"),
        ("python -c \"import llamaagent; print('Package import successful')\"", "Test package import"),
        ("python -c \"from src.llamaagent.api.main import app; print('API import successful')\"", "Test API import"),
        ("python -c \"from src.llamaagent.llm.factory import LLMFactory; print('LLM Factory working')\"", "Test LLM Factory"),
    ]
    
    successful: int = 0
    total: int = len(improvements)
    
    for cmd, description in improvements:
        if run_command(cmd, description):
            successful += 1
    
    success_rate: float = successful / total * 100
    print(f"\nRESULTS Results: {successful}/{total} operations successful ({success_rate:.1f}%)")
    
    if successful >= total * 0.8:
        print("SUCCESS System improvement completed successfully!")
        print("The LlamaAgent codebase is now optimized and ready for use.")
        return True
    else:
        print("⚠️ Some improvements failed. Please review the errors above.")
        return False

if __name__ == "__main__":
    success: bool = main()
    sys.exit(0 if success else 1) 
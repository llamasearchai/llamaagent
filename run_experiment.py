#!/usr/bin/env python3
"""
LlamaAgent Research Experiment Launcher

Author: Nik Jois <nikjois@llamasearch.ai>

This script provides a complete walkthrough of the LlamaAgent research experiment,
demonstrating all capabilities with an interactive command-line interface.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from llamaagent.cli.interactive import run_interactive_experiment
except ImportError as e:
    print(f"Error importing LlamaAgent: {e}")
    print("Make sure you have installed the package:")
    print("pip install -e .")
    sys.exit(1)


def main():
    """Main launcher function."""
    print("LlamaAgent Research Experiment Launcher")
    print("=" * 50)
    print()
    
    try:
        # Run the interactive experiment
        asyncio.run(run_interactive_experiment())
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user. Goodbye! 👋")
    except Exception as e:
        print(f"\nError running experiment: {e}")
        print("Please check your installation and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main() 
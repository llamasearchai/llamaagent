#!/usr/bin/env python3
"""
LlamaAgent Enhanced CLI - Main Entry Point

A feature-rich command-line interface for interacting with LlamaAgent,
complete with progress bars, animations, and a beautiful user experience.
"""

import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llamaagent.cli.enhanced_cli import main

if __name__ == "__main__":
    main()

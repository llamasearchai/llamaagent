#!/usr/bin/env python3
"""
Demo script for the enhanced LlamaAgent CLI with animations and progress bars.

This demonstrates the beautiful, feature-rich command-line interface.
"""

import os
import sys

# Set up the environment to use the enhanced CLI
os.environ["LLAMAAGENT_ENHANCED_CLI"] = "true"

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print(""")
╔══════════════════════════════════════════════════════════════════════════════╗
║                    LlamaAgent LlamaAgent Enhanced CLI Demo LlamaAgent                         ║
║                                                                              ║
║  This demo showcases the enhanced CLI with:                                 ║
║  • Beautiful ASCII llama animations                                          ║
║  • Real-time progress bars                                                   ║
║  • Interactive command interface                                             ║
║  • Rich formatting and colors                                                ║
║                                                                              ║
║  To run the enhanced CLI:                                                    ║
║  1. Direct: python llamaagent_cli.py                                        ║
║  2. Module: python -m llamaagent enhanced                                    ║
║  3. With options: python -m llamaagent enhanced --provider openai           ║
║                                                                              ║
║  Available commands in the CLI:                                              ║
║  • /help    - Show available commands                                        ║
║  • /status  - Show system status                                             ║
║  • /stats   - Show usage statistics with visual charts                       ║
║  • /history - Show conversation history                                      ║
║  • /config  - Show current configuration                                     ║
║  • /exit    - Exit (with goodbye animation)                                 ║
║                                                                              ║
║  The CLI features:                                                           ║
║  • Animated llama that changes based on system state                        ║
║  • Progress tracking for all operations                                      ║
║  • Beautiful error handling with sad llama                                   ║
║  • Usage statistics and performance metrics                                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Press Enter to launch the enhanced CLI...
""")

input()

# Launch the enhanced CLI
from llamaagent.cli.enhanced_cli import main

main()
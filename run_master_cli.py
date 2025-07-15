#!/usr/bin/env python3
"""
LlamaAgent Master CLI Runner
============================

Simple runner script to test the comprehensive master CLI system.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from llamaagent.cli.master_cli_enhanced import main
    print("LlamaAgent Starting LlamaAgent Enhanced Master CLI...")
    asyncio.run(main())
except ImportError as e:
    print(f"Import error: {e}")
    print("Falling back to standalone master CLI...")
    
    # Try standalone version
    try:
        import subprocess
        subprocess.run([sys.executable, "llamaagent_master_cli.py"])
    except Exception as e2:
        print(f"Failed to run standalone CLI: {e2}")
        sys.exit(1)
except KeyboardInterrupt:
    print("\nGoodbye!")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1) 
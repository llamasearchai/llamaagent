#!/usr/bin/env python3
"""Test script for OpenAI Comprehensive API."""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from llamaagent.api.openai_comprehensive_api import app
    from llamaagent.integration.openai_comprehensive import OpenAIModelType
    from llamaagent.tools.openai_tools import OPENAI_TOOLS

    print(" Successfully imported API modules")

    # List available model types
    print("\nAvailable model types:")
    for model_type in OpenAIModelType:
        print(f"  - {model_type.value}")

    # List available tools
    print("\nAvailable tools:")
    for tool_name in OPENAI_TOOLS:
        print(f"  - {tool_name}")

    print("\nAPI endpoints:")
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            print(f"  - {', '.join(route.methods)} {route.path}")

    print("\n API structure verified successfully!")

except ImportError as e:
    print(f" Import error: {e}")
    import traceback

    traceback.print_exc()
except Exception as e:
    print(f" Error: {e}")
    import traceback

    traceback.print_exc()

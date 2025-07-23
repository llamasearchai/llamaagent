#!/usr/bin/env python3
"""
Comprehensive syntax error fixer for llamaagent codebase.
This script fixes the most critical syntax errors preventing compilation.
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def read_file_safe(file_path: str) -> str:
    """Safely read a file with error handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


def write_file_safe(file_path: str, content: str) -> bool:
    """Safely write a file with error handling."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"Error writing {file_path}: {e}")
        return False


def fix_openai_stub_file():
    """Fix the _openai_stub.py file with incomplete mock replacements."""
    file_path = "src/llamaagent/integration/_openai_stub.py"
    print(f"Fixing {file_path}...")

    content = read_file_safe(file_path)
    if not content:
        return

    # Fix the incomplete mock assignments
    fixes = [
        (
            r'self\.chat = # TODO: Replace mock usage',
            'self.chat = type("Chat", (), {})()',
        ),
        (
            r'mock_openai = # TODO: Replace mock usage',
            'mock_openai = type("MockOpenAI", (), {})()',
        ),
    ]

    for pattern, replacement in fixes:
        content = re.sub(pattern, replacement, content)

    write_file_safe(file_path, content)
    print(f"Fixed {file_path}")


def fix_cohere_provider_file():
    """Fix the cohere_provider.py file with syntax errors."""
    file_path = "src/llamaagent/llm/providers/cohere_provider.py"
    print(f"Fixing {file_path}...")

    # This file appears to be corrupted, let's create a minimal working version
    content = '''"""
Cohere LLM provider implementation.
"""

import asyncio
import os
from typing import Any, Dict, List, Optional, Union

import httpx
from pydantic import BaseModel, Field

from ..base import BaseLLMProvider
from ..messages import LLMMessage, LLMResponse


class CohereConfig(BaseModel):
    """Configuration for Cohere provider."""

    api_key: str = Field(..., description="Cohere API key")
    base_url: str = Field(
        default="https://api.cohere.ai/v1",
        description="Cohere API base URL"
    )
    timeout: int = Field(default=60, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries")


class CohereProvider(BaseLLMProvider):
    """Cohere LLM provider implementation."""

    def __init__(self, config: Optional[CohereConfig] = None) -> None:
        if config is None:
            config = CohereConfig(api_key=os.getenv("COHERE_API_KEY", ""))

        self.config = config
        self.client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=config.timeout,
            headers={"Authorization": f"Bearer {config.api_key}"}
        )

        # Available models
        self.available_models = [
            "command",
            "command-light",
            "command-nightly",
            "command-light-nightly"
        ]

    async def generate_response(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Cohere."""

        # Use default model if none specified
        if model is None:
            model = "command"

        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        try:
            response = await self._make_request("/generate", payload)

            if "message" in response and response.get("message"):
                raise Exception(f"Cohere API error: {response['message']}")

            # Extract response content
            generations = response.get("generations", [])

            if not generations:
                raise Exception("No generations returned from Cohere API")

            content = generations[0].get("text", "")

            # Calculate tokens (approximate)
            input_tokens = len(prompt.split())
            output_tokens = len(content.split())
            total_tokens = input_tokens + output_tokens

            return LLMResponse(
                content=content,
                provider="cohere",
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens
            )

        except Exception as e:
            return LLMResponse(
                content=f"Error occurred: {str(e)}",
                provider="cohere",
                model=model,
                error=str(e)
            )

    async def complete(
        self,
        messages: List[LLMMessage],
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Complete using Cohere API with messages interface."""

        # Convert messages to a single prompt for Cohere
        prompt_parts = []
        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")

        prompt = "\\n".join(prompt_parts)
        if not prompt_parts or prompt_parts[-1].startswith("User:"):
            prompt += "\\nAssistant:"

        # Use the existing generate_response method
        return await self.generate_response(prompt=prompt, model=model, **kwargs)

    async def _make_request(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make HTTP request to Cohere API."""

        for attempt in range(self.config.max_retries):
            try:
                if data:
                    response = await self.client.post(endpoint, json=data)
                else:
                    response = await self.client.get(endpoint)

                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                if attempt == self.config.max_retries - 1:
                    raise Exception(f"HTTP {e.response.status_code}: {e.response.text}")
                await asyncio.sleep(2**attempt)  # Exponential backoff

            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise e
                await asyncio.sleep(2**attempt)

        # This should never be reached due to the exceptions above
        raise Exception("Request failed after all retries")

    async def health_check(self) -> bool:
        """Check if Cohere API is accessible."""
        try:
            # Make a simple request to check API health
            test_payload = {
                "model": "command",
                "prompt": "Test",
                "max_tokens": 1
            }
            await self._make_request("/generate", test_payload)
            return True
        except Exception:
            return False

    async def list_models(self) -> List[str]:
        """List available models from Cohere."""
        return self.available_models.copy()

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

    def __del__(self) -> None:
        """Cleanup on deletion."""
        try:
            asyncio.create_task(self.close())
        except Exception:
            pass
'''

    write_file_safe(file_path, content)
    print(f"Fixed {file_path}")


def fix_monitoring_logging_file():
    """Fix the monitoring/logging.py file."""
    file_path = "src/llamaagent/monitoring/logging.py"
    print(f"Fixing {file_path}...")

    content = read_file_safe(file_path)
    if not content:
        return

    # Fix the unmatched parenthesis
    content = re.sub(
        r'return StructuredLogger\(name\)\)', 'return StructuredLogger(name)', content
    )

    # Fix other syntax issues
    fixes = [
        (r'except Exception as, e:', 'except Exception as e:'),
        (r'for msg in, messages:', 'for msg in messages:'),
        (r'if, include_memory:', 'if include_memory:'),
        (r'if, enable_console:', 'if enable_console:'),
        (r'if enable_file and, file_path:', 'if enable_file and file_path:'),
        (r'def _log\(self, \*\*kwargs\)', 'def _log(self, level, message, **kwargs)'),
        (r'def critical\(self, \*\*kwargs\)', 'def critical(self, message, **kwargs)'),
        (r'def error\(self, \*\*kwargs\)', 'def error(self, message, **kwargs)'),
        (r'def info\(self, \*\*kwargs\)', 'def info(self, message, **kwargs)'),
        (r'def warning\(self, \*\*kwargs\)', 'def warning(self, message, **kwargs)'),
        (r'def debug\(self, \*\*kwargs\)', 'def debug(self, message, **kwargs)'),
    ]

    for pattern, replacement in fixes:
        content = re.sub(pattern, replacement, content)

    write_file_safe(file_path, content)
    print(f"Fixed {file_path}")


def fix_premium_endpoints_file():
    """Fix the premium_endpoints.py file."""
    file_path = "src/llamaagent/api/premium_endpoints.py"
    print(f"Fixing {file_path}...")

    content = read_file_safe(file_path)
    if not content:
        return

    # Fix the malformed Field definitions
    fixes = [
        (r'description="Dataset description"\)', 'description="Dataset description")'),
        (r'description="Dataset name"\)', 'description="Dataset name")'),
        (
            r'description="Expected output for the sample"\)',
            'description="Expected output for the sample")',
        ),
        (
            r'description="Filter by knowledge type"\)',
            'description="Filter by knowledge type")',
        ),
        (
            r'description="Filter by model name"\)',
            'description="Filter by model name")',
        ),
        (
            r'description="Filter by prompt type"\)',
            'description="Filter by prompt type")',
        ),
        (r'description="Filter by tags"\)', 'description="Filter by tags")'),
        (r'description="Generation strategy"\)', 'description="Generation strategy")'),
        (
            r'context: Optional\[Dict\[str, description="Dataset description"\)',
            'context: Optional[Dict[str, Any]] = Field(default=None, description="Dataset description")',
        ),
        (
            r'metadata: Optional\[Dict\[str, description="Dataset name"\)',
            'metadata: Optional[Dict[str, Any]] = Field(default=None, description="Dataset name")',
        ),
    ]

    for pattern, replacement in fixes:
        content = re.sub(pattern, replacement, content)

    write_file_safe(file_path, content)
    print(f"Fixed {file_path}")


def fix_indentation_errors():
    """Fix common indentation errors in except blocks."""
    print("Fixing indentation errors...")

    # Find all Python files with indentation issues
    python_files = []
    for root, dirs, files in os.walk("src/llamaagent"):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))

    for file_path in python_files:
        content = read_file_safe(file_path)
        if not content:
            continue

        # Fix common indentation patterns
        lines = content.split('\n')
        fixed_lines = []

        for i, line in enumerate(lines):
            # Fix except blocks with missing indentation
            if 'except Exception as' in line and line.strip().endswith(':'):
                fixed_lines.append(line)
                # Check if next line is properly indented
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if next_line.strip() and not next_line.startswith('    '):
                        # Add a pass statement if the next line isn't indented
                        fixed_lines.append('    pass')
                    else:
                        continue
            else:
                fixed_lines.append(line)

        fixed_content = '\n'.join(fixed_lines)
        if fixed_content != content:
            write_file_safe(file_path, fixed_content)
            print(f"Fixed indentation in {file_path}")


def fix_import_errors():
    """Fix common import statement errors."""
    print("Fixing import errors...")

    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk("src/llamaagent"):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))

    for file_path in python_files:
        content = read_file_safe(file_path)
        if not content:
            continue

        # Fix malformed import statements
        fixes = [
            (
                r'from typing import ""',
                'from typing import Any, Dict, List, Optional, Union',
            ),
            (r'import, json', 'import json'),
            (r'except, Exception:', 'except Exception:'),
            (r'if, ', 'if '),
            (r'for, ', 'for '),
            (r'def, ', 'def '),
            (r'class, ', 'class '),
        ]

        original_content = content
        for pattern, replacement in fixes:
            content = re.sub(pattern, replacement, content)

        if content != original_content:
            write_file_safe(file_path, content)
            print(f"Fixed imports in {file_path}")


def main():
    """Main function to fix all critical syntax errors."""
    print("Starting comprehensive syntax error fixing...")

    # Change to the project directory
    os.chdir("/Users/nemesis/llamaagent")

    # Fix specific files with known issues
    fix_openai_stub_file()
    fix_cohere_provider_file()
    fix_monitoring_logging_file()
    fix_premium_endpoints_file()

    # Fix common patterns across all files
    fix_indentation_errors()
    fix_import_errors()

    print("Syntax error fixing completed!")
    print("Run 'python -m py_compile' on individual files to verify fixes.")


if __name__ == "__main__":
    main()

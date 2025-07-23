#!/usr/bin/env python3
"""
Focused Script to Fix Missing Parentheses in LlamaAgent

This script specifically targets the missing parentheses pattern found
in function calls, sum() expressions, and other common patterns.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import logging
import re
import subprocess
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParenthesesFixer:
    """Fix missing parentheses in Python files."""

    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.fixes_applied = []

    def fix_common_patterns(self, content: str) -> str:
        """Fix common patterns of missing parentheses."""

        # Fix missing closing parentheses in function calls
        fixes = [
            # Fix sum() expressions
            (r'sum\(1 for [^)]*\bif\b[^)]*(?!\))', lambda m: m.group(0) + ')'),
            # Fix function calls with missing closing parentheses
            (
                r'ReactAgent\(config=AgentConfig\([^)]*\)(?!\))',
                lambda m: m.group(0) + ')',
            ),
            # Fix len() calls with missing closing parentheses
            (r'len\([^)]*\[([^\]]*)\](?!\))', lambda m: m.group(0) + ')'),
            # Fix division operations with missing parentheses
            (r'len\(getattr\([^)]*\]\s*/\s*max\([^)]*\)', lambda m: m.group(0) + ')'),
            # Fix asyncio.run() calls
            (
                r'asyncio\.run\([^)]*\)(?!\))',
                lambda m: m.group(0) + ')'
                if not m.group(0).endswith('))')
                else m.group(0),
            ),
            # Fix specific patterns found in the codebase
            (
                r'raise HTTPException\([^)]*detail=str\(e\)(?!\))',
                lambda m: m.group(0) + ')',
            ),
            (
                r'datetime\.now\(timezone\.utc\)\)(?!\))',
                lambda m: m.group(0)[:-1],
            ),  # Remove extra parenthesis
            (
                r'return\s+StructuredLogger\(name\)\)',
                lambda m: m.group(0)[:-1],
            ),  # Remove extra parenthesis
        ]

        for pattern, replacement in fixes:
            if callable(replacement):
                content = re.sub(pattern, replacement, content)
            else:
                content = re.sub(pattern, replacement, content)

        return content

    def fix_file(self, file_path: Path) -> bool:
        """Fix a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            fixed_content = self.fix_common_patterns(content)

            if fixed_content != original_content:
                # Test if the fixed content compiles
                try:
                    compile(fixed_content, str(file_path), 'exec')
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(fixed_content)
                    logger.info(f"Fixed: {file_path}")
                    self.fixes_applied.append(str(file_path))
                    return True
                except SyntaxError as e:
                    logger.warning(
                        f"Fixed content still has syntax errors: {file_path} - {e}"
                    )
                    return False
            else:
                logger.debug(f"No changes needed: {file_path}")
                return True

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return False

    def fix_all_syntax_errors(self) -> None:
        """Fix all files with syntax errors."""
        logger.info("Finding files with syntax errors...")

        # Get all Python files with syntax errors
        error_files = []
        python_files = list(self.root_path.rglob("src/**/*.py"))

        for py_file in python_files:
            try:
                result = subprocess.run(
                    ["python3", "-m", "py_compile", str(py_file)],
                    capture_output=True,
                    text=True,
                    cwd=self.root_path,
                )
                if result.returncode != 0:
                    error_files.append(py_file)
            except Exception as e:
                logger.error(f"Error checking {py_file}: {e}")

        logger.info(f"Found {len(error_files)} files with syntax errors")

        # Fix each file
        for file_path in error_files:
            self.fix_file(file_path)

        logger.info(f"Fixed {len(self.fixes_applied)} files")

        # Show remaining errors
        remaining_errors = []
        for py_file in error_files:
            try:
                result = subprocess.run(
                    ["python3", "-m", "py_compile", str(py_file)],
                    capture_output=True,
                    text=True,
                    cwd=self.root_path,
                )
                if result.returncode != 0:
                    remaining_errors.append(py_file)
            except Exception as e:
                logger.error(f"Error checking {py_file}: {e}")

        logger.info(f"Remaining files with syntax errors: {len(remaining_errors)}")
        for error_file in remaining_errors[:10]:  # Show first 10
            logger.warning(f"Still has errors: {error_file}")


def main():
    """Main execution function."""
    fixer = ParenthesesFixer()
    fixer.fix_all_syntax_errors()


if __name__ == "__main__":
    main()

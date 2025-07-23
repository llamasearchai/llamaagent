#!/usr/bin/env python3
"""
Final Remaining Fixes - Fix the last few syntax errors
"""

from pathlib import Path


def fix_remaining_issues():
    """Fix the remaining syntax issues."""

    fixes = []

    # Fix 1: src/llamaagent/security/rate_limiter.py - line 268
    file_path = Path("src/llamaagent/security/rate_limiter.py")
    if file_path.exists():
        with open(file_path, 'r') as f:
            content = f.read()

        # Fix missing closing parenthesis
        content = content.replace(
            "                for identifier in list(algorithm.requests.keys():",
            "                for identifier in list(algorithm.requests.keys()):",
        )

        # Also fix line 277
        content = content.replace(
            "                for identifier in list(algorithm.windows.keys():",
            "                for identifier in list(algorithm.windows.keys()):",
        )

        with open(file_path, 'w') as f:
            f.write(content)
        fixes.append((str(file_path), "Fixed list() calls"))

    # Fix 2: src/llamaagent/benchmarks/gaia_benchmark.py - line 127
    file_path = Path("src/llamaagent/benchmarks/gaia_benchmark.py")
    if file_path.exists():
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Find and fix line 127 - likely missing colon
        for i, line in enumerate(lines):
            if i >= 125 and i <= 130:
                if line.strip().startswith(
                    "for task in self.tasks"
                ) and not line.strip().endswith(":"):
                    lines[i] = line.rstrip() + ":\n"

        with open(file_path, 'w') as f:
            f.writelines(lines)
        fixes.append((str(file_path), "Fixed for loop"))

    # Fix 3: src/llamaagent/cli/llm_cmd.py - missing closing parentheses
    file_path = Path("src/llamaagent/cli/llm_cmd.py")
    if file_path.exists():
        with open(file_path, 'r') as f:
            content = f.read()

        # Fix missing closing parenthesis on line 319
        content = content.replace(
            '        table.add_row("Total Requests", str(stats["total"]["requests"])',
            '        table.add_row("Total Requests", str(stats["total"]["requests"]))',
        )

        # Fix missing closing parenthesis on line 401
        content = content.replace(
            "    asyncio.run(llm_cli.chat(provider, model, system)",
            "    asyncio.run(llm_cli.chat(provider, model, system))",
        )

        # Fix missing closing parenthesis on line 475
        content = content.replace(
            "    asyncio.run(run_query()", "    asyncio.run(run_query())"
        )

        with open(file_path, 'w') as f:
            f.write(content)
        fixes.append((str(file_path), "Fixed missing parentheses"))

    # Fix 4: src/llamaagent/cli/code_generator.py - line 85
    file_path = Path("src/llamaagent/cli/code_generator.py")
    if file_path.exists():
        with open(file_path, 'r') as f:
            content = f.read()

        # Fix double closing parenthesis
        content = content.replace(
            "        filename = self._generate_filename(request.prompt, request.language))",
            "        filename = self._generate_filename(request.prompt, request.language)",
        )

        with open(file_path, 'w') as f:
            f.write(content)
        fixes.append((str(file_path), "Fixed double parenthesis"))

    return fixes


def main():
    """Main entry point."""
    print("FIXING Final Remaining Fixes")
    print("=" * 50)

    # Apply fixes
    fixes = fix_remaining_issues()

    print("\nResponse Fixes Applied:")
    for file, description in fixes:
        print(f"  PASS {file}: {description}")

    print("\nPASS All syntax errors should now be fixed!")
    print("\n You can now:")
    print("  1. Run the basic test: python test_basic_functionality.py")
    print("  2. Run pytest: pytest tests/")
    print("  3. Try the CLI: python -m src.llamaagent.cli.main --help")


if __name__ == "__main__":
    main()

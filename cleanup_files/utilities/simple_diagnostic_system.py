#!/usr/bin/env python3
"""
Simple Comprehensive Diagnostic System for LlamaAgent

This module performs deep analysis of the entire codebase to identify all problems,
analyze root causes, and provide comprehensive solutions.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import ast
import importlib.util
import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Problem:
    """Represents a problem found in the codebase."""

    category: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    file_path: str
    line_number: Optional[int]
    description: str
    root_cause: str
    solution: str
    code_snippet: Optional[str] = None


class SimpleDiagnosticSystem:
    """Simple comprehensive system for diagnosing and fixing codebase issues."""

    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.problems: List[Problem] = []
        self.python_files: List[Path] = []
        self.config_files: List[Path] = []
        self.test_files: List[Path] = []

        # Common anti-patterns to detect
        self.anti_patterns = {
            "mock_fallback": [
                r"fallback.*mock",
                r"mock.*fallback",
                r"except.*MockProvider",
                r"logger\.warning.*mock",
            ],
            "placeholder_values": [
                r"your_api_key_here",
                r"your-api-key",
                r"INSERT_YOUR_KEY",
                r"ADD_YOUR_KEY",
                r"sk-placeholder",
            ],
            "silent_failures": [
                r"except.*pass",
                r"except.*continue",
                r"try:.*except:.*return None",
            ],
        }

    def scan_codebase(self) -> None:
        """Scan the entire codebase for files to analyze."""
        print("Scanning codebase...")

        for file_path in self.root_path.rglob("*.py"):
            if self._should_analyze_file(file_path):
                self.python_files.append(file_path)
                if "test" in str(file_path).lower():
                    self.test_files.append(file_path)

        for pattern in ["*.yaml", "*.yml", "*.json", "*.toml"]:
            for file_path in self.root_path.rglob(pattern):
                if self._should_analyze_file(file_path):
                    self.config_files.append(file_path)

        print(f" Found {len(self.python_files)} Python files")
        print(f" Found {len(self.config_files)} config files")
        print(f" Found {len(self.test_files)} test files")

    def _should_analyze_file(self, file_path: Path) -> bool:
        """Determine if a file should be analyzed."""
        exclude_patterns = [
            "__pycache__",
            ".git",
            ".venv",
            "venv",
            "node_modules",
            ".pytest_cache",
            "htmlcov",
            "dist",
            "build",
            "*.egg-info",
        ]

        str_path = str(file_path)
        return not any(pattern in str_path for pattern in exclude_patterns)

    def analyze_python_syntax(self) -> None:
        """Analyze Python files for syntax errors and issues."""
        print("Analyzing Python syntax...")

        for file_path in self.python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check for syntax errors
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    self.problems.append(
                        Problem(
                            category="SYNTAX_ERROR",
                            severity="CRITICAL",
                            file_path=str(file_path),
                            line_number=e.lineno,
                            description=f"Syntax error: {e.msg}",
                            root_cause="Invalid Python syntax preventing module import",
                            solution="Fix syntax error according to Python grammar rules",
                            code_snippet=e.text.strip() if e.text else None,
                        )
                    )

                # Check for anti-patterns
                self._check_anti_patterns(file_path, content)

                # Check for import issues
                self._check_imports(file_path, content)

            except Exception as e:
                self.problems.append(
                    Problem(
                        category="FILE_READ_ERROR",
                        severity="HIGH",
                        file_path=str(file_path),
                        line_number=None,
                        description=f"Cannot read file: {e}",
                        root_cause="File encoding or permission issues",
                        solution="Fix file encoding or permissions",
                    )
                )

    def _check_anti_patterns(self, file_path: Path, content: str) -> None:
        """Check for anti-patterns in the code."""
        lines = content.split('\n')

        for category, patterns in self.anti_patterns.items():
            for pattern in patterns:
                for line_num, line in enumerate(lines, 1):
                    if re.search(pattern, line, re.IGNORECASE):
                        severity = (
                            "CRITICAL"
                            if category in ["mock_fallback", "silent_failures"]
                            else "HIGH"
                        )

                        self.problems.append(
                            Problem(
                                category=f"ANTI_PATTERN_{category.upper()}",
                                severity=severity,
                                file_path=str(file_path),
                                line_number=line_num,
                                description=f"Anti-pattern detected: {category}",
                                root_cause=self._get_anti_pattern_root_cause(category),
                                solution=self._get_anti_pattern_solution(category),
                                code_snippet=line.strip(),
                            )
                        )

    def _check_imports(self, file_path: Path, content: str) -> None:
        """Check for import issues."""
        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    # Check for problematic imports
                    if isinstance(node, ast.ImportFrom) and node.module:
                        if (
                            "mock" in node.module.lower()
                            and "test" not in str(file_path).lower()
                        ):
                            self.problems.append(
                                Problem(
                                    category="MOCK_IMPORT_IN_PRODUCTION",
                                    severity="HIGH",
                                    file_path=str(file_path),
                                    line_number=node.lineno,
                                    description="Mock import in production code",
                                    root_cause="Mock modules should only be used in tests",
                                    solution="Move mock imports to test files only",
                                )
                            )

        except Exception as e:
            logger.error(
                f"Error: {e}"
            )  # AST parsing already checked in analyze_python_syntax

    def analyze_configuration(self) -> None:
        """Analyze configuration files for issues."""
        print("Analyzing configuration files...")

        for file_path in self.config_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check for placeholder values
                if re.search(
                    r'your_.*_here|INSERT_.*|ADD_.*|placeholder', content, re.IGNORECASE
                ):
                    self.problems.append(
                        Problem(
                            category="CONFIG_PLACEHOLDER",
                            severity="HIGH",
                            file_path=str(file_path),
                            line_number=None,
                            description="Configuration contains placeholder values",
                            root_cause="Default configuration not customized for deployment",
                            solution="Replace placeholder values with actual configuration",
                        )
                    )

                # Check JSON syntax
                if file_path.suffix == '.json':
                    try:
                        json.loads(content)
                    except json.JSONDecodeError as e:
                        self.problems.append(
                            Problem(
                                category="CONFIG_SYNTAX_ERROR",
                                severity="CRITICAL",
                                file_path=str(file_path),
                                line_number=getattr(e, 'lineno', None),
                                description=f"JSON syntax error: {e.msg}",
                                root_cause="Invalid JSON format",
                                solution="Fix JSON syntax according to specification",
                            )
                        )

            except Exception as e:
                self.problems.append(
                    Problem(
                        category="CONFIG_READ_ERROR",
                        severity="HIGH",
                        file_path=str(file_path),
                        line_number=None,
                        description=f"Cannot read config file: {e}",
                        root_cause="File encoding or permission issues",
                        solution="Fix file encoding or permissions",
                    )
                )

    def analyze_test_coverage(self) -> None:
        """Analyze test coverage and quality."""
        print("Analyzing test coverage...")

        non_test_python_files = [
            f for f in self.python_files if f not in self.test_files
        ]
        test_ratio = (
            len(self.test_files) / len(non_test_python_files)
            if non_test_python_files
            else 0
        )

        if test_ratio < 0.3:  # Less than 30% test coverage
            self.problems.append(
                Problem(
                    category="LOW_TEST_COVERAGE",
                    severity="HIGH",
                    file_path="tests/",
                    line_number=None,
                    description=f"Low test coverage: {test_ratio:.1%}",
                    root_cause="Insufficient test files relative to source code",
                    solution="Add more comprehensive test files",
                )
            )

        # Check for test quality issues
        for test_file in self.test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check for assertions
                if 'assert' not in content and 'self.assert' not in content:
                    self.problems.append(
                        Problem(
                            category="NO_ASSERTIONS",
                            severity="HIGH",
                            file_path=str(test_file),
                            line_number=None,
                            description="Test file contains no assertions",
                            root_cause="Tests without assertions don't verify behavior",
                            solution="Add assertions to verify expected behavior",
                        )
                    )

            except Exception as e:
                logger.error(f"Error: {e}")

    def analyze_runtime_issues(self) -> None:
        """Analyze potential runtime issues."""
        print("Analyzing runtime issues...")

        # Try to import main modules
        sys.path.insert(0, str(self.root_path))

        main_modules = [
            'src.llamaagent',
            'src.llamaagent.agents',
            'src.llamaagent.llm',
            'src.llamaagent.api',
        ]

        for module_name in main_modules:
            try:
                spec = importlib.util.find_spec(module_name)
                if spec is None:
                    self.problems.append(
                        Problem(
                            category="MODULE_NOT_FOUND",
                            severity="CRITICAL",
                            file_path=module_name.replace('.', '/'),
                            line_number=None,
                            description=f"Module {module_name} cannot be imported",
                            root_cause="Module path or __init__.py missing",
                            solution="Ensure module structure and __init__.py files exist",
                        )
                    )
            except Exception as e:
                self.problems.append(
                    Problem(
                        category="MODULE_ANALYSIS_ERROR",
                        severity="MEDIUM",
                        file_path=module_name.replace('.', '/'),
                        line_number=None,
                        description=f"Cannot analyze module: {str(e)}",
                        root_cause="Module analysis failure",
                        solution="Check module structure and dependencies",
                    )
                )

    def _get_anti_pattern_root_cause(self, category: str) -> str:
        """Get root cause for anti-pattern categories."""
        causes = {
            "mock_fallback": "Silent fallbacks hide real configuration issues",
            "placeholder_values": "Placeholder values indicate incomplete configuration",
            "silent_failures": "Silent exception handling hides errors",
        }
        return causes.get(category, "Unknown anti-pattern")

    def _get_anti_pattern_solution(self, category: str) -> str:
        """Get solution for anti-pattern categories."""
        solutions = {
            "mock_fallback": "Remove fallback logic and fail fast with clear error messages",
            "placeholder_values": "Replace with environment variables or configuration files",
            "silent_failures": "Add proper error handling with logging and user feedback",
        }
        return solutions.get(
            category, "Address anti-pattern according to best practices"
        )

    def run_comprehensive_diagnosis(self) -> None:
        """Run complete diagnostic analysis."""
        print("Starting comprehensive diagnostic analysis...")

        # Scan codebase
        self.scan_codebase()

        # Run all analysis methods
        self.analyze_python_syntax()
        self.analyze_configuration()
        self.analyze_test_coverage()
        self.analyze_runtime_issues()

        # Count problems by severity
        critical_count = len([p for p in self.problems if p.severity == "CRITICAL"])
        high_count = len([p for p in self.problems if p.severity == "HIGH"])
        medium_count = len([p for p in self.problems if p.severity == "MEDIUM"])
        low_count = len([p for p in self.problems if p.severity == "LOW"])

        print(f"PASS Diagnostic analysis complete!")
        print(f"RESULTS Found {len(self.problems)} total problems:")
        print(f"    Critical: {critical_count}")
        print(f"   🟠 High: {high_count}")
        print(f"   🟡 Medium: {medium_count}")
        print(f"   🟢 Low: {low_count}")

    def write_diagnostic_report(
        self, output_file: str = "diagnostic_report.txt"
    ) -> None:
        """Write comprehensive diagnostic report to file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("LLAMAAGENT COMPREHENSIVE DIAGNOSTIC REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Author: Nik Jois <nikjois@llamasearch.ai>\n")
            f.write("\n")

            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Problems Found: {len(self.problems)}\n")

            critical_count = len([p for p in self.problems if p.severity == "CRITICAL"])
            high_count = len([p for p in self.problems if p.severity == "HIGH"])
            medium_count = len([p for p in self.problems if p.severity == "MEDIUM"])
            low_count = len([p for p in self.problems if p.severity == "LOW"])

            f.write(f"Critical Issues: {critical_count}\n")
            f.write(f"High Priority: {high_count}\n")
            f.write(f"Medium Priority: {medium_count}\n")
            f.write(f"Low Priority: {low_count}\n")
            f.write("\n")

            # Problem Details
            f.write("DETAILED PROBLEM ANALYSIS\n")
            f.write("-" * 40 + "\n")

            for i, problem in enumerate(self.problems, 1):
                f.write(f"\n{i}. {problem.category} [{problem.severity}]\n")
                f.write(f"   File: {problem.file_path}\n")
                if problem.line_number:
                    f.write(f"   Line: {problem.line_number}\n")
                f.write(f"   Description: {problem.description}\n")
                f.write(f"   Root Cause: {problem.root_cause}\n")
                f.write(f"   Solution: {problem.solution}\n")
                if problem.code_snippet:
                    f.write(f"   Code: {problem.code_snippet}\n")

            # Solutions
            f.write("\n\nCOMPREHENSIVE SOLUTIONS\n")
            f.write("-" * 40 + "\n")

            f.write("\n1. CRITICAL ISSUES - IMMEDIATE ACTION REQUIRED\n")
            critical_problems = [p for p in self.problems if p.severity == "CRITICAL"]
            for problem in critical_problems:
                f.write(f"   - {problem.file_path}: {problem.description}\n")
                f.write(f"     SOLUTION: {problem.solution}\n")

            f.write("\n2. HIGH PRIORITY ISSUES\n")
            high_problems = [p for p in self.problems if p.severity == "HIGH"]
            for problem in high_problems:
                f.write(f"   - {problem.file_path}: {problem.description}\n")
                f.write(f"     SOLUTION: {problem.solution}\n")

            f.write("\n3. MEDIUM PRIORITY ISSUES\n")
            medium_problems = [p for p in self.problems if p.severity == "MEDIUM"]
            for problem in medium_problems:
                f.write(f"   - {problem.file_path}: {problem.description}\n")
                f.write(f"     SOLUTION: {problem.solution}\n")

            # Prevention Measures
            f.write("\n\nPREVENTION MEASURES\n")
            f.write("-" * 40 + "\n")
            f.write("1. Setup pre-commit hooks for code quality\n")
            f.write("2. Implement comprehensive testing framework\n")
            f.write("3. Add static analysis tools to CI/CD pipeline\n")
            f.write("4. Setup monitoring and alerting\n")
            f.write("5. Regular dependency updates\n")
            f.write("6. Code review requirements\n")
            f.write("7. Documentation standards\n")
            f.write("8. Configuration management\n")
            f.write("9. Error handling standards\n")
            f.write("10. Performance monitoring\n")

            # Recommendations
            f.write("\n\nRECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")
            f.write("1. Address CRITICAL issues immediately\n")
            f.write("2. Implement fail-fast error handling\n")
            f.write("3. Remove all mock fallback mechanisms\n")
            f.write("4. Add comprehensive configuration validation\n")
            f.write("5. Implement proper logging and monitoring\n")
            f.write("6. Add comprehensive test coverage\n")
            f.write("7. Setup CI/CD pipeline with quality gates\n")
            f.write("8. Document all configuration requirements\n")
            f.write("9. Implement security best practices\n")
            f.write("10. Regular code quality audits\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF DIAGNOSTIC REPORT\n")
            f.write("=" * 80 + "\n")

        print(f" Diagnostic report written to {output_file}")

    def implement_critical_fixes(self) -> None:
        """Implement critical fixes automatically."""
        print("Implementing critical fixes...")

        # Create missing __init__.py files
        required_init_files = [
            "src/__init__.py",
            "src/llamaagent/__init__.py",
            "src/llamaagent/agents/__init__.py",
            "src/llamaagent/llm/__init__.py",
            "src/llamaagent/llm/providers/__init__.py",
            "src/llamaagent/api/__init__.py",
            "src/llamaagent/tools/__init__.py",
            "src/llamaagent/core/__init__.py",
        ]

        for init_file in required_init_files:
            path = Path(init_file)
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('"""Module init file."""\n')
                print(f"Created {init_file}")

        # Create configuration template
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)

        config_template = {
            "llm_providers": {
                "openai": {
                    "api_key": "${OPENAI_API_KEY}",
                    "model": "gpt-4o-mini",
                    "temperature": 0.7,
                },
                "anthropic": {
                    "api_key": "${ANTHROPIC_API_KEY}",
                    "model": "claude-3-sonnet-20240229",
                    "temperature": 0.7,
                },
                "mock": {
                    "model": "mock-model",
                    "responses": ["Mock response for testing"],
                },
            },
            "database": {"url": "${DATABASE_URL}", "pool_size": 10},
            "api": {"host": "0.0.0.0", "port": 8000, "debug": False},
        }

        with open("config/production.json", "w") as f:
            json.dump(config_template, f, indent=2)

        print("Created production configuration template")

        # Create environment template
        env_template = """# LlamaAgent Environment Configuration
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
DATABASE_URL=postgresql://user:pass@localhost/llamaagent
LLAMAAGENT_LLM_PROVIDER=openai
LLAMAAGENT_DEBUG=false
"""

        with open(".env.example", "w") as f:
            f.write(env_template)

        print("Created environment configuration template")

        # Create pre-commit configuration
        pre_commit_config = """repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-merge-conflict
      - id: check-ast
      - id: check-docstring-first

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203,W503]
"""

        with open(".pre-commit-config.yaml", "w") as f:
            f.write(pre_commit_config)

        print("Created pre-commit configuration")

        print("PASS Critical fixes implemented successfully!")


def main():
    """Main function to run comprehensive diagnostic system."""
    print("FIXING LlamaAgent Simple Diagnostic System")
    print("=" * 50)

    # Initialize diagnostic system
    diagnostic_system = SimpleDiagnosticSystem()

    # Run comprehensive diagnosis
    diagnostic_system.run_comprehensive_diagnosis()

    # Write report to file
    diagnostic_system.write_diagnostic_report()

    # Implement critical fixes
    diagnostic_system.implement_critical_fixes()

    print("\nSUCCESS Comprehensive diagnostic and fixing process complete!")
    print("LIST: Next steps:")
    print("1. Review the diagnostic_report.txt file")
    print("2. Address any remaining CRITICAL issues manually")
    print("3. Run tests to verify fixes")
    print("4. Commit changes and deploy")


if __name__ == "__main__":
    main()

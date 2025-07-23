#!/usr/bin/env python3
"""
Simple Standalone Diagnostics for LlamaAgent

Performs basic syntax checking and identifies critical issues without
requiring the main module to be importable.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import ast
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def main():
    """Run basic diagnostics and generate a text report."""
    print("Scanning LlamaAgent Simple Diagnostics")
    print("=" * 60)

    project_root = Path(__file__).parent

    # Collect all Python files
    python_files = list(project_root.rglob("src/**/*.py"))
    python_files.extend(project_root.rglob("tests/**/*.py"))
    python_files.extend(project_root.rglob("*.py"))

    # Exclude certain directories
    exclude_patterns = ["__pycache__", ".venv", "venv", ".git", "build", "dist"]
    python_files = [
        f
        for f in python_files
        if not any(pattern in str(f) for pattern in exclude_patterns)
    ]

    print(f" Found {len(python_files)} Python files to analyze")

    # Initialize problem tracking
    problems = []
    files_with_errors = 0
    total_lines = 0

    # Analyze each file
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()

            lines = content.split('\n')
            total_lines += len(lines)

            # Check syntax
            try:
                ast.parse(content)
                print(f"PASS {py_file.relative_to(project_root)}")
            except SyntaxError as e:
                print(f"FAIL {py_file.relative_to(project_root)} - SYNTAX ERROR")
                problems.append(
                    {
                        'severity': 'CRITICAL',
                        'category': 'SYNTAX_ERROR',
                        'file': str(py_file.relative_to(project_root)),
                        'line': e.lineno or 0,
                        'title': f'Syntax Error in {py_file.name}',
                        'description': str(e),
                        'fix': 'Fix syntax error based on error message',
                    }
                )
                files_with_errors += 1
                continue

            # Check for other issues
            check_imports(py_file, content, lines, problems, project_root)
            check_security_patterns(py_file, content, lines, problems, project_root)
            check_code_quality(py_file, content, lines, problems, project_root)

        except UnicodeDecodeError:
            print(f"WARNING:  {py_file.relative_to(project_root)} - Encoding error")
            problems.append(
                {
                    'severity': 'HIGH',
                    'category': 'CONFIGURATION_ERROR',
                    'file': str(py_file.relative_to(project_root)),
                    'line': 0,
                    'title': f'Encoding Error in {py_file.name}',
                    'description': 'File contains non-UTF-8 characters',
                    'fix': 'Convert file to UTF-8 encoding',
                }
            )
            files_with_errors += 1
        except Exception as e:
            print(f"WARNING:  {py_file.relative_to(project_root)} - {str(e)}")

    # Check for missing critical files
    check_project_structure(project_root, problems)
    check_configuration_files(project_root, problems)

    # Generate report
    generate_report(
        project_root, problems, len(python_files), total_lines, files_with_errors
    )

    # Return exit code
    critical_count = len([p for p in problems if p['severity'] == 'CRITICAL'])
    high_count = len([p for p in problems if p['severity'] == 'HIGH'])

    print(f"\nRESULTS Analysis Summary:")
    print(f"   • Files Analyzed: {len(python_files)}")
    print(f"   • Lines Analyzed: {total_lines}")
    print(f"   • Files with Errors: {files_with_errors}")
    print(f"   • Total Issues: {len(problems)}")
    print(f"   • Critical Issues: {critical_count}")
    print(f"   • High Priority Issues: {high_count}")

    if critical_count > 0:
        print(f"\nFAIL RESULT: {critical_count} critical issues found!")
        print("FIXING Fix critical issues first before running the system.")
        return 1
    elif high_count > 0:
        print(f"\nWARNING:  RESULT: {high_count} high-priority issues found.")
        print("LIST: See full report for details.")
        return 0
    else:
        print(f"\nPASS RESULT: No critical issues found!")
        return 0


def check_imports(
    py_file: Path,
    content: str,
    lines: List[str],
    problems: List[Dict],
    project_root: Path,
):
    """Check for import-related issues."""
    # Look for common problematic imports
    risky_imports = ['eval', 'exec', 'subprocess.call.*shell=True']

    for line_no, line in enumerate(lines, 1):
        line_stripped = line.strip()

        # Check for risky imports
        for pattern in risky_imports:
            if re.search(pattern, line):
                problems.append(
                    {
                        'severity': 'HIGH',
                        'category': 'SECURITY_VULNERABILITY',
                        'file': str(py_file.relative_to(project_root)),
                        'line': line_no,
                        'title': f'Risky Import/Usage: {pattern}',
                        'description': f'Potentially dangerous usage detected: {line_stripped}',
                        'fix': 'Review and secure this usage',
                    }
                )


def check_security_patterns(
    py_file: Path,
    content: str,
    lines: List[str],
    problems: List[Dict],
    project_root: Path,
):
    """Check for security-related issues."""
    security_patterns = [
        (r'password\s*=\s*["\'][^"\']+["\']', 'Hardcoded password'),
        (r'api_key\s*=\s*["\'][^"\']+["\']', 'Hardcoded API key'),
        (r'secret\s*=\s*["\'][^"\']+["\']', 'Hardcoded secret'),
        (r'token\s*=\s*["\'][^"\']+["\']', 'Hardcoded token'),
    ]

    for line_no, line in enumerate(lines, 1):
        for pattern, description in security_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                problems.append(
                    {
                        'severity': 'HIGH',
                        'category': 'SECURITY_VULNERABILITY',
                        'file': str(py_file.relative_to(project_root)),
                        'line': line_no,
                        'title': f'Security Issue: {description}',
                        'description': f'Potential secret exposure: {line.strip()}',
                        'fix': 'Move secret to environment variable',
                    }
                )


def check_code_quality(
    py_file: Path,
    content: str,
    lines: List[str],
    problems: List[Dict],
    project_root: Path,
):
    """Check for code quality issues."""
    for line_no, line in enumerate(lines, 1):
        # Check for very long lines
        if len(line) > 120:
            problems.append(
                {
                    'severity': 'LOW',
                    'category': 'CODE_QUALITY',
                    'file': str(py_file.relative_to(project_root)),
                    'line': line_no,
                    'title': 'Long Line',
                    'description': f'Line exceeds 120 characters ({len(line)} chars)',
                    'fix': 'Break line into multiple lines',
                }
            )


def check_project_structure(project_root: Path, problems: List[Dict]):
    """Check for missing critical files and directories."""
    critical_files = [
        'requirements.txt',
        'src/llamaagent/__init__.py',
        'src/llamaagent/cli/__init__.py',
        'src/llamaagent/core/__init__.py',
    ]

    for file_path in critical_files:
        full_path = project_root / file_path
        if not full_path.exists():
            problems.append(
                {
                    'severity': 'CRITICAL',
                    'category': 'CONFIGURATION_ERROR',
                    'file': file_path,
                    'line': 0,
                    'title': f'Missing Critical File: {file_path}',
                    'description': f'Critical file {file_path} is missing',
                    'fix': f'Create the missing file {file_path}',
                }
            )


def check_configuration_files(project_root: Path, problems: List[Dict]):
    """Check configuration files."""
    config_files = [
        ('README.md', 'MEDIUM', 'Project documentation'),
        ('pyproject.toml', 'MEDIUM', 'Project configuration'),
        ('.gitignore', 'MEDIUM', 'Git ignore rules'),
        ('Dockerfile', 'LOW', 'Docker configuration'),
        ('docker-compose.yml', 'LOW', 'Docker Compose configuration'),
    ]

    for filename, severity, description in config_files:
        file_path = project_root / filename
        if not file_path.exists():
            problems.append(
                {
                    'severity': severity,
                    'category': 'CONFIGURATION_ERROR',
                    'file': filename,
                    'line': 0,
                    'title': f'Missing {description}: {filename}',
                    'description': f'{description} file is missing',
                    'fix': f'Create {filename}',
                }
            )


def generate_report(
    project_root: Path,
    problems: List[Dict],
    total_files: int,
    total_lines: int,
    files_with_errors: int,
):
    """Generate comprehensive text report."""
    output_file = project_root / "llamaagent_diagnostic_report.txt"

    # Group problems by severity
    problems_by_severity = {}
    for problem in problems:
        severity = problem['severity']
        if severity not in problems_by_severity:
            problems_by_severity[severity] = []
        problems_by_severity[severity].append(problem)

    # Group problems by category
    problems_by_category = {}
    for problem in problems:
        category = problem['category']
        if category not in problems_by_category:
            problems_by_category[category] = []
        problems_by_category[category].append(problem)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("LLAMAAGENT COMPREHENSIVE DIAGNOSTIC REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Project Root: {project_root}\n")
        f.write("\n")

        # Executive Summary
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total Issues Found: {len(problems)}\n")
        f.write(f"Files Analyzed: {total_files}\n")
        f.write(f"Lines Analyzed: {total_lines}\n")
        f.write(f"Files with Errors: {files_with_errors}\n")
        f.write("\n")

        # Issues by Severity
        f.write("ISSUES BY SEVERITY\n")
        f.write("-" * 50 + "\n")
        for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO']:
            count = len(problems_by_severity.get(severity, []))
            f.write(f"{severity}: {count} issues\n")
        f.write("\n")

        # Issues by Category
        f.write("ISSUES BY CATEGORY\n")
        f.write("-" * 50 + "\n")
        for category, issues in problems_by_category.items():
            f.write(f"{category}: {len(issues)} issues\n")
        f.write("\n")

        # Recommendations
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 50 + "\n")
        critical_count = len(problems_by_severity.get('CRITICAL', []))
        high_count = len(problems_by_severity.get('HIGH', []))

        if critical_count > 0:
            f.write(
                f"URGENT URGENT: Fix {critical_count} critical issues immediately - system may not function\n"
            )
        if high_count > 0:
            f.write(
                f"WARNING:  HIGH PRIORITY: Address {high_count} high-priority issues to prevent major problems\n"
            )

        if problems_by_category.get('SYNTAX_ERROR'):
            f.write("FIXING Fix syntax errors first - they prevent code execution\n")
        if problems_by_category.get('SECURITY_VULNERABILITY'):
            f.write(
                "SECURITY Address security vulnerabilities to protect against attacks\n"
            )
        if problems_by_category.get('CONFIGURATION_ERROR'):
            f.write("Analyzing  Fix configuration errors for proper system operation\n")

        f.write(
            "TARGET Focus on critical and high-priority issues first for maximum impact\n"
        )
        f.write("\n")

        # Detailed Issues
        f.write("DETAILED PROBLEM ANALYSIS\n")
        f.write("=" * 80 + "\n")

        for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO']:
            issues = problems_by_severity.get(severity, [])
            if not issues:
                continue

            f.write(f"\n{severity} PRIORITY ISSUES ({len(issues)} issues)\n")
            f.write("=" * 60 + "\n")

            for i, problem in enumerate(issues, 1):
                f.write(f"\n{i}. {problem['title']}\n")
                f.write(f"   Category: {problem['category']}\n")
                f.write(f"   Location: {problem['file']}\n")
                if problem['line'] > 0:
                    f.write(f"   Line: {problem['line']}\n")
                f.write(f"   Description: {problem['description']}\n")
                if problem.get('fix'):
                    f.write(f"   Suggested Fix: {problem['fix']}\n")
                f.write("   " + "-" * 60 + "\n")

        # Statistics
        f.write("\nFINAL STATISTICS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total Python Files: {total_files}\n")
        f.write(f"Total Lines of Code: {total_lines}\n")
        f.write(f"Files with Issues: {files_with_errors}\n")
        f.write(f"Issue Density: {len(problems)/total_files:.2f} issues per file\n")
        f.write(
            f"Critical Issue Rate: {critical_count/total_files:.2f} critical issues per file\n"
        )

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF DIAGNOSTIC REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"\nPASS Detailed diagnostic report saved to: {output_file}")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

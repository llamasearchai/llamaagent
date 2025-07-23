"""
Code Analyzer - Advanced Python Code Analysis

Performs deep static analysis of Python code for syntax errors,
code quality issues, security vulnerabilities, and performance problems.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import ast
import re
from pathlib import Path
from typing import Any, Dict, List


class CodeAnalyzer:
    """Advanced Python code analyzer."""

    def __init__(self):
        """Initialize the code analyzer."""
        self.issues = []

        # Security patterns to detect
        self.security_patterns = [
            (r"eval\s*\(", "Use of eval() is dangerous", "HIGH"),
            (r"exec\s*\(", "Use of exec() is dangerous", "HIGH"),
            (
                r"subprocess\.call\s*\(.*shell\s*=\s*True",
                "Shell injection vulnerability",
                "CRITICAL",
            ),
            (r'password\s*=\s*["\'][^"\']*["\']', "Hardcoded password", "HIGH"),
            (r'api_key\s*=\s*["\'][^"\']*["\']', "Hardcoded API key", "HIGH"),
            (r'secret\s*=\s*["\'][^"\']*["\']', "Hardcoded secret", "HIGH"),
        ]

        # Performance anti-patterns
        self.performance_patterns = [
            (
                r"for\s+\w+\s+in\s+range\(len\([^)]+\)\)",
                "Use enumerate() instead",
                "LOW",
            ),
            (r"\.join\([^)]*\+[^)]*\)", "Inefficient string concatenation", "MEDIUM"),
            (r"global\s+\w+", "Global variables can hurt performance", "LOW"),
        ]

    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single Python file comprehensively."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            results = {
                "file_path": str(file_path),
                "syntax_errors": [],
                "import_errors": [],
                "security_issues": [],
                "performance_issues": [],
                "code_quality_issues": [],
                "complexity_metrics": {},
                "line_count": len(content.splitlines()),
                "char_count": len(content),
            }

            # Check syntax
            results["syntax_errors"] = self._check_syntax(content)
            if not results["syntax_errors"]:  # Only proceed if syntax is valid
                # Parse AST for deeper analysis
                tree = ast.parse(content)
                results["import_errors"] = self._check_imports(tree, content)
                results["code_quality_issues"] = self._check_code_quality(tree, content)
                results["complexity_metrics"] = self._calculate_complexity(tree)
            # Pattern-based checks (work even with syntax errors)
            results["security_issues"] = self._check_security_patterns(content)
            results["performance_issues"] = self._check_performance_patterns(content)
            return results

        except Exception as e:
            return {
                "file_path": str(file_path),
                "error": f"Analysis failed: {str(e)}",
                "syntax_errors": [],
                "import_errors": [],
                "security_issues": [],
                "performance_issues": [],
                "code_quality_issues": [],
                "complexity_metrics": {},
                "line_count": 0,
                "char_count": 0,
            }

    def _check_syntax(self, content: str) -> List[Dict[str, Any]]:
        """Check for syntax errors."""
        errors = []
        try:
            ast.parse(content)
        except SyntaxError as e:
            errors.append(
                {
                    "type": "syntax_error",
                    "message": str(e),
                    "line": e.lineno,
                    "column": e.offset,
                    "severity": "CRITICAL",
                }
            )
        return errors

    def _check_imports(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Check for problematic imports."""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ["os", "subprocess", "pickle"]:
                        issues.append(
                            {
                                "type": "risky_import",
                                "message": f"Import of '{alias.name}' can be risky",
                                "line": node.lineno,
                                "severity": "MEDIUM",
                                "module": alias.name,
                            }
                        )

            elif isinstance(node, ast.ImportFrom):
                if node.module == "__future__":
                    continue  # Future imports are fine

                # Check for relative imports that go up too many levels
                if node.level > 2:
                    issues.append(
                        {
                            "type": "deep_relative_import",
                            "message": f"Relative import goes up {node.level} levels",
                            "line": node.lineno,
                            "severity": "MEDIUM",
                        }
                    )

        return issues

    def _check_security_patterns(self, content: str) -> List[Dict[str, Any]]:
        """Check for security vulnerabilities using pattern matching."""
        issues = []
        lines = content.splitlines()

        for pattern, message, severity in self.security_patterns:
            for line_no, line in enumerate(lines, 1):
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(
                        {
                            "type": "security_vulnerability",
                            "message": message,
                            "line": line_no,
                            "severity": severity,
                            "code": line.strip(),
                            "pattern": pattern,
                        }
                    )

        return issues

    def _check_performance_patterns(self, content: str) -> List[Dict[str, Any]]:
        """Check for performance anti-patterns."""
        issues = []
        lines = content.splitlines()

        for pattern, message, severity in self.performance_patterns:
            for line_no, line in enumerate(lines, 1):
                if re.search(pattern, line):
                    issues.append(
                        {
                            "type": "performance_issue",
                            "message": message,
                            "line": line_no,
                            "severity": severity,
                            "code": line.strip(),
                            "pattern": pattern,
                        }
                    )

        return issues

    def _check_code_quality(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Check for code quality issues."""
        issues = []
        lines = content.splitlines()

        # Check for long lines
        for line_no, line in enumerate(lines, 1):
            if len(line) > 120:
                issues.append(
                    {
                        "type": "long_line",
                        "message": f"Line too long ({len(line)} > 120 characters)",
                        "line": line_no,
                        "severity": "LOW",
                        "length": len(line),
                    }
                )

        # Check for missing docstrings
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not ast.get_docstring(node):
                    issues.append(
                        {
                            "type": "missing_docstring",
                            "message": f"Function '{node.name}' lacks docstring",
                            "line": node.lineno,
                            "severity": "LOW",
                            "function_name": node.name,
                        }
                    )

            elif isinstance(node, ast.ClassDef):
                if not ast.get_docstring(node):
                    issues.append(
                        {
                            "type": "missing_docstring",
                            "message": f"Class '{node.name}' lacks docstring",
                            "line": node.lineno,
                            "severity": "MEDIUM",
                            "class_name": node.name,
                        }
                    )

        # Check for too many arguments
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                arg_count = len(node.args.args)
                if arg_count > 7:
                    issues.append(
                        {
                            "type": "too_many_arguments",
                            "message": f"Function '{node.name}' has {arg_count} arguments (> 7)",
                            "line": node.lineno,
                            "severity": "MEDIUM",
                            "function_name": node.name,
                            "argument_count": arg_count,
                        }
                    )

        return issues

    def _calculate_complexity(self, tree: ast.AST) -> Dict[str, Any]:
        """Calculate complexity metrics."""
        metrics = {
            "cyclomatic_complexity": 0,
            "function_count": 0,
            "class_count": 0,
            "total_lines": 0,
            "complexity_by_function": {},
        }

        # Count nodes that contribute to complexity
        complexity = 1  # Base complexity

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.Try):
                complexity += len(node.handlers)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                metrics["function_count"] += 1
                func_complexity = self._calculate_function_complexity(node)
                metrics["complexity_by_function"][node.name] = func_complexity
            elif isinstance(node, ast.ClassDef):
                metrics["class_count"] += 1

        metrics["cyclomatic_complexity"] = complexity
        return metrics

    def _calculate_function_complexity(self, func_node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity for a single function."""
        complexity = 1

        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.Try):
                complexity += len(node.handlers)
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1

        return complexity

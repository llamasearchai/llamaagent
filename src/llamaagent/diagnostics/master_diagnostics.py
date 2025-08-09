"""
Master Diagnostics Module - Comprehensive System Analysis

This module performs deep analysis of the entire LlamaAgent codebase,
identifying syntax errors, import issues, dependency problems, configuration
errors, security vulnerabilities, and performance bottlenecks.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import ast
import importlib
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
# Optional imports for advanced analysis
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    PYLINT_AVAILABLE = True
except ImportError:
    PYLINT_AVAILABLE = False

try:
    SAFETY_AVAILABLE = True
except ImportError:
    SAFETY_AVAILABLE = False


class ProblemSeverity(Enum):
    """Severity levels for identified problems."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class ProblemCategory(Enum):
    """Categories of problems that can be identified."""

    SYNTAX_ERROR = "SYNTAX_ERROR"
    IMPORT_ERROR = "IMPORT_ERROR"
    DEPENDENCY_MISSING = "DEPENDENCY_MISSING"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    SECURITY_VULNERABILITY = "SECURITY_VULNERABILITY"
    PERFORMANCE_ISSUE = "PERFORMANCE_ISSUE"
    CODE_QUALITY = "CODE_QUALITY"
    DOCUMENTATION = "DOCUMENTATION"
    TESTING = "TESTING"
    DEPLOYMENT = "DEPLOYMENT"
    COMPATIBILITY = "COMPATIBILITY"
    ARCHITECTURE = "ARCHITECTURE"


@dataclass
class Problem:
    """Represents a single identified problem."""

    severity: ProblemSeverity
    category: ProblemCategory
    title: str
    description: str
    location: str
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    suggested_fix: Optional[str] = None
    code_snippet: Optional[str] = None
    related_files: List[str] = field(default_factory=list)
    impact: str = ""
    priority: int = 0  # 1-10, higher is more urgent
    estimated_fix_time: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class DiagnosticReport:
    """Complete diagnostic report for the system."""

    timestamp: str
    total_problems: int
    problems_by_severity: Dict[ProblemSeverity, int]
    problems_by_category: Dict[ProblemCategory, int]
    problems: List[Problem]
    system_info: Dict[str, Any]
    analysis_summary: Dict[str, Any]
    recommendations: List[str]
    total_files_analyzed: int
    total_lines_analyzed: int
    analysis_duration: float


class MasterDiagnostics:
    """
    Master diagnostic system for comprehensive LlamaAgent analysis.

    This class performs deep analysis of the entire codebase and generates
    actionable reports for fixing identified issues.
    """

    def __init__(self, project_root: Optional[str] = None):
        """Initialize the master diagnostics system."""
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.problems: List[Problem] = []
        self.analysis_start_time = 0.0
        self.total_files_analyzed = 0
        self.total_lines_analyzed = 0

        # Analysis configuration
        self.include_patterns = [
            "*.py",
            "*.yml",
            "*.yaml",
            "*.toml",
            "*.json",
            "*.md",
            "*.txt",
            "*.sh",
            "Dockerfile*",
            "docker-compose*",
            "requirements*",
            "setup.py",
            "pyproject.toml",
        ]

        self.exclude_patterns = [
            "__pycache__",
            "*.pyc",
            ".git",
            ".pytest_cache",
            ".coverage",
            "htmlcov",
            "dist",
            "build",
            "*.egg-info",
            ".venv",
            "venv",
            ".env",
        ]

        self.critical_files = [
            "src/llamaagent/__init__.py",
            "src/llamaagent/cli/__init__.py",
            "src/llamaagent/core/__init__.py",
            "src/llamaagent/agents/__init__.py",
            "src/llamaagent/llm/__init__.py",
            "requirements.txt",
            "setup.py",
            "pyproject.toml",
        ]

    def run_comprehensive_analysis(self) -> DiagnosticReport:
        """Run complete diagnostic analysis of the LlamaAgent system."""
        print("Analyzing Starting comprehensive LlamaAgent diagnostics...")
        self.analysis_start_time = time.time()
        self.problems = []

        # System information
        system_info = self._gather_system_info()

        # Analysis phases
        self._analyze_project_structure()
        self._analyze_dependencies()
        self._analyze_python_code()
        self._analyze_configuration_files()
        self._analyze_documentation()
        self._analyze_tests()
        self._analyze_security()
        self._analyze_performance()
        self._analyze_deployment()

        # Generate report
        analysis_duration = time.time() - self.analysis_start_time
        report = self._generate_report(system_info, analysis_duration)
        print(
            f"PASS Analysis complete! Found {len(self.problems)} issues in {analysis_duration:.2f}s"
        )
        return report

    def _gather_system_info(self) -> Dict[str, Any]:
        """Gather system information for the report."""
        info = {
            "python_version": sys.version,
            "platform": sys.platform,
            "project_root": str(self.project_root),
            "working_directory": os.getcwd(),
            "environment_variables": dict(os.environ),
        }

        if PSUTIL_AVAILABLE:
            info.update(
                {
                    "cpu_count": psutil.cpu_count(),
                    "memory_total": psutil.virtual_memory().total,
                    "memory_available": psutil.virtual_memory().available,
                    "disk_usage": psutil.disk_usage("/").percent,
                }
            )

        return info

    def _analyze_project_structure(self) -> None:
        """Analyze the project structure for missing critical files."""
        print(" Analyzing project structure...")
        # Check for critical files
        for critical_file in self.critical_files:
            file_path = self.project_root / critical_file
            if not file_path.exists():
                self._add_problem(
                    severity=ProblemSeverity.CRITICAL,
                    category=ProblemCategory.CONFIGURATION_ERROR,
                    title=f"Missing Critical File: {critical_file}",
                    description=f"The critical file {critical_file} is missing from the project.",
                    location=str(file_path),
                    suggested_fix=f"Create the missing file {critical_file}",
                    impact="System may not function correctly without this file",
                    priority=9,
                    estimated_fix_time="5-15 minutes",
                )

        # Check directory structure
        expected_dirs = [
            "src/llamaagent",
            "src/llamaagent/cli",
            "src/llamaagent/core",
            "src/llamaagent/agents",
            "src/llamaagent/llm",
            "src/llamaagent/cache",
            "tests",
            "docs",
        ]

        for expected_dir in expected_dirs:
            dir_path = self.project_root / expected_dir
            if not dir_path.exists():
                self._add_problem(
                    severity=ProblemSeverity.HIGH,
                    category=ProblemCategory.ARCHITECTURE,
                    title=f"Missing Directory: {expected_dir}",
                    description=f"Expected directory {expected_dir} is missing.",
                    location=str(dir_path),
                    suggested_fix=f"Create directory: mkdir -p {expected_dir}",
                    impact="Module imports may fail",
                    priority=7,
                    estimated_fix_time="1 minute",
                )

    def _analyze_dependencies(self) -> None:
        """Analyze project dependencies for issues."""
        print("Analyzing dependencies...")
        # Check requirements files
        req_files = [
            "requirements.txt",
            "requirements-dev.txt",
            "requirements-test.txt",
            "pyproject.toml",
            "setup.py",
        ]

        found_req_files = []
        for req_file in req_files:
            file_path = self.project_root / req_file
            if file_path.exists():
                found_req_files.append(req_file)
        if not found_req_files:
            self._add_problem(
                severity=ProblemSeverity.CRITICAL,
                category=ProblemCategory.DEPENDENCY_MISSING,
                title="No Dependency Files Found",
                description="No requirements.txt, setup.py, or pyproject.toml found.",
                location=str(self.project_root),
                suggested_fix="Create requirements.txt with project dependencies",
                impact="Cannot install project dependencies",
                priority=10,
                estimated_fix_time="30 minutes",
            )

        # Analyze requirements.txt if it exists
        req_path = self.project_root / "requirements.txt"
        if req_path.exists():
            self._analyze_requirements_file(req_path)
        # Check for common missing dependencies
        self._check_missing_dependencies()

    def _analyze_requirements_file(self, file_path: Path) -> None:
        """Analyze a requirements file for issues."""
        try:
            with open(file_path, "r") as f:
                lines = f.readlines()

            self.total_files_analyzed += 1
            self.total_lines_analyzed += len(lines)
            for line_no, line in enumerate(lines, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Check for unpinned versions
                if "==" not in line and ">=" not in line and "~=" not in line:
                    self._add_problem(
                        severity=ProblemSeverity.MEDIUM,
                        category=ProblemCategory.DEPENDENCY_MISSING,
                        title=f"Unpinned Dependency: {line}",
                        description=f"Dependency {line} is not pinned to a specific version.",
                        location=str(file_path),
                        line_number=line_no,
                        suggested_fix=f"Pin version: {line}==<version>",
                        impact="May cause version conflicts in production",
                        priority=5,
                        estimated_fix_time="2 minutes",
                    )

        except Exception as e:
            self._add_problem(
                severity=ProblemSeverity.HIGH,
                category=ProblemCategory.CONFIGURATION_ERROR,
                title=f"Cannot Read Requirements File: {file_path.name}",
                description=f"Error reading requirements file: {str(e)}",
                location=str(file_path),
                suggested_fix="Fix file encoding or permissions",
                impact="Cannot determine project dependencies",
                priority=8,
                estimated_fix_time="5 minutes",
            )

    def _check_missing_dependencies(self) -> None:
        """Check for commonly missing dependencies."""
        common_deps = [
            "typer",
            "rich",
            "fastapi",
            "uvicorn",
            "openai",
            "anthropic",
            "redis",
            "pytest",
            "pytest-asyncio",
            "pytest-cov",
        ]

        for dep in common_deps:
            try:
                importlib.import_module(dep.replace("-", "_"))
            except ImportError:
                self._add_problem(
                    severity=ProblemSeverity.MEDIUM,
                    category=ProblemCategory.DEPENDENCY_MISSING,
                    title=f"Missing Dependency: {dep}",
                    description=f"Common dependency {dep} is not installed.",
                    location="system",
                    suggested_fix=f"Install dependency: pip install {dep}",
                    impact="Some features may not work",
                    priority=6,
                    estimated_fix_time="1 minute",
                )

    def _analyze_python_code(self) -> None:
        """Analyze all Python files for syntax and import errors."""
        print("Analyzing Python code...")
        python_files = list(self.project_root.rglob("*.py"))
        for py_file in python_files:
            if self._should_skip_file(py_file):
                continue

            self._analyze_python_file(py_file)

    def _analyze_python_file(self, file_path: Path) -> None:
        """Analyze a single Python file for issues."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.total_files_analyzed += 1
            lines = content.split("\n")
            self.total_lines_analyzed += len(lines)
            # Check for syntax errors
            try:
                ast.parse(content)
            except SyntaxError as e:
                self._add_problem(
                    severity=ProblemSeverity.CRITICAL,
                    category=ProblemCategory.SYNTAX_ERROR,
                    title=f"Syntax Error in {file_path.name}",
                    description=f"Syntax error: {str(e)}",
                    location=str(file_path),
                    line_number=e.lineno,
                    column_number=e.offset,
                    code_snippet=lines[e.lineno - 1]
                    if e.lineno and e.lineno <= len(lines)
                    else None,
                    suggested_fix="Fix syntax error based on error message",
                    impact="File cannot be imported or executed",
                    priority=10,
                    estimated_fix_time="5-30 minutes",
                )
                return  # Skip further analysis if syntax is broken

            # Check for import errors
            self._check_imports(file_path, content)
            # Check for code quality issues
            self._check_code_quality(file_path, content, lines)
        except UnicodeDecodeError:
            self._add_problem(
                severity=ProblemSeverity.HIGH,
                category=ProblemCategory.CONFIGURATION_ERROR,
                title=f"Encoding Error in {file_path.name}",
                description="File contains non-UTF-8 characters",
                location=str(file_path),
                suggested_fix="Convert file to UTF-8 encoding",
                impact="File cannot be read properly",
                priority=7,
                estimated_fix_time="2 minutes",
            )

        except Exception as e:
            self._add_problem(
                severity=ProblemSeverity.MEDIUM,
                category=ProblemCategory.CODE_QUALITY,
                title=f"Analysis Error in {file_path.name}",
                description=f"Error analyzing file: {str(e)}",
                location=str(file_path),
                suggested_fix="Check file for corruption or unusual content",
                impact="Cannot perform complete analysis",
                priority=4,
                estimated_fix_time="10 minutes",
            )

    def _check_imports(self, file_path: Path, content: str) -> None:
        """Check for import-related issues."""
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self._validate_import(file_path, alias.name, node.lineno)
                elif isinstance(node, ast.ImportFrom):
                    module_name = node.module
                    if module_name:
                        self._validate_import(file_path, module_name, node.lineno)
        except Exception:
            # Already handled by syntax error check
            pass

    def _validate_import(self, file_path: Path, module_name: str, line_no: int) -> None:
        """Validate that an import can be resolved."""
        try:
            # Skip relative imports - they're harder to validate
            if module_name.startswith("."):
                return

            # Try to import the module
            importlib.import_module(module_name)
        except ImportError:
            self._add_problem(
                severity=ProblemSeverity.HIGH,
                category=ProblemCategory.IMPORT_ERROR,
                title=f"Import Error: {module_name}",
                description=f"Cannot import module '{module_name}' in {file_path.name}",
                location=str(file_path),
                line_number=line_no,
                suggested_fix="Install missing dependency or fix import path",
                impact="Module cannot be imported",
                priority=8,
                estimated_fix_time="5-15 minutes",
            )
        except Exception:
            # Other import issues (circular imports, etc.)
            pass

    def _check_code_quality(
        self, file_path: Path, content: str, lines: List[str]
    ) -> None:
        """Check for code quality issues."""
        # Check for very long lines
        for line_no, line in enumerate(lines, 1):
            if len(line) > 120:
                self._add_problem(
                    severity=ProblemSeverity.LOW,
                    category=ProblemCategory.CODE_QUALITY,
                    title=f"Long Line in {file_path.name}",
                    description=f"Line {line_no} exceeds 120 characters ({len(line)} chars)",
                    location=str(file_path),
                    line_number=line_no,
                    suggested_fix="Break line into multiple lines",
                    impact="Reduced code readability",
                    priority=2,
                    estimated_fix_time="1 minute",
                )

        # Check for missing docstrings in classes and functions
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    if not ast.get_docstring(node):
                        self._add_problem(
                            severity=ProblemSeverity.LOW,
                            category=ProblemCategory.DOCUMENTATION,
                            title=f"Missing Docstring: {node.name}",
                            description=f"Function/class '{node.name}' lacks documentation",
                            location=str(file_path),
                            line_number=node.lineno,
                            suggested_fix="Add docstring explaining purpose and parameters",
                            impact="Reduced code maintainability",
                            priority=3,
                            estimated_fix_time="5 minutes",
                        )
        except Exception as e:
            logger.error(f"Error: {e}")

    def _analyze_configuration_files(self) -> None:
        """Analyze configuration files for issues."""
        print("Analyzing  Analyzing configuration files...")
        config_files = [
            "pyproject.toml",
            "setup.cfg",
            "tox.ini",
            ".flake8",
            ".pylintrc",
            "pytest.ini",
            "docker-compose.yml",
            "docker-compose.yaml",
            "Dockerfile",
        ]

        for config_file in config_files:
            file_path = self.project_root / config_file
            if file_path.exists():
                self._analyze_config_file(file_path)

    def _analyze_config_file(self, file_path: Path) -> None:
        """Analyze a configuration file."""
        try:
            with open(file_path, "r") as f:
                content = f.read()

            self.total_files_analyzed += 1
            self.total_lines_analyzed += len(content.split("\n"))
            # Check for common configuration issues
            if (
                file_path.name == "docker-compose.yml"
                or file_path.name == "docker-compose.yaml"
            ):
                self._check_docker_compose(file_path, content)
            elif file_path.name == "Dockerfile":
                self._check_dockerfile(file_path, content)
        except Exception as e:
            self._add_problem(
                severity=ProblemSeverity.MEDIUM,
                category=ProblemCategory.CONFIGURATION_ERROR,
                title=f"Cannot Read Config File: {file_path.name}",
                description=f"Error reading configuration file: {str(e)}",
                location=str(file_path),
                suggested_fix="Check file permissions and encoding",
                impact="Configuration may not be applied",
                priority=5,
                estimated_fix_time="5 minutes",
            )

    def _check_docker_compose(self, file_path: Path, content: str) -> None:
        """Check Docker Compose file for issues."""
        # Check for port conflicts
        ports = re.findall(r"(\d+):\d+", content)
        port_counts = {}
        for port in ports:
            port_counts[port] = port_counts.get(port, 0) + 1

        for port, count in port_counts.items():
            if count > 1:
                self._add_problem(
                    severity=ProblemSeverity.HIGH,
                    category=ProblemCategory.CONFIGURATION_ERROR,
                    title=f"Port Conflict: {port}",
                    description=f"Port {port} is used by multiple services",
                    location=str(file_path),
                    suggested_fix="Use different ports for each service",
                    impact="Services may fail to start",
                    priority=8,
                    estimated_fix_time="5 minutes",
                )

    def _check_dockerfile(self, file_path: Path, content: str) -> None:
        """Check Dockerfile for issues."""
        lines = content.split("\n")
        # Check for FROM instruction
        if not any(line.strip().startswith("FROM") for line in lines):
            self._add_problem(
                severity=ProblemSeverity.CRITICAL,
                category=ProblemCategory.CONFIGURATION_ERROR,
                title="Missing FROM Instruction",
                description="Dockerfile lacks FROM instruction",
                location=str(file_path),
                suggested_fix="Add FROM instruction with base image",
                impact="Docker build will fail",
                priority=10,
                estimated_fix_time="2 minutes",
            )

    def _analyze_documentation(self) -> None:
        """Analyze documentation completeness."""
        print("Analyzing documentation...")
        # Check for README
        readme_files = ["README.md", "README.rst", "README.txt"]
        has_readme = any(
            (self.project_root / readme).exists() for readme in readme_files
        )

        if not has_readme:
            self._add_problem(
                severity=ProblemSeverity.HIGH,
                category=ProblemCategory.DOCUMENTATION,
                title="Missing README",
                description="No README file found in project root",
                location=str(self.project_root),
                suggested_fix="Create README.md with project description",
                impact="Users won't understand how to use the project",
                priority=7,
                estimated_fix_time="30 minutes",
            )

    def _analyze_tests(self) -> None:
        """Analyze test coverage and quality."""
        print("Analyzing tests...")
        # Check for test directory
        test_dirs = ["tests", "test", "src/tests"]
        has_tests = any(
            (self.project_root / test_dir).exists() for test_dir in test_dirs
        )

        if not has_tests:
            self._add_problem(
                severity=ProblemSeverity.HIGH,
                category=ProblemCategory.TESTING,
                title="Missing Tests Directory",
                description="No tests directory found",
                location=str(self.project_root),
                suggested_fix="Create tests/ directory with test files",
                impact="No automated testing available",
                priority=6,
                estimated_fix_time="1 hour",
            )

        # Check for pytest configuration
        pytest_configs = ["pytest.ini", "pyproject.toml", "tox.ini", "setup.cfg"]
        has_pytest_config = any(
            (self.project_root / config).exists() for config in pytest_configs
        )

        if not has_pytest_config:
            self._add_problem(
                severity=ProblemSeverity.MEDIUM,
                category=ProblemCategory.TESTING,
                title="Missing Pytest Configuration",
                description="No pytest configuration found",
                location=str(self.project_root),
                suggested_fix="Add pytest.ini or configure pytest in pyproject.toml",
                impact="Test configuration may be suboptimal",
                priority=4,
                estimated_fix_time="10 minutes",
            )

    def _analyze_security(self) -> None:
        """Analyze security issues."""
        print("Analyzing security...")
        # Check for exposed secrets
        secret_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']',
        ]

        for py_file in self.project_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue

            try:
                with open(py_file, "r") as f:
                    content = f.read()

                for pattern in secret_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_no = content[: match.start()].count("\n") + 1
                        self._add_problem(
                            severity=ProblemSeverity.HIGH,
                            category=ProblemCategory.SECURITY_VULNERABILITY,
                            title=f"Potential Secret Exposure in {py_file.name}",
                            description="Hardcoded secret detected in source code",
                            location=str(py_file),
                            line_number=line_no,
                            suggested_fix="Move secret to environment variable",
                            impact="Secrets may be exposed in version control",
                            priority=9,
                            estimated_fix_time="5 minutes",
                        )
            except Exception as e:
                logger.error(f"Error: {e}")

    def _analyze_performance(self) -> None:
        """Analyze performance issues."""
        print("Analyzing performance...")
        # Check for common performance anti-patterns
        performance_patterns = [
            (
                r"for\s+\w+\s+in\s+range\(len\([^)]+\)\):",
                "Use enumerate() instead of range(len()",
            ),
            (
                r"\.append\([^)]+\)\s*$",
                "Consider list comprehension for better performance",
            ),
        ]

        for py_file in self.project_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue

            try:
                with open(py_file, "r") as f:
                    content = f.read()

                for pattern, suggestion in performance_patterns:
                    matches = re.finditer(pattern, content, re.MULTILINE)
                    for match in matches:
                        line_no = content[: match.start()].count("\n") + 1
                        self._add_problem(
                            severity=ProblemSeverity.LOW,
                            category=ProblemCategory.PERFORMANCE_ISSUE,
                            title=f"Performance Issue in {py_file.name}",
                            description=f"Potential performance improvement: {suggestion}",
                            location=str(py_file),
                            line_number=line_no,
                            suggested_fix=suggestion,
                            impact="Minor performance impact",
                            priority=2,
                            estimated_fix_time="2 minutes",
                        )
            except Exception as e:
                logger.error(f"Error: {e}")

    def _analyze_deployment(self) -> None:
        """Analyze deployment configuration."""
        print("Analyzing deployment configuration...")
        # Check for deployment files
        deployment_files = [
            "Dockerfile",
            "docker-compose.yml",
            "docker-compose.yaml",
            ".github/workflows",
            "k8s",
            "kubernetes",
            "helm",
        ]

        has_deployment = any(
            (self.project_root / dep_file).exists() for dep_file in deployment_files
        )

        if not has_deployment:
            self._add_problem(
                severity=ProblemSeverity.MEDIUM,
                category=ProblemCategory.DEPLOYMENT,
                title="Missing Deployment Configuration",
                description="No deployment configuration found",
                location=str(self.project_root),
                suggested_fix="Add Dockerfile and docker-compose.yml",
                impact="Cannot deploy application easily",
                priority=5,
                estimated_fix_time="2 hours",
            )

    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if a file should be skipped during analysis."""
        file_str = str(file_path)
        for exclude_pattern in self.exclude_patterns:
            if exclude_pattern.replace("*", "") in file_str:
                return True

        return False

    def _add_problem(self, **kwargs) -> None:
        """Add a problem to the list."""
        problem = Problem(**kwargs)
        self.problems.append(problem)

    def _generate_report(
        self, system_info: Dict[str, Any], analysis_duration: float
    ) -> DiagnosticReport:
        """Generate the final diagnostic report."""
        # Count problems by severity and category
        problems_by_severity = {}
        problems_by_category = {}

        for problem in self.problems:
            problems_by_severity[problem.severity] = (
                problems_by_severity.get(problem.severity, 0) + 1
            )
            problems_by_category[problem.category] = (
                problems_by_category.get(problem.category, 0) + 1
            )

        # Generate recommendations
        recommendations = self._generate_recommendations()

        # Analysis summary
        analysis_summary = {
            "critical_issues": problems_by_severity.get(ProblemSeverity.CRITICAL, 0),
            "high_priority_issues": problems_by_severity.get(ProblemSeverity.HIGH, 0),
            "total_files_with_issues": len(set(p.location for p in self.problems)),
            "most_common_category": max(
                problems_by_category.items(), key=lambda x: x[1]
            )[0].value
            if problems_by_category
            else None,
            "estimated_total_fix_time": self._calculate_total_fix_time(),
        }

        return DiagnosticReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            total_problems=len(self.problems),
            problems_by_severity=problems_by_severity,
            problems_by_category=problems_by_category,
            problems=sorted(
                self.problems,
                key=lambda p: (p.priority, p.severity.value),
                reverse=True,
            ),
            system_info=system_info,
            analysis_summary=analysis_summary,
            recommendations=recommendations,
            total_files_analyzed=self.total_files_analyzed,
            total_lines_analyzed=self.total_lines_analyzed,
            analysis_duration=analysis_duration,
        )

    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on found problems."""
        recommendations = []

        critical_count = sum(
            1 for p in self.problems if p.severity == ProblemSeverity.CRITICAL
        )
        high_count = sum(1 for p in self.problems if p.severity == ProblemSeverity.HIGH)
        if critical_count > 0:
            recommendations.append(
                f"URGENT URGENT: Fix {critical_count} critical issues immediately - system may not function"
            )

        if high_count > 0:
            recommendations.append(
                f"WARNING:  HIGH PRIORITY: Address {high_count} high-priority issues to prevent major problems"
            )

        # Category-specific recommendations
        category_counts = {}
        for problem in self.problems:
            category_counts[problem.category] = (
                category_counts.get(problem.category, 0) + 1
            )

        if category_counts.get(ProblemCategory.SYNTAX_ERROR, 0) > 0:
            recommendations.append(
                "Fix Fix syntax errors first - they prevent code execution"
            )

        if category_counts.get(ProblemCategory.IMPORT_ERROR, 0) > 0:
            recommendations.append(
                "Analyzing Resolve import errors by installing missing dependencies"
            )

        if category_counts.get(ProblemCategory.SECURITY_VULNERABILITY, 0) > 0:
            recommendations.append(
                "Analyzing Address security vulnerabilities to protect against attacks"
            )

        if category_counts.get(ProblemCategory.TESTING, 0) > 0:
            recommendations.append(
                "Analyzing Improve test coverage to catch bugs early"
            )
        if category_counts.get(ProblemCategory.DOCUMENTATION, 0) > 0:
            recommendations.append(
                "Analyzing Add documentation to improve maintainability"
            )
        recommendations.append(
            "Focus Focus on high-priority issues first for maximum impact"
        )
        recommendations.append(
            "TIME:  Estimated total fix time: " + self._calculate_total_fix_time()
        )

        return recommendations

    def _calculate_total_fix_time(self) -> str:
        """Calculate estimated total fix time."""
        total_minutes = 0

        for problem in self.problems:
            est = problem.estimated_fix_time
            if est:
                # Parse estimates like "5 minutes", "1 hour", "30 minutes"
                time_str = est.lower()
                match = re.search(r"(\d+)", time_str)
                if not match:
                    continue
                value = int(match.group(1))
                if "hour" in time_str:
                    total_minutes += value * 60
                elif "minute" in time_str:
                    total_minutes += value

        if total_minutes < 60:
            return f"{total_minutes} minutes"
        else:
            hours = total_minutes // 60
            minutes = total_minutes % 60
            return f"{hours} hours {minutes} minutes"

    def save_report_to_file(
        self,
        report: DiagnosticReport,
        filename: str = "llamaagent_diagnostic_report.txt",
    ) -> str:
        """Save the diagnostic report to a text file."""
        output_path = self.project_root / filename

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self._format_report_as_text(report))
        return str(output_path)

    def _format_report_as_text(self, report: DiagnosticReport) -> str:
        """Format the diagnostic report as a comprehensive text document."""
        lines = []

        # Header
        lines.append("=" * 80)
        lines.append("LLAMAAGENT COMPREHENSIVE DIAGNOSTIC REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {report.timestamp}")
        lines.append(f"Analysis Duration: {report.analysis_duration:.2f} seconds")
        lines.append(f"Files Analyzed: {report.total_files_analyzed}")
        lines.append(f"Lines Analyzed: {report.total_lines_analyzed}")
        lines.append("")
        # Executive Summary
        lines.append("EXECUTIVE SUMMARY")
        lines.append("-" * 50)
        lines.append(f"Total Issues Found: {report.total_problems}")
        lines.append(
            f"Critical Issues: {report.problems_by_severity.get(ProblemSeverity.CRITICAL, 0)}"
        )
        lines.append(
            f"High Priority Issues: {report.problems_by_severity.get(ProblemSeverity.HIGH, 0)}"
        )
        lines.append(
            f"Medium Priority Issues: {report.problems_by_severity.get(ProblemSeverity.MEDIUM, 0)}"
        )
        lines.append(
            f"Low Priority Issues: {report.problems_by_severity.get(ProblemSeverity.LOW, 0)}"
        )
        lines.append("")
        # System Information
        lines.append("SYSTEM INFORMATION")
        lines.append("-" * 50)
        lines.append(
            f"Python Version: {report.system_info.get('python_version', 'Unknown')}"
        )
        lines.append(f"Platform: {report.system_info.get('platform', 'Unknown')}")
        lines.append(
            f"Project Root: {report.system_info.get('project_root', 'Unknown')}"
        )
        lines.append("")
        # Recommendations
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 50)
        for i, recommendation in enumerate(report.recommendations, 1):
            lines.append(f"{i}. {recommendation}")
        lines.append("")
        # Detailed Problems
        lines.append("DETAILED PROBLEM ANALYSIS")
        lines.append("=" * 80)
        # Group problems by severity
        for severity in [
            ProblemSeverity.CRITICAL,
            ProblemSeverity.HIGH,
            ProblemSeverity.MEDIUM,
            ProblemSeverity.LOW,
        ]:
            severity_problems = [p for p in report.problems if p.severity == severity]
            if not severity_problems:
                continue

            lines.append("")
            lines.append(
                f"{severity.value} PRIORITY ISSUES ({len(severity_problems)} issues)"
            )
            lines.append("=" * 60)
            for i, problem in enumerate(severity_problems, 1):
                lines.append("")
                lines.append(f"{i}. {problem.title}")
                lines.append(f"   Category: {problem.category.value}")
                lines.append(f"   Location: {problem.location}")
                if problem.line_number:
                    lines.append(f"   Line: {problem.line_number}")
                lines.append(f"   Priority: {problem.priority}/10")
                lines.append(f"   Estimated Fix Time: {problem.estimated_fix_time}")
                lines.append(f"   Description: {problem.description}")
                lines.append(f"   Impact: {problem.impact}")
                if problem.suggested_fix:
                    lines.append(f"   Suggested Fix: {problem.suggested_fix}")
                if problem.code_snippet:
                    lines.append(f"   Code Snippet: {problem.code_snippet}")
                if problem.related_files:
                    lines.append(
                        f"   Related Files: {', '.join(problem.related_files)}"
                    )
                lines.append("   " + "-" * 60)
        # Statistics
        lines.append("")
        lines.append("STATISTICS BY CATEGORY")
        lines.append("-" * 50)
        for category, count in sorted(
            report.problems_by_category.items(), key=lambda x: x[1], reverse=True
        ):
            lines.append(f"{category.value}: {count} issues")
        lines.append("")
        lines.append("ANALYSIS SUMMARY")
        lines.append("-" * 50)
        for key, value in report.analysis_summary.items():
            lines.append(f"{key.replace('_', ' ').title()}: {value}")

        # Footer
        lines.append("")
        lines.append("=" * 80)
        lines.append("END OF DIAGNOSTIC REPORT")
        lines.append("=" * 80)
        return "\n".join(lines)


def main():
    """Main function to run comprehensive diagnostics."""
    diagnostics = MasterDiagnostics()
    report = diagnostics.run_comprehensive_analysis()

    # Save report to file
    output_file = diagnostics.save_report_to_file(report)
    print(f"\nSUCCESS Comprehensive diagnostic report saved to: {output_file}")
    print(
        f"Found Found {report.total_problems} issues across {report.total_files_analyzed} files"
    )
    print(f"TIME:  Estimated total fix time: {diagnostics._calculate_total_fix_time()}")

    # Print summary
    critical_count = report.problems_by_severity.get(ProblemSeverity.CRITICAL, 0)
    high_count = report.problems_by_severity.get(ProblemSeverity.HIGH, 0)
    if critical_count > 0:
        print(
            f"URGENT URGENT: {critical_count} critical issues need immediate attention!"
        )
    if high_count > 0:
        print(f"WARNING:  WARNING: {high_count} high-priority issues found!")
    return output_file


if __name__ == "__main__":
    main()

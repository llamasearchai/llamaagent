"""
Dependency Checker - Project Dependency Analysis

Analyzes project dependencies for version conflicts, security vulnerabilities,
and missing packages.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


class DependencyChecker:
    """Comprehensive dependency analysis tool."""

    def __init__(self, project_root: Optional[str] = None):
        """Initialize the dependency checker."""
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.installed_packages = {}
        self.requirements = {}

    def analyze_dependencies(self) -> Dict[str, Any]:
        """Perform comprehensive dependency analysis."""
        results = {
            "installed_packages": self._get_installed_packages(),
            "requirements_files": self._analyze_requirements_files(),
            "missing_dependencies": [],
            "version_conflicts": [],
            "security_vulnerabilities": [],
            "outdated_packages": [],
            "unused_packages": [],
            "analysis_summary": {},
        }

        # Analyze each aspect
        results["missing_dependencies"] = self._find_missing_dependencies()
        results["version_conflicts"] = self._find_version_conflicts()
        results["outdated_packages"] = self._find_outdated_packages()

        # Generate summary
        results["analysis_summary"] = self._generate_summary(results)
        return results

    def _get_installed_packages(self) -> Dict[str, str]:
        """Get list of installed packages and their versions."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                check=True,
            )
            import json

            packages = json.loads(result.stdout)
            return {pkg["name"].lower(): pkg["version"] for pkg in packages}

        except Exception:
            return {}

    def _analyze_requirements_files(self) -> Dict[str, Any]:
        """Analyze all requirements files in the project."""
        req_files = [
            "requirements.txt",
            "requirements-dev.txt",
            "requirements-test.txt",
            "requirements-prod.txt",
            "dev-requirements.txt",
        ]

        analysis = {}

        for req_file in req_files:
            file_path = self.project_root / req_file
            if file_path.exists():
                analysis[req_file] = self._parse_requirements_file(file_path)
        # Also check pyproject.toml
        pyproject_path = self.project_root / "pyproject.toml"
        if pyproject_path.exists():
            analysis["pyproject.toml"] = self._parse_pyproject_toml(pyproject_path)
        # Check setup.py
        setup_path = self.project_root / "setup.py"
        if setup_path.exists():
            analysis["setup.py"] = self._parse_setup_py(setup_path)
        return analysis

    def _parse_requirements_file(self, file_path: Path) -> Dict[str, Any]:
        """Parse a requirements.txt file."""
        try:
            with open(file_path, "r") as f:
                lines = f.readlines()

            requirements = []
            issues = []

            for line_no, line in enumerate(lines, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Parse requirement
                req_info = self._parse_requirement_line(line, line_no)
                if req_info:
                    requirements.append(req_info)
                else:
                    issues.append(
                        {
                            "line": line_no,
                            "content": line,
                            "issue": "Invalid requirement format",
                        }
                    )
            return {
                "file_path": str(file_path),
                "requirements": requirements,
                "issues": issues,
                "total_requirements": len(requirements),
            }

        except Exception as e:
            return {
                "file_path": str(file_path),
                "error": str(e),
                "requirements": [],
                "issues": [],
                "total_requirements": 0,
            }

    def _parse_requirement_line(
        self, line: str, line_no: int
    ) -> Optional[Dict[str, Any]]:
        """Parse a single requirement line."""
        import re

        # Handle git URLs and other complex requirements
        if line.startswith("git+") or line.startswith("http"):
            return {
                "name": "external_dependency",
                "version_spec": line,
                "line": line_no,
                "type": "external",
            }

        # Parse standard requirements
        # Examples: package==1.0.0, package>=1.0, package~=1.0
        pattern = r"^([a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9])\s*(.*?)(?:\s*#.*)?$"
        match = re.match(pattern, line)
        if match:
            name = match.group(1)
            version_spec = match.group(2).strip()

            return {
                "name": name.lower(),
                "version_spec": version_spec,
                "line": line_no,
                "type": "standard",
                "is_pinned": "==" in version_spec,
                "has_version_constraint": bool(version_spec),
            }

        return None

    def _parse_pyproject_toml(self, file_path: Path) -> Dict[str, Any]:
        """Parse dependencies from pyproject.toml."""
        try:
            import tomli

            with open(file_path, "rb") as f:
                data = tomli.load(f)
            dependencies = []

            # Get main dependencies
            if "project" in data and "dependencies" in data["project"]:
                for dep in data["project"]["dependencies"]:
                    req_info = self._parse_requirement_line(dep, 0)
                    if req_info:
                        dependencies.append(req_info)
            # Get optional dependencies
            optional_deps = []
            if "project" in data and "optional-dependencies" in data["project"]:
                for group, deps in data["project"]["optional-dependencies"].items():
                    for dep in deps:
                        req_info = self._parse_requirement_line(dep, 0)
                        if req_info:
                            req_info["group"] = group
                            optional_deps.append(req_info)
            return {
                "file_path": str(file_path),
                "dependencies": dependencies,
                "optional_dependencies": optional_deps,
                "total_dependencies": len(dependencies) + len(optional_deps),
            }

        except ImportError:
            return {
                "file_path": str(file_path),
                "error": "tomli package required to parse pyproject.toml",
                "dependencies": [],
                "optional_dependencies": [],
                "total_dependencies": 0,
            }
        except Exception as e:
            return {
                "file_path": str(file_path),
                "error": str(e),
                "dependencies": [],
                "optional_dependencies": [],
                "total_dependencies": 0,
            }

    def _parse_setup_py(self, file_path: Path) -> Dict[str, Any]:
        """Parse dependencies from setup.py (basic parsing)."""
        try:
            with open(file_path, "r") as f:
                content = f.read()

            # Look for install_requires
            import re

            pattern = r"install_requires\s*=\s*\[(.*?)\]"
            match = re.search(pattern, content, re.DOTALL)
            dependencies = []
            if match:
                deps_text = match.group(1)
                # Extract quoted strings
                dep_pattern = r'["\']([^"\']+)["\']'
                deps = re.findall(dep_pattern, deps_text)
                for dep in deps:
                    req_info = self._parse_requirement_line(dep, 0)
                    if req_info:
                        dependencies.append(req_info)
            return {
                "file_path": str(file_path),
                "dependencies": dependencies,
                "total_dependencies": len(dependencies),
                "note": "Basic parsing - may miss complex setups",
            }

        except Exception as e:
            return {
                "file_path": str(file_path),
                "error": str(e),
                "dependencies": [],
                "total_dependencies": 0,
            }

    def _find_missing_dependencies(self) -> List[Dict[str, Any]]:
        """Find dependencies that are required but not installed."""
        missing = []
        installed = self._get_installed_packages()

        # Check all requirements files
        req_analysis = self._analyze_requirements_files()

        for file_name, file_data in req_analysis.items():
            if "error" in file_data:
                continue

            requirements = file_data.get("requirements", [])
            if "dependencies" in file_data:  # pyproject.toml format
                requirements.extend(file_data["dependencies"])
            for req in requirements:
                if req["type"] == "standard":
                    pkg_name = req["name"].lower()
                    if pkg_name not in installed:
                        missing.append(
                            {
                                "package": req["name"],
                                "version_spec": req["version_spec"],
                                "source_file": file_name,
                                "line": req.get("line", 0),
                            }
                        )

        return missing

    def _find_version_conflicts(self) -> List[Dict[str, Any]]:
        """Find version conflicts between requirements and installed packages."""
        conflicts = []
        installed = self._get_installed_packages()

        req_analysis = self._analyze_requirements_files()

        for file_name, file_data in req_analysis.items():
            if "error" in file_data:
                continue

            requirements = file_data.get("requirements", [])
            if "dependencies" in file_data:
                requirements.extend(file_data["dependencies"])
            for req in requirements:
                if req["type"] == "standard" and req["version_spec"]:
                    pkg_name = req["name"].lower()
                    if pkg_name in installed:
                        installed_version = installed[pkg_name]
                        if not self._version_satisfies(
                            installed_version, req["version_spec"]
                        ):
                            conflicts.append(
                                {
                                    "package": req["name"],
                                    "required_version": req["version_spec"],
                                    "installed_version": installed_version,
                                    "source_file": file_name,
                                    "line": req.get("line", 0),
                                }
                            )

        return conflicts

    def _find_outdated_packages(self) -> List[Dict[str, Any]]:
        """Find packages that have newer versions available."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True,
                check=True,
            )
            import json

            outdated = json.loads(result.stdout)
            return [
                {
                    "package": pkg["name"],
                    "current_version": pkg["version"],
                    "latest_version": pkg["latest_version"],
                    "latest_filetype": pkg.get("latest_filetype", "unknown"),
                }
                for pkg in outdated
            ]

        except Exception:
            return []

    def _version_satisfies(self, installed_version: str, version_spec: str) -> bool:
        """Check if installed version satisfies the requirement specification."""
        if not version_spec:
            return True

        # Simple version checking - would need packaging library for proper implementation
        if "==" in version_spec:
            required = version_spec.replace("==", "").strip()
            return installed_version == required
        elif ">=" in version_spec:
            required = version_spec.replace(">=", "").strip()
            return self._compare_versions(installed_version, required) >= 0
        elif "<=" in version_spec:
            required = version_spec.replace("<=", "").strip()
            return self._compare_versions(installed_version, required) <= 0
        elif ">" in version_spec:
            required = version_spec.replace(">", "").strip()
            return self._compare_versions(installed_version, required) > 0
        elif "<" in version_spec:
            required = version_spec.replace("<", "").strip()
            return self._compare_versions(installed_version, required) < 0

        return True  # Default to satisfied if we can't parse

    def _compare_versions(self, v1: str, v2: str) -> int:
        """Simple version comparison. Returns -1, 0, or 1."""
        try:
            parts1 = [int(x) for x in v1.split(".")]
            parts2 = [int(x) for x in v2.split(".")]

            # Pad shorter version with zeros
            max_len = max(len(parts1), len(parts2))
            parts1.extend([0] * (max_len - len(parts1)))
            parts2.extend([0] * (max_len - len(parts2)))
            for p1, p2 in zip(parts1, parts2, strict=False):
                if p1 < p2:
                    return -1
                elif p1 > p2:
                    return 1

            return 0

        except ValueError:
            # If we can't parse as numbers, use string comparison
            if v1 < v2:
                return -1
            elif v1 > v2:
                return 1
            return 0

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis summary."""
        return {
            "total_installed_packages": len(results["installed_packages"]),
            "total_requirements_files": len(results["requirements_files"]),
            "total_missing_dependencies": len(results["missing_dependencies"]),
            "total_version_conflicts": len(results["version_conflicts"]),
            "total_outdated_packages": len(results["outdated_packages"]),
            "health_score": self._calculate_health_score(results),
        }

    def _calculate_health_score(self, results: Dict[str, Any]) -> float:
        """Calculate dependency health score (0-100)."""
        score = 100.0

        # Deduct points for issues
        score -= len(results["missing_dependencies"]) * 10  # -10 per missing dep
        score -= len(results["version_conflicts"]) * 15  # -15 per conflict
        score -= len(results["outdated_packages"]) * 2  # -2 per outdated package

        return max(0.0, score)

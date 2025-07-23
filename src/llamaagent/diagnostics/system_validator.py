"""
System Validator - Environment and Configuration Validation

Validates system environment, configuration files, and deployment readiness.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional


class SystemValidator:
    """Comprehensive system environment validator."""

    def __init__(self, project_root: Optional[str] = None):
        """Initialize the system validator."""
        self.project_root = Path(project_root) if project_root else Path.cwd()

    def validate_system(self) -> Dict[str, Any]:
        """Perform comprehensive system validation."""
        results = {
            "python_environment": self._validate_python_environment(),
            "system_resources": self._validate_system_resources(),
            "required_tools": self._validate_required_tools(),
            "configuration_files": self._validate_configuration_files(),
            "environment_variables": self._validate_environment_variables(),
            "file_permissions": self._validate_file_permissions(),
            "network_connectivity": self._validate_network_connectivity(),
            "deployment_readiness": self._validate_deployment_readiness(),
        }

        # Calculate overall health score
        results["overall_health_score"] = self._calculate_health_score(results)

        return results

    def _validate_python_environment(self) -> Dict[str, Any]:
        """Validate Python environment and version."""
        result = {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "architecture": platform.architecture(),
            "issues": [],
            "score": 100,
        }

        # Check Python version
        version_info = sys.version_info
        if version_info.major < 3:
            result["issues"].append(
                {
                    "severity": "CRITICAL",
                    "message": "Python 2 is not supported",
                    "fix": "Upgrade to Python 3.8 or higher",
                }
            )
            result["score"] -= 50
        elif version_info.minor < 8:
            result["issues"].append(
                {
                    "severity": "HIGH",
                    "message": f"Python {version_info.major}.{version_info.minor} is outdated",
                    "fix": "Upgrade to Python 3.8 or higher",
                }
            )
            result["score"] -= 30

        # Check virtual environment
        if not hasattr(sys, "real_prefix") and not (
            hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
        ):
            result["issues"].append(
                {
                    "severity": "MEDIUM",
                    "message": "Not running in virtual environment",
                    "fix": "Create and activate virtual environment",
                }
            )
            result["score"] -= 20

        return result

    def _validate_system_resources(self) -> Dict[str, Any]:
        """Validate system resources (CPU, memory, disk)."""
        result = {
            "cpu_count": os.cpu_count(),
            "memory_info": {},
            "disk_info": {},
            "issues": [],
            "score": 100,
        }

        try:
            import psutil

            # Memory check
            memory = psutil.virtual_memory()
            result["memory_info"] = {
                "total": memory.total,
                "available": memory.available,
                "percent_used": memory.percent,
            }

            if memory.available < 1024 * 1024 * 1024:  # Less than 1GB available
                result["issues"].append(
                    {
                        "severity": "HIGH",
                        "message": f"Low available memory: {memory.available / (1024**3):.1f}GB",
                        "fix": "Free up memory or increase system RAM",
                    }
                )
                result["score"] -= 30

            # Disk check
            disk = psutil.disk_usage(str(self.project_root))
            result["disk_info"] = {
                "total": disk.total,
                "free": disk.free,
                "percent_used": (disk.used / disk.total) * 100,
            }

            if disk.free < 1024 * 1024 * 1024:  # Less than 1GB free
                result["issues"].append(
                    {
                        "severity": "HIGH",
                        "message": f"Low disk space: {disk.free / (1024**3):.1f}GB free",
                        "fix": "Free up disk space",
                    }
                )
                result["score"] -= 30

        except ImportError:
            result["issues"].append(
                {
                    "severity": "LOW",
                    "message": "psutil not available for detailed system info",
                    "fix": "Install psutil: pip install psutil",
                }
            )
            result["score"] -= 10

        return result

    def _validate_required_tools(self) -> Dict[str, Any]:
        """Validate required external tools."""
        required_tools = [
            ("git", "Git version control"),
            ("docker", "Docker containerization"),
            ("pip", "Python package manager"),
        ]

        result = {"tools": {}, "issues": [], "score": 100}

        for tool, description in required_tools:
            tool_path = shutil.which(tool)
            if tool_path:
                result["tools"][tool] = {
                    "available": True,
                    "path": tool_path,
                    "description": description,
                }
            else:
                result["tools"][tool] = {
                    "available": False,
                    "path": None,
                    "description": description,
                }

                if tool == "git":
                    result["issues"].append(
                        {
                            "severity": "HIGH",
                            "message": f"Required tool {tool} not found",
                            "fix": f"Install {tool}",
                        }
                    )
                    result["score"] -= 25
                elif tool == "docker":
                    result["issues"].append(
                        {
                            "severity": "MEDIUM",
                            "message": f"Tool {tool} not found (needed for deployment)",
                            "fix": f"Install {tool}",
                        }
                    )
                    result["score"] -= 15
                else:
                    result["issues"].append(
                        {
                            "severity": "LOW",
                            "message": f"Tool {tool} not found",
                            "fix": f"Install {tool}",
                        }
                    )
                    result["score"] -= 5

        return result

    def _validate_configuration_files(self) -> Dict[str, Any]:
        """Validate configuration files."""
        config_files = [
            ("requirements.txt", "Python dependencies", "HIGH"),
            ("pyproject.toml", "Project configuration", "MEDIUM"),
            ("setup.py", "Package setup", "MEDIUM"),
            (".gitignore", "Git ignore rules", "MEDIUM"),
            ("README.md", "Project documentation", "MEDIUM"),
            ("Dockerfile", "Docker configuration", "LOW"),
            ("docker-compose.yml", "Docker Compose configuration", "LOW"),
            (".env.example", "Environment variables template", "LOW"),
        ]

        result = {"files": {}, "issues": [], "score": 100}

        for filename, description, severity in config_files:
            file_path = self.project_root / filename
            if file_path.exists():
                result["files"][filename] = {
                    "exists": True,
                    "path": str(file_path),
                    "size": file_path.stat().st_size,
                    "description": description,
                }

                # Check if file is empty
                if file_path.stat().st_size == 0:
                    result["issues"].append(
                        {
                            "severity": severity,
                            "message": f"{filename} exists but is empty",
                            "fix": f"Add content to {filename}",
                        }
                    )
                    result["score"] -= 10
            else:
                result["files"][filename] = {
                    "exists": False,
                    "path": str(file_path),
                    "description": description,
                }

                if severity == "HIGH":
                    result["issues"].append(
                        {
                            "severity": severity,
                            "message": f"Critical file {filename} is missing",
                            "fix": f"Create {filename}",
                        }
                    )
                    result["score"] -= 25
                elif severity == "MEDIUM":
                    result["issues"].append(
                        {
                            "severity": severity,
                            "message": f"Important file {filename} is missing",
                            "fix": f"Create {filename}",
                        }
                    )
                    result["score"] -= 15
                else:
                    result["issues"].append(
                        {
                            "severity": severity,
                            "message": f"Optional file {filename} is missing",
                            "fix": f"Consider creating {filename}",
                        }
                    )
                    result["score"] -= 5

        return result

    def _validate_environment_variables(self) -> Dict[str, Any]:
        """Validate environment variables."""
        important_env_vars = [
            ("OPENAI_API_KEY", "OpenAI API access", "MEDIUM"),
            ("ANTHROPIC_API_KEY", "Anthropic API access", "LOW"),
            ("REDIS_URL", "Redis connection", "LOW"),
            ("DATABASE_URL", "Database connection", "LOW"),
            ("PYTHONPATH", "Python path configuration", "LOW"),
        ]

        result = {"variables": {}, "issues": [], "score": 100}

        for var_name, description, severity in important_env_vars:
            var_value = os.environ.get(var_name)
            if var_value:
                result["variables"][var_name] = {
                    "set": True,
                    "value_length": len(var_value),
                    "description": description,
                }
            else:
                result["variables"][var_name] = {
                    "set": False,
                    "value_length": 0,
                    "description": description,
                }

                if severity == "MEDIUM":
                    result["issues"].append(
                        {
                            "severity": severity,
                            "message": f"Environment variable {var_name} not set",
                            "fix": f"Set {var_name} environment variable",
                        }
                    )
                    result["score"] -= 15
                elif severity == "LOW":
                    result["issues"].append(
                        {
                            "severity": severity,
                            "message": f"Optional environment variable {var_name} not set",
                            "fix": f"Consider setting {var_name}",
                        }
                    )
                    result["score"] -= 5

        return result

    def _validate_file_permissions(self) -> Dict[str, Any]:
        """Validate file permissions."""
        result = {"permissions": {}, "issues": [], "score": 100}

        # Check key directories
        key_paths = [
            self.project_root,
            self.project_root / "src",
            self.project_root / "tests",
        ]

        for path in key_paths:
            if path.exists():
                path.stat()
                result["permissions"][str(path)] = {
                    "readable": os.access(path, os.R_OK),
                    "writable": os.access(path, os.W_OK),
                    "executable": os.access(path, os.X_OK) if path.is_dir() else False,
                }

                if not os.access(path, os.R_OK):
                    result["issues"].append(
                        {
                            "severity": "HIGH",
                            "message": f"Cannot read {path}",
                            "fix": f"Fix permissions for {path}",
                        }
                    )
                    result["score"] -= 30

                if path.is_dir() and not os.access(path, os.X_OK):
                    result["issues"].append(
                        {
                            "severity": "MEDIUM",
                            "message": f"Cannot execute/traverse {path}",
                            "fix": f"Fix execute permissions for {path}",
                        }
                    )
                    result["score"] -= 15

        return result

    def _validate_network_connectivity(self) -> Dict[str, Any]:
        """Validate network connectivity to required services."""
        result = {"connectivity": {}, "issues": [], "score": 100}

        # Test connections to common services
        services = [
            ("pypi.org", 443, "PyPI package repository"),
            ("api.openai.com", 443, "OpenAI API"),
            ("github.com", 443, "GitHub"),
        ]

        for host, port, description in services:
            try:
                import socket

                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result_code = sock.connect_ex((host, port))
                sock.close()

                if result_code == 0:
                    result["connectivity"][host] = {
                        "reachable": True,
                        "description": description,
                    }
                else:
                    result["connectivity"][host] = {
                        "reachable": False,
                        "description": description,
                    }
                    result["issues"].append(
                        {
                            "severity": "MEDIUM",
                            "message": f"Cannot reach {host}:{port} ({description})",
                            "fix": "Check network connectivity and firewall settings",
                        }
                    )
                    result["score"] -= 15

            except Exception as e:
                result["connectivity"][host] = {
                    "reachable": False,
                    "error": str(e),
                    "description": description,
                }
                result["issues"].append(
                    {
                        "severity": "LOW",
                        "message": f"Network test failed for {host}: {str(e)}",
                        "fix": "Check network configuration",
                    }
                )
                result["score"] -= 10

        return result

    def _validate_deployment_readiness(self) -> Dict[str, Any]:
        """Validate deployment readiness."""
        result = {"readiness_checks": {}, "issues": [], "score": 100}

        # Check Docker files
        docker_files = ["Dockerfile", "docker-compose.yml", "docker-compose.yaml"]
        has_docker = any((self.project_root / f).exists() for f in docker_files)

        result["readiness_checks"]["docker_configuration"] = {
            "ready": has_docker,
            "description": "Docker deployment configuration",
        }

        if not has_docker:
            result["issues"].append(
                {
                    "severity": "MEDIUM",
                    "message": "No Docker configuration found",
                    "fix": "Create Dockerfile and docker-compose.yml",
                }
            )
            result["score"] -= 20

        # Check for environment configuration
        env_files = [".env.example", ".env.template", "config.example.yml"]
        has_env_template = any((self.project_root / f).exists() for f in env_files)

        result["readiness_checks"]["environment_template"] = {
            "ready": has_env_template,
            "description": "Environment configuration template",
        }

        if not has_env_template:
            result["issues"].append(
                {
                    "severity": "LOW",
                    "message": "No environment template found",
                    "fix": "Create .env.example with required variables",
                }
            )
            result["score"] -= 10

        # Check for CI/CD configuration
        ci_files = [
            ".github/workflows",
            ".gitlab-ci.yml",
            "Jenkinsfile",
            "azure-pipelines.yml",
        ]
        has_ci = any((self.project_root / f).exists() for f in ci_files)

        result["readiness_checks"]["ci_cd"] = {
            "ready": has_ci,
            "description": "CI/CD pipeline configuration",
        }

        if not has_ci:
            result["issues"].append(
                {
                    "severity": "LOW",
                    "message": "No CI/CD configuration found",
                    "fix": "Setup GitHub Actions or other CI/CD pipeline",
                }
            )
            result["score"] -= 10

        return result

    def _calculate_health_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall system health score."""
        scores = []

        for category, data in results.items():
            if isinstance(data, dict) and "score" in data:
                scores.append(data["score"])

        return sum(scores) / len(scores) if scores else 0.0

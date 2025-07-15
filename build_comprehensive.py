#!/usr/bin/env python3
"""
Comprehensive Build Script for LlamaAgent

Complete automated build, test, and deployment pipeline:
- Code quality checks and linting
- Unit and integration testing
- Security scanning
- Performance testing
- Docker image building
- Documentation generation
- Distribution packaging
- Deployment orchestration

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import argparse
import asyncio
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('build.log')
    ]
)
logger = logging.getLogger(__name__)

# Build configuration
BUILD_CONFIG = {
    "project_name": "llamaagent",
    "version": "1.0.0",
    "python_version": "3.11",
    "docker_registry": "ghcr.io/nikjois",
    "test_timeout": 300,
    "coverage_threshold": 85.0,
    "security_severity": "high",
    "performance_benchmarks": True,
    "build_docs": True,
    "create_distribution": True,
    "push_images": False,
    "deploy_staging": False,
    "deploy_production": False,
}

# Environment setup
VENV_PATH = Path("venv")
DIST_PATH = Path("dist")
DOCS_PATH = Path("docs")
REPORTS_PATH = Path("reports")
DOCKER_PATH = Path("docker")


class ComprehensiveBuildSystem:
    """Complete build system for LlamaAgent"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.start_time = time.time()
        self.build_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.reports: Dict[str, Any] = {}
        
        # Create directories
        for path in [DIST_PATH, REPORTS_PATH, DOCS_PATH]:
            path.mkdir(exist_ok=True)
    
    def run_command(self, cmd: List[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> Tuple[int, str, str]:
        """Run a command and return exit code, stdout, stderr"""
        logger.info(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.config.get("command_timeout", 300)
            )
            
            if result.returncode != 0:
                logger.error(f"Command failed with exit code {result.returncode}")
                logger.error(f"STDERR: {result.stderr}")
            
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {' '.join(cmd)}")
            return 1, "", "Command timed out"
        except Exception as e:
            logger.error(f"Command failed with exception: {e}")
            return 1, "", str(e)
    
    def setup_environment(self) -> bool:
        """Set up Python virtual environment"""
        logger.info("Setting up virtual environment...")
        
        # Create virtual environment
        if not VENV_PATH.exists():
            exit_code, stdout, stderr = self.run_command([
                sys.executable, "-m", "venv", str(VENV_PATH)
            ])
            if exit_code != 0:
                logger.error("Failed to create virtual environment")
                return False
        
        # Activate virtual environment
        if os.name == "nt":
            pip_path = VENV_PATH / "Scripts" / "pip"
            python_path = VENV_PATH / "Scripts" / "python"
        else:
            pip_path = VENV_PATH / "bin" / "pip"
            python_path = VENV_PATH / "bin" / "python"
        
        # Upgrade pip
        exit_code, stdout, stderr = self.run_command([
            str(python_path), "-m", "pip", "install", "--upgrade", "pip"
        ])
        if exit_code != 0:
            logger.error("Failed to upgrade pip")
            return False
        
        # Install development dependencies
        dev_deps = [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-xdist>=3.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "bandit>=1.7.0",
            "safety>=2.0.0",
            "pre-commit>=3.0.0",
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "coverage>=7.0.0",
            "twine>=4.0.0",
            "build>=0.10.0",
            "wheel>=0.40.0",
            "setuptools>=65.0.0",
        ]
        
        exit_code, stdout, stderr = self.run_command([
            str(pip_path), "install"
        ] + dev_deps)
        if exit_code != 0:
            logger.error("Failed to install development dependencies")
            return False
        
        # Install project dependencies
        if Path("requirements.txt").exists():
            exit_code, stdout, stderr = self.run_command([
                str(pip_path), "install", "-r", "requirements.txt"
            ])
            if exit_code != 0:
                logger.error("Failed to install project dependencies")
                return False
        
        # Install project in development mode
        exit_code, stdout, stderr = self.run_command([
            str(pip_path), "install", "-e", "."
        ])
        if exit_code != 0:
            logger.error("Failed to install project in development mode")
            return False
        
        logger.info("Virtual environment setup complete")
        return True
    
    def code_quality_checks(self) -> bool:
        """Run code quality checks"""
        logger.info("Running code quality checks...")
        
        python_path = VENV_PATH / "bin" / "python" if os.name != "nt" else VENV_PATH / "Scripts" / "python"
        
        # Black formatting check
        logger.info("Running Black formatter check...")
        exit_code, stdout, stderr = self.run_command([
            str(python_path), "-m", "black", "--check", "--diff", "src", "tests"
        ])
        self.reports["black_check"] = {"exit_code": exit_code, "output": stdout}
        
        # isort import sorting check
        logger.info("Running isort import sorting check...")
        exit_code, stdout, stderr = self.run_command([
            str(python_path), "-m", "isort", "--check-only", "--diff", "src", "tests"
        ])
        self.reports["isort_check"] = {"exit_code": exit_code, "output": stdout}
        
        # Flake8 linting
        logger.info("Running flake8 linting...")
        exit_code, stdout, stderr = self.run_command([
            str(python_path), "-m", "flake8", "src", "tests", "--max-line-length=88", "--extend-ignore=E203,W503"
        ])
        self.reports["flake8_check"] = {"exit_code": exit_code, "output": stdout}
        
        # MyPy type checking
        logger.info("Running MyPy type checking...")
        exit_code, stdout, stderr = self.run_command([
            str(python_path), "-m", "mypy", "src", "--ignore-missing-imports"
        ])
        self.reports["mypy_check"] = {"exit_code": exit_code, "output": stdout}
        
        # Check if all quality checks passed
        quality_passed = all(
            self.reports[check]["exit_code"] == 0
            for check in ["black_check", "isort_check", "flake8_check", "mypy_check"]
        )
        
        if not quality_passed:
            logger.warning("Some code quality checks failed")
        else:
            logger.info("All code quality checks passed")
        
        return quality_passed
    
    def security_scanning(self) -> bool:
        """Run security scanning"""
        logger.info("Running security scanning...")
        
        python_path = VENV_PATH / "bin" / "python" if os.name != "nt" else VENV_PATH / "Scripts" / "python"
        
        # Bandit security linting
        logger.info("Running Bandit security scanning...")
        exit_code, stdout, stderr = self.run_command([
            str(python_path), "-m", "bandit", "-r", "src", "-f", "json", "-o", str(REPORTS_PATH / "bandit-report.json")
        ])
        self.reports["bandit_scan"] = {"exit_code": exit_code, "output": stdout}
        
        # Safety dependency scanning
        logger.info("Running Safety dependency scanning...")
        exit_code, stdout, stderr = self.run_command([
            str(python_path), "-m", "safety", "check", "--json", "--output", str(REPORTS_PATH / "safety-report.json")
        ])
        self.reports["safety_scan"] = {"exit_code": exit_code, "output": stdout}
        
        security_passed = self.reports["bandit_scan"]["exit_code"] == 0 and self.reports["safety_scan"]["exit_code"] == 0
        
        if not security_passed:
            logger.warning("Security scanning found issues")
        else:
            logger.info("Security scanning passed")
        
        return security_passed
    
    def run_tests(self) -> bool:
        """Run comprehensive test suite"""
        logger.info("Running comprehensive test suite...")
        
        python_path = VENV_PATH / "bin" / "python" if os.name != "nt" else VENV_PATH / "Scripts" / "python"
        
        # Run pytest with coverage
        logger.info("Running pytest with coverage...")
        exit_code, stdout, stderr = self.run_command([
            str(python_path), "-m", "pytest",
            "tests/",
            "--cov=src",
            "--cov-report=html:reports/coverage_html",
            "--cov-report=xml:reports/coverage.xml",
            "--cov-report=term-missing",
            "--cov-fail-under=" + str(self.config["coverage_threshold"]),
            "--junitxml=reports/junit.xml",
            "-v",
            "--tb=short",
            "-x"
        ])
        
        self.reports["pytest"] = {"exit_code": exit_code, "output": stdout}
        
        # Run specific comprehensive tests
        logger.info("Running comprehensive functionality tests...")
        exit_code, stdout, stderr = self.run_command([
            str(python_path), "-m", "pytest",
            "tests/test_comprehensive_functionality.py",
            "-v",
            "--tb=short"
        ])
        
        self.reports["comprehensive_tests"] = {"exit_code": exit_code, "output": stdout}
        
        tests_passed = self.reports["pytest"]["exit_code"] == 0 and self.reports["comprehensive_tests"]["exit_code"] == 0
        
        if not tests_passed:
            logger.error("Test suite failed")
        else:
            logger.info("All tests passed")
        
        return tests_passed
    
    def build_documentation(self) -> bool:
        """Build comprehensive documentation"""
        if not self.config["build_docs"]:
            logger.info("Skipping documentation build")
            return True
        
        logger.info("Building documentation...")
        
        python_path = VENV_PATH / "bin" / "python" if os.name != "nt" else VENV_PATH / "Scripts" / "python"
        
        # Create Sphinx documentation
        docs_source = DOCS_PATH / "source"
        docs_build = DOCS_PATH / "build"
        
        docs_source.mkdir(exist_ok=True)
        docs_build.mkdir(exist_ok=True)
        
        # Generate API documentation
        exit_code, stdout, stderr = self.run_command([
            str(python_path), "-m", "sphinx.apidoc",
            "-f", "-o", str(docs_source), "src"
        ])
        
        if exit_code != 0:
            logger.error("Failed to generate API documentation")
            return False
        
        # Build HTML documentation
        exit_code, stdout, stderr = self.run_command([
            str(python_path), "-m", "sphinx",
            "-b", "html", str(docs_source), str(docs_build)
        ])
        
        self.reports["docs_build"] = {"exit_code": exit_code, "output": stdout}
        
        docs_passed = self.reports["docs_build"]["exit_code"] == 0
        
        if not docs_passed:
            logger.error("Documentation build failed")
        else:
            logger.info("Documentation built successfully")
        
        return docs_passed
    
    def build_docker_images(self) -> bool:
        """Build Docker images"""
        logger.info("Building Docker images...")
        
        # Build main application image
        exit_code, stdout, stderr = self.run_command([
            "docker", "build",
            "-t", f"{self.config['docker_registry']}/{self.config['project_name']}:latest",
            "-t", f"{self.config['docker_registry']}/{self.config['project_name']}:{self.config['version']}",
            "-f", "Dockerfile.complete",
            "."
        ])
        
        self.reports["docker_build"] = {"exit_code": exit_code, "output": stdout}
        
        if exit_code != 0:
            logger.error("Docker build failed")
            return False
        
        # Build development image
        exit_code, stdout, stderr = self.run_command([
            "docker", "build",
            "-t", f"{self.config['docker_registry']}/{self.config['project_name']}:dev",
            "-f", "Dockerfile",
            "."
        ])
        
        if exit_code != 0:
            logger.warning("Development Docker build failed")
        
        logger.info("Docker images built successfully")
        return True
    
    def create_distribution(self) -> bool:
        """Create distribution packages"""
        if not self.config["create_distribution"]:
            logger.info("Skipping distribution creation")
            return True
        
        logger.info("Creating distribution packages...")
        
        python_path = VENV_PATH / "bin" / "python" if os.name != "nt" else VENV_PATH / "Scripts" / "python"
        
        # Clean previous distributions
        if DIST_PATH.exists():
            shutil.rmtree(DIST_PATH)
        DIST_PATH.mkdir(exist_ok=True)
        
        # Build wheel and source distribution
        exit_code, stdout, stderr = self.run_command([
            str(python_path), "-m", "build", "--outdir", str(DIST_PATH)
        ])
        
        self.reports["distribution_build"] = {"exit_code": exit_code, "output": stdout}
        
        if exit_code != 0:
            logger.error("Distribution build failed")
            return False
        
        # Check distribution
        exit_code, stdout, stderr = self.run_command([
            str(python_path), "-m", "twine", "check", str(DIST_PATH / "*")
        ])
        
        if exit_code != 0:
            logger.error("Distribution check failed")
            return False
        
        logger.info("Distribution packages created successfully")
        return True
    
    def performance_benchmarks(self) -> bool:
        """Run performance benchmarks"""
        if not self.config["performance_benchmarks"]:
            logger.info("Skipping performance benchmarks")
            return True
        
        logger.info("Running performance benchmarks...")
        
        python_path = VENV_PATH / "bin" / "python" if os.name != "nt" else VENV_PATH / "Scripts" / "python"
        
        # Run benchmark tests
        exit_code, stdout, stderr = self.run_command([
            str(python_path), "-m", "pytest",
            "tests/performance/",
            "-v",
            "--tb=short"
        ])
        
        self.reports["performance_benchmarks"] = {"exit_code": exit_code, "output": stdout}
        
        benchmarks_passed = self.reports["performance_benchmarks"]["exit_code"] == 0
        
        if not benchmarks_passed:
            logger.warning("Performance benchmarks failed")
        else:
            logger.info("Performance benchmarks passed")
        
        return benchmarks_passed
    
    def generate_build_report(self) -> None:
        """Generate comprehensive build report"""
        logger.info("Generating build report...")
        
        build_time = time.time() - self.start_time
        
        report = {
            "build_id": self.build_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "build_time": build_time,
            "config": self.config,
            "reports": self.reports,
            "summary": {
                "total_checks": len(self.reports),
                "passed_checks": sum(1 for r in self.reports.values() if r.get("exit_code") == 0),
                "failed_checks": sum(1 for r in self.reports.values() if r.get("exit_code") != 0),
                "success_rate": sum(1 for r in self.reports.values() if r.get("exit_code") == 0) / len(self.reports) * 100 if self.reports else 0
            }
        }
        
        # Save report
        report_file = REPORTS_PATH / f"build_report_{self.build_id}.json"
        with open(report_file, 'w') as f:
            import json
            json.dump(report, f, indent=2)
        
        logger.info(f"Build report saved to {report_file}")
        
        # Print summary
        logger.info("=" * 60)
        logger.info("BUILD SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Build ID: {self.build_id}")
        logger.info(f"Build Time: {build_time:.2f} seconds")
        logger.info(f"Total Checks: {report['summary']['total_checks']}")
        logger.info(f"Passed Checks: {report['summary']['passed_checks']}")
        logger.info(f"Failed Checks: {report['summary']['failed_checks']}")
        logger.info(f"Success Rate: {report['summary']['success_rate']:.1f}%")
        logger.info("=" * 60)
    
    def run_complete_build(self) -> bool:
        """Run complete build pipeline"""
        logger.info("Starting comprehensive build pipeline...")
        
        build_steps = [
            ("Environment Setup", self.setup_environment),
            ("Code Quality Checks", self.code_quality_checks),
            ("Security Scanning", self.security_scanning),
            ("Test Suite", self.run_tests),
            ("Documentation Build", self.build_documentation),
            ("Docker Images", self.build_docker_images),
            ("Distribution Packages", self.create_distribution),
            ("Performance Benchmarks", self.performance_benchmarks),
        ]
        
        success = True
        for step_name, step_func in build_steps:
            logger.info(f"Running {step_name}...")
            try:
                step_success = step_func()
                if not step_success:
                    logger.error(f"{step_name} failed")
                    success = False
                else:
                    logger.info(f"{step_name} completed successfully")
            except Exception as e:
                logger.error(f"{step_name} failed with exception: {e}")
                success = False
        
        # Generate build report
        self.generate_build_report()
        
        if success:
            logger.info("Complete build pipeline succeeded!")
        else:
            logger.error("Complete build pipeline failed!")
        
        return success


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Comprehensive build script for LlamaAgent"
    )
    parser.add_argument(
        "--skip-quality", action="store_true",
        help="Skip code quality checks"
    )
    parser.add_argument(
        "--skip-security", action="store_true",
        help="Skip security scanning"
    )
    parser.add_argument(
        "--skip-tests", action="store_true",
        help="Skip test suite"
    )
    parser.add_argument(
        "--skip-docs", action="store_true",
        help="Skip documentation build"
    )
    parser.add_argument(
        "--skip-docker", action="store_true",
        help="Skip Docker image building"
    )
    parser.add_argument(
        "--skip-dist", action="store_true",
        help="Skip distribution creation"
    )
    parser.add_argument(
        "--skip-benchmarks", action="store_true",
        help="Skip performance benchmarks"
    )
    parser.add_argument(
        "--coverage-threshold", type=float, default=85.0,
        help="Code coverage threshold"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Update config based on arguments
    config = BUILD_CONFIG.copy()
    config["coverage_threshold"] = args.coverage_threshold
    
    # Create build system
    build_system = ComprehensiveBuildSystem(config)
    
    # Run individual steps if requested
    if args.skip_quality and args.skip_security and args.skip_tests and args.skip_docs and args.skip_docker and args.skip_dist and args.skip_benchmarks:
        logger.info("All steps skipped, running environment setup only")
        success = build_system.setup_environment()
    else:
        # Run complete build
        success = build_system.run_complete_build()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 
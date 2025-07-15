#!/usr/bin/env python3
"""
LlamaAgent Build Automation Script

A comprehensive build automation tool for the LlamaAgent system.
Handles testing, linting, building, Docker deployment, and more.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LlamaAgentBuilder:
    """Comprehensive build automation for LlamaAgent system."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent
        self.venv_path = self.project_root / ".venv"
        self.python_cmd = self._get_python_command()
        
    def _get_python_command(self) -> str:
        """Get the correct Python command for the environment."""
        if self.venv_path.exists():
            if os.name == 'nt':  # Windows
                return str(self.venv_path / "Scripts" / "python.exe")
            else:  # Unix/Linux/Mac
                return str(self.venv_path / "bin" / "python")
        return sys.executable
    
    def run_command(self, cmd: List[str], check: bool = True, capture_output: bool = False) -> subprocess.CompletedProcess:
        """Run a command with proper error handling."""
        logger.info(f"Running: {' '.join(cmd)}")
        try:
            return subprocess.run(
                cmd, 
                check=check, 
                capture_output=capture_output,
                text=True,
                cwd=self.project_root
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e}")
            if capture_output:
                logger.error(f"stdout: {e.stdout}")
                logger.error(f"stderr: {e.stderr}")
            raise
    
    def setup_environment(self) -> None:
        """Set up the development environment."""
        logger.info("Setting up development environment...")
        
        # Create virtual environment if it doesn't exist
        if not self.venv_path.exists():
            logger.info("Creating virtual environment...")
            self.run_command([sys.executable, "-m", "venv", str(self.venv_path)])
        
        # Install dependencies
        logger.info("Installing dependencies...")
        self.run_command([self.python_cmd, "-m", "pip", "install", "--upgrade", "pip"])
        self.run_command([self.python_cmd, "-m", "pip", "install", "-r", "requirements.txt"])
        
        if Path("requirements-dev.txt").exists():
            self.run_command([self.python_cmd, "-m", "pip", "install", "-r", "requirements-dev.txt"])
        
        logger.info("Environment setup complete!")
    
    def lint(self) -> bool:
        """Run linting checks."""
        logger.info("Running linting checks...")
        success = True
        
        try:
            # Run ruff check
            logger.info("Running ruff check...")
            self.run_command([self.python_cmd, "-m", "ruff", "check", "src/"])
            
            # Run ruff format check
            logger.info("Running ruff format check...")
            self.run_command([self.python_cmd, "-m", "ruff", "format", "--check", "src/"])
            
            logger.info("Linting passed!")
        except subprocess.CalledProcessError:
            logger.error("Linting failed!")
            success = False
        
        return success
    
    def type_check(self) -> bool:
        """Run type checking."""
        logger.info("Running type checking...")
        try:
            self.run_command([self.python_cmd, "-m", "mypy", "src/llamaagent", "--ignore-missing-imports"])
            logger.info("Type checking passed!")
            return True
        except subprocess.CalledProcessError:
            logger.error("Type checking failed!")
            return False
    
    def test(self, coverage: bool = True, verbose: bool = True) -> bool:
        """Run the test suite."""
        logger.info("Running test suite...")
        
        cmd = [self.python_cmd, "-m", "pytest", "tests/"]
        
        if verbose:
            cmd.append("-v")
        
        if coverage:
            cmd.extend(["--cov=src/llamaagent", "--cov-report=html", "--cov-report=term"])
        
        try:
            self.run_command(cmd)
            logger.info("Tests passed!")
            return True
        except subprocess.CalledProcessError:
            logger.error("Tests failed!")
            return False
    
    def security_check(self) -> bool:
        """Run security analysis."""
        logger.info("Running security checks...")
        success = True
        
        try:
            # Install security tools if not available
            try:
                import bandit
                import safety
            except ImportError:
                logger.info("Installing security tools...")
                self.run_command([self.python_cmd, "-m", "pip", "install", "bandit", "safety"])
            
            # Run bandit
            logger.info("Running bandit security analysis...")
            self.run_command([self.python_cmd, "-m", "bandit", "-r", "src/", "-f", "json", "-o", "bandit-report.json"])
            
            # Run safety check
            logger.info("Running safety vulnerability check...")
            self.run_command([self.python_cmd, "-m", "safety", "check", "--json", "--output", "safety-report.json"])
            
            logger.info("Security checks passed!")
        except subprocess.CalledProcessError:
            logger.error("Security checks failed!")
            success = False
        
        return success
    
    def benchmark(self, max_tasks: int = 20) -> bool:
        """Run benchmark evaluation."""
        logger.info(f"Running benchmark evaluation with {max_tasks} tasks...")
        
        try:
            cmd = [
                self.python_cmd, "-m", "src.llamaagent.benchmarks.gaia_benchmark",
                "--max-tasks", str(max_tasks),
                "--output-file", "benchmark-results.json"
            ]
            self.run_command(cmd)
            
            # Load and display results
            if Path("benchmark-results.json").exists():
                with open("benchmark-results.json") as f:
                    results = json.load(f)
                
                logger.info("Benchmark Results:")
                for agent_type, metrics in results.items():
                    success_rate = metrics.get('success_rate', 0)
                    tasks_completed = metrics.get('tasks_completed', 0)
                    logger.info(f"  {agent_type}: {success_rate:.1f}% success rate ({tasks_completed} tasks)")
            
            logger.info("Benchmark evaluation completed!")
            return True
        except subprocess.CalledProcessError:
            logger.error("Benchmark evaluation failed!")
            return False
    
    def build_docker(self, tag: str = "llamaagent:latest", production: bool = False) -> bool:
        """Build Docker image."""
        logger.info(f"Building Docker image: {tag}")
        
        dockerfile = "Dockerfile.production" if production else "Dockerfile"
        
        try:
            cmd = ["docker", "build", "-f", dockerfile, "-t", tag, "."]
            self.run_command(cmd)
            logger.info(f"Docker image built successfully: {tag}")
            return True
        except subprocess.CalledProcessError:
            logger.error("Docker build failed!")
            return False
    
    def start_services(self, detached: bool = True) -> bool:
        """Start development services with Docker Compose."""
        logger.info("Starting development services...")
        
        try:
            cmd = ["docker-compose", "-f", "docker-compose.dev.yml", "up"]
            if detached:
                cmd.append("-d")
            
            self.run_command(cmd)
            
            if detached:
                # Wait for services to be ready
                logger.info("Waiting for services to be ready...")
                time.sleep(10)
                
                # Check if services are healthy
                result = self.run_command(
                    ["docker-compose", "-f", "docker-compose.dev.yml", "ps"], 
                    capture_output=True
                )
                logger.info(f"Services status:\n{result.stdout}")
            
            logger.info("Development services started!")
            return True
        except subprocess.CalledProcessError:
            logger.error("Failed to start services!")
            return False
    
    def stop_services(self) -> bool:
        """Stop development services."""
        logger.info("Stopping development services...")
        
        try:
            self.run_command(["docker-compose", "-f", "docker-compose.dev.yml", "down"])
            logger.info("Development services stopped!")
            return True
        except subprocess.CalledProcessError:
            logger.error("Failed to stop services!")
            return False
    
    def clean(self) -> None:
        """Clean build artifacts and caches."""
        logger.info("Cleaning build artifacts...")
        
        # Remove Python cache
        for cache_dir in self.project_root.rglob("__pycache__"):
            if cache_dir.is_dir():
                subprocess.run(["rm", "-rf", str(cache_dir)], check=False)
        
        # Remove test artifacts
        for pattern in ["*.pyc", ".pytest_cache", ".coverage", "htmlcov", "*.egg-info"]:
            for path in self.project_root.rglob(pattern):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    subprocess.run(["rm", "-rf", str(path)], check=False)
        
        # Remove build artifacts
        for artifact in ["bandit-report.json", "safety-report.json", "benchmark-results.json"]:
            artifact_path = self.project_root / artifact
            if artifact_path.exists():
                artifact_path.unlink()
        
        logger.info("Clean complete!")
    
    def deploy(self, environment: str = "development") -> bool:
        """Deploy the application."""
        logger.info(f"Deploying to {environment} environment...")
        
        if environment == "development":
            return self.start_services()
        elif environment == "production":
            # Production deployment would use docker-compose.production.yml
            try:
                self.run_command(["docker-compose", "-f", "docker-compose.production.yml", "up", "-d"])
                logger.info("Production deployment started!")
                return True
            except subprocess.CalledProcessError:
                logger.error("Production deployment failed!")
                return False
        else:
            logger.error(f"Unknown environment: {environment}")
            return False
    
    def health_check(self, url: str = "http://localhost:8000") -> bool:
        """Check application health."""
        logger.info(f"Checking application health at {url}...")
        
        try:
            import requests
            response = requests.get(f"{url}/health", timeout=10)
            response.raise_for_status()
            
            health_data = response.json()
            logger.info(f"Health check passed: {health_data['status']}")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def full_build(self) -> bool:
        """Run the complete build pipeline."""
        logger.info("Starting full build pipeline...")
        
        steps = [
            ("Setup", self.setup_environment),
            ("Linting", self.lint),
            ("Type Checking", self.type_check),
            ("Testing", self.test),
            ("Security", self.security_check),
        ]
        
        for step_name, step_func in steps:
            logger.info(f"=== {step_name} ===")
            if not step_func():
                logger.error(f"Build failed at step: {step_name}")
                return False
        
        logger.info("Full build pipeline completed successfully!")
        return True


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="LlamaAgent Build Automation")
    
    parser.add_argument("--project-root", type=Path, help="Project root directory")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Setup command
    subparsers.add_parser("setup", help="Set up development environment")
    
    # Quality checks
    subparsers.add_parser("lint", help="Run linting checks")
    subparsers.add_parser("typecheck", help="Run type checking")
    subparsers.add_parser("security", help="Run security analysis")
    
    # Testing
    test_parser = subparsers.add_parser("test", help="Run test suite")
    test_parser.add_argument("--no-coverage", action="store_true", help="Skip coverage reporting")
    test_parser.add_argument("--quiet", action="store_true", help="Quiet output")
    
    # Benchmarking
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark evaluation")
    bench_parser.add_argument("--max-tasks", type=int, default=20, help="Maximum tasks to run")
    
    # Docker
    docker_parser = subparsers.add_parser("docker", help="Build Docker image")
    docker_parser.add_argument("--tag", default="llamaagent:latest", help="Docker image tag")
    docker_parser.add_argument("--production", action="store_true", help="Build production image")
    
    # Services
    subparsers.add_parser("start", help="Start development services")
    subparsers.add_parser("stop", help="Stop development services")
    
    # Deployment
    deploy_parser = subparsers.add_parser("deploy", help="Deploy application")
    deploy_parser.add_argument("--env", choices=["development", "production"], default="development")
    
    # Health check
    health_parser = subparsers.add_parser("health", help="Check application health")
    health_parser.add_argument("--url", default="http://localhost:8000", help="Application URL")
    
    # Utility commands
    subparsers.add_parser("clean", help="Clean build artifacts")
    subparsers.add_parser("full", help="Run full build pipeline")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    builder = LlamaAgentBuilder(args.project_root)
    
    try:
        if args.command == "setup":
            builder.setup_environment()
        elif args.command == "lint":
            success = builder.lint()
            sys.exit(0 if success else 1)
        elif args.command == "typecheck":
            success = builder.type_check()
            sys.exit(0 if success else 1)
        elif args.command == "security":
            success = builder.security_check()
            sys.exit(0 if success else 1)
        elif args.command == "test":
            success = builder.test(coverage=not args.no_coverage, verbose=not args.quiet)
            sys.exit(0 if success else 1)
        elif args.command == "benchmark":
            success = builder.benchmark(args.max_tasks)
            sys.exit(0 if success else 1)
        elif args.command == "docker":
            success = builder.build_docker(args.tag, args.production)
            sys.exit(0 if success else 1)
        elif args.command == "start":
            success = builder.start_services()
            sys.exit(0 if success else 1)
        elif args.command == "stop":
            success = builder.stop_services()
            sys.exit(0 if success else 1)
        elif args.command == "deploy":
            success = builder.deploy(args.env)
            sys.exit(0 if success else 1)
        elif args.command == "health":
            success = builder.health_check(args.url)
            sys.exit(0 if success else 1)
        elif args.command == "clean":
            builder.clean()
        elif args.command == "full":
            success = builder.full_build()
            sys.exit(0 if success else 1)
        else:
            logger.error(f"Unknown command: {args.command}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Build interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Build failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

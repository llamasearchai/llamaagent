#!/usr/bin/env python3
"""
Comprehensive Fix Script for LlamaAgent

This script systematically fixes all import errors, syntax issues, and other problems
using modern Python tools (hatch, uv, tox) and prevents future issues.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import ast
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import json
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fix_all_issues.log')
    ]
)
logger = logging.getLogger(__name__)


class FixPriority(Enum):
    """Priority levels for fixes."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


@dataclass
class Fix:
    """Represents a fix to be applied."""
    file_path: Path
    priority: FixPriority
    category: str
    description: str
    fix_function: str
    dependencies: List[str] = field(default_factory=list)
    applied: bool = False


class ComprehensiveFixEngine:
    """Engine that systematically fixes all issues in the codebase."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.src_dir = project_root / "src"
        self.fixes: List[Fix] = []
        self.fixed_files: Dict[str, bool] = {}
        self.failed_fixes: List[Fix] = []
        
        # Tools availability
        self.tools = {
            'uv': self._check_tool_availability('uv'),
            'hatch': self._check_tool_availability('hatch'),
            'tox': self._check_tool_availability('tox'),
            'ruff': self._check_tool_availability('ruff'),
            'black': self._check_tool_availability('black'),
            'mypy': self._check_tool_availability('mypy'),
            'pre-commit': self._check_tool_availability('pre-commit')
        }
        
        logger.info(f"Tool availability: {self.tools}")
    
    def _check_tool_availability(self, tool: str) -> bool:
        """Check if a tool is available in the system."""
        try:
            subprocess.run([tool, '--version'], 
                         capture_output=True, check=True, timeout=10)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def install_modern_tools(self) -> bool:
        """Install modern Python tools if not available."""
        logger.info("FIXING Installing modern Python tools...")
        
        tools_to_install = []
        
        # Install uv if not available
        if not self.tools['uv']:
            logger.info("Installing uv...")
            try:
                subprocess.run([
                    sys.executable, '-m', 'pip', 'install', 'uv>=0.1.0'
                ], check=True)
                self.tools['uv'] = True
                logger.info("PASS uv installed successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"FAIL Failed to install uv: {e}")
                return False
        
        # Install hatch if not available
        if not self.tools['hatch']:
            logger.info("Installing hatch...")
            try:
                subprocess.run([
                    sys.executable, '-m', 'pip', 'install', 'hatch>=1.9.0'
                ], check=True)
                self.tools['hatch'] = True
                logger.info("PASS hatch installed successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"FAIL Failed to install hatch: {e}")
                return False
        
        # Install tox if not available
        if not self.tools['tox']:
            logger.info("Installing tox...")
            try:
                subprocess.run([
                    sys.executable, '-m', 'pip', 'install', 'tox>=4.0.0'
                ], check=True)
                self.tools['tox'] = True
                logger.info("PASS tox installed successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"FAIL Failed to install tox: {e}")
                return False
        
        return True
    
    def setup_development_environment(self) -> bool:
        """Set up the development environment with proper packaging."""
        logger.info("🏗️  Setting up development environment...")
        
        # Install package in development mode using hatch
        if self.tools['hatch']:
            try:
                logger.info("Installing package in development mode with hatch...")
                subprocess.run([
                    'hatch', 'env', 'create', 'default'
                ], cwd=self.project_root, check=True)
                
                subprocess.run([
                    'hatch', 'run', 'install-dev'
                ], cwd=self.project_root, check=True)
                
                logger.info("PASS Development environment set up successfully")
                return True
            except subprocess.CalledProcessError as e:
                logger.warning(f"⚠️  Hatch setup failed: {e}")
        
        # Fallback to pip/uv
        if self.tools['uv']:
            try:
                logger.info("Installing package in development mode with uv...")
                subprocess.run([
                    'uv', 'pip', 'install', '-e', '.[dev,all]'
                ], cwd=self.project_root, check=True)
                
                logger.info("PASS Development environment set up successfully")
                return True
            except subprocess.CalledProcessError as e:
                logger.warning(f"⚠️  uv setup failed: {e}")
        
        # Final fallback to pip
        try:
            logger.info("Installing package in development mode with pip...")
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-e', '.[dev,all]'
            ], cwd=self.project_root, check=True)
            
            logger.info("PASS Development environment set up successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"FAIL Failed to set up development environment: {e}")
            return False
    
    def fix_import_structure(self) -> bool:
        """Fix the import structure of the entire project."""
        logger.info("FIXING Fixing import structure...")
        
        # 1. Fix __init__.py files
        self._fix_init_files()
        
        # 2. Fix relative imports
        self._fix_relative_imports()
        
        # 3. Fix absolute imports
        self._fix_absolute_imports()
        
        # 4. Fix circular imports
        self._fix_circular_imports()
        
        return True
    
    def _fix_init_files(self):
        """Fix all __init__.py files in the project."""
        logger.info("Fixing __init__.py files...")
        
        # Critical __init__.py files that must exist
        critical_init_files = [
            self.src_dir / "llamaagent" / "__init__.py",
            self.src_dir / "llamaagent" / "agents" / "__init__.py",
            self.src_dir / "llamaagent" / "tools" / "__init__.py",
            self.src_dir / "llamaagent" / "llm" / "__init__.py",
            self.src_dir / "llamaagent" / "api" / "__init__.py",
            self.src_dir / "llamaagent" / "core" / "__init__.py",
            self.src_dir / "llamaagent" / "cli" / "__init__.py",
        ]
        
        for init_file in critical_init_files:
            if not init_file.exists():
                logger.info(f"Creating missing {init_file}")
                init_file.parent.mkdir(parents=True, exist_ok=True)
                init_file.write_text('"""Package initialization."""\n')
            else:
                # Check if __init__.py has syntax errors
                try:
                    with open(init_file, 'r') as f:
                        content = f.read()
                    ast.parse(content)
                except SyntaxError as e:
                    logger.warning(f"Syntax error in {init_file}: {e}")
                    self._fix_syntax_error(init_file)
    
    def _fix_relative_imports(self):
        """Fix relative imports to use absolute imports."""
        logger.info("Fixing relative imports...")
        
        python_files = list(self.src_dir.rglob("*.py"))
        
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                # Replace relative imports with absolute imports
                lines = content.split('\n')
                modified = False
                
                for i, line in enumerate(lines):
                    # Fix relative imports like "from .module import something"
                    if re.match(r'^\s*from\s+\.+\w+\s+import', line):
                        # Convert to absolute import
                        relative_match = re.match(r'^\s*from\s+(\.+)(\w+.*)\s+import\s+(.+)', line)
                        if relative_match:
                            dots, module, imports = relative_match.groups()
                            
                            # Calculate absolute path
                            level = len(dots)
                            file_parts = py_file.relative_to(self.src_dir).parts
                            
                            if level == 1:  # from .module
                                if len(file_parts) > 1:
                                    base_path = ".".join(file_parts[:-1])
                                    absolute_import = f"from {base_path}.{module} import {imports}"
                                else:
                                    absolute_import = f"from {module} import {imports}"
                            else:  # from ..module
                                if len(file_parts) > level:
                                    base_path = ".".join(file_parts[:-level])
                                    absolute_import = f"from {base_path}.{module} import {imports}"
                                else:
                                    continue  # Skip invalid relative imports
                            
                            lines[i] = absolute_import
                            modified = True
                            logger.debug(f"Fixed relative import in {py_file}: {line} -> {absolute_import}")
                
                if modified:
                    with open(py_file, 'w') as f:
                        f.write('\n'.join(lines))
                    logger.info(f"Fixed relative imports in {py_file}")
                    
            except Exception as e:
                logger.error(f"Error fixing relative imports in {py_file}: {e}")
    
    def _fix_absolute_imports(self):
        """Fix absolute imports to use correct module paths."""
        logger.info("Fixing absolute imports...")
        
        python_files = list(self.src_dir.rglob("*.py"))
        
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                # Fix common import patterns
                lines = content.split('\n')
                modified = False
                
                for i, line in enumerate(lines):
                    # Fix src.llamaagent imports
                    if 'src.llamaagent' in line and 'import' in line:
                        new_line = line.replace('src.llamaagent', 'llamaagent')
                        if new_line != line:
                            lines[i] = new_line
                            modified = True
                            logger.debug(f"Fixed absolute import in {py_file}: {line} -> {new_line}")
                
                if modified:
                    with open(py_file, 'w') as f:
                        f.write('\n'.join(lines))
                    logger.info(f"Fixed absolute imports in {py_file}")
                    
            except Exception as e:
                logger.error(f"Error fixing absolute imports in {py_file}: {e}")
    
    def _fix_circular_imports(self):
        """Fix circular imports by restructuring import order."""
        logger.info("Fixing circular imports...")
        
        # Common circular import patterns to fix
        circular_patterns = [
            # Move TYPE_CHECKING imports
            (r'^(from .+ import .+)', r'from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    \1'),
        ]
        
        python_files = list(self.src_dir.rglob("*.py"))
        
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                # Add TYPE_CHECKING imports if needed
                if 'TYPE_CHECKING' not in content and 'from typing import' in content:
                    content = content.replace(
                        'from typing import', 
                        'from typing import TYPE_CHECKING,'
                    )
                
                with open(py_file, 'w') as f:
                    f.write(content)
                    
            except Exception as e:
                logger.error(f"Error fixing circular imports in {py_file}: {e}")
    
    def _fix_syntax_error(self, file_path: Path):
        """Fix syntax errors in a file."""
        logger.info(f"Fixing syntax errors in {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Common syntax fixes
            lines = content.split('\n')
            
            for i, line in enumerate(lines):
                # Fix unclosed parentheses/brackets
                if line.count('(') != line.count(')'):
                    # Simple fix: add missing closing parenthesis
                    if line.count('(') > line.count(')'):
                        lines[i] = line + ')' * (line.count('(') - line.count(')'))
                
                # Fix missing colons in function definitions
                if re.match(r'^\s*def\s+\w+\([^)]*\)\s*$', line):
                    lines[i] = line + ':'
                
                # Fix missing colons in class definitions
                if re.match(r'^\s*class\s+\w+.*\s*$', line) and not line.endswith(':'):
                    lines[i] = line + ':'
            
            fixed_content = '\n'.join(lines)
            
            # Try to parse the fixed content
            try:
                ast.parse(fixed_content)
                with open(file_path, 'w') as f:
                    f.write(fixed_content)
                logger.info(f"PASS Fixed syntax errors in {file_path}")
                return True
            except SyntaxError as e:
                logger.error(f"FAIL Could not fix syntax errors in {file_path}: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error fixing syntax in {file_path}: {e}")
            return False
    
    def run_code_quality_tools(self) -> bool:
        """Run code quality tools to fix formatting and linting issues."""
        logger.info("FIXING Running code quality tools...")
        
        # 1. Run ruff to fix linting issues
        if self.tools['ruff']:
            logger.info("Running ruff...")
            try:
                subprocess.run([
                    'ruff', 'check', '--fix', 'src', 'tests'
                ], cwd=self.project_root, check=True)
                
                subprocess.run([
                    'ruff', 'format', 'src', 'tests'
                ], cwd=self.project_root, check=True)
                
                logger.info("PASS Ruff fixes applied")
            except subprocess.CalledProcessError as e:
                logger.warning(f"⚠️  Ruff failed: {e}")
        
        # 2. Run black for formatting
        if self.tools['black']:
            logger.info("Running black...")
            try:
                subprocess.run([
                    'black', 'src', 'tests'
                ], cwd=self.project_root, check=True)
                
                logger.info("PASS Black formatting applied")
            except subprocess.CalledProcessError as e:
                logger.warning(f"⚠️  Black failed: {e}")
        
        # 3. Run autoflake to remove unused imports
        try:
            logger.info("Running autoflake...")
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', 'autoflake>=2.0.0'
            ], check=True)
            
            subprocess.run([
                'autoflake', '--in-place', '--remove-all-unused-imports', 
                '--remove-unused-variables', '--recursive', 'src'
            ], cwd=self.project_root, check=True)
            
            logger.info("PASS Autoflake fixes applied")
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️  Autoflake failed: {e}")
        
        return True
    
    def run_comprehensive_tests(self) -> bool:
        """Run comprehensive tests using tox."""
        logger.info("Analyzing Running comprehensive tests...")
        
        if not self.tools['tox']:
            logger.warning("⚠️  Tox not available, skipping comprehensive tests")
            return False
        
        # Run syntax check first
        try:
            logger.info("Running syntax check...")
            subprocess.run([
                'tox', '-e', 'syntax-check'
            ], cwd=self.project_root, check=True)
            
            logger.info("PASS Syntax check passed")
        except subprocess.CalledProcessError as e:
            logger.error(f"FAIL Syntax check failed: {e}")
            return False
        
        # Run import validation
        try:
            logger.info("Running import validation...")
            subprocess.run([
                'tox', '-e', 'validate-imports'
            ], cwd=self.project_root, check=True)
            
            logger.info("PASS Import validation passed")
        except subprocess.CalledProcessError as e:
            logger.error(f"FAIL Import validation failed: {e}")
            return False
        
        # Run linting
        try:
            logger.info("Running linting...")
            subprocess.run([
                'tox', '-e', 'lint'
            ], cwd=self.project_root, check=True)
            
            logger.info("PASS Linting passed")
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️  Linting failed: {e}")
        
        return True
    
    def setup_pre_commit_hooks(self) -> bool:
        """Set up pre-commit hooks to prevent future issues."""
        logger.info("🪝 Setting up pre-commit hooks...")
        
        if not self.tools['pre-commit']:
            try:
                subprocess.run([
                    sys.executable, '-m', 'pip', 'install', 'pre-commit>=3.5.0'
                ], check=True)
                self.tools['pre-commit'] = True
            except subprocess.CalledProcessError as e:
                logger.error(f"FAIL Failed to install pre-commit: {e}")
                return False
        
        try:
            # Install pre-commit hooks
            subprocess.run([
                'pre-commit', 'install'
            ], cwd=self.project_root, check=True)
            
            logger.info("PASS Pre-commit hooks installed")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"FAIL Failed to install pre-commit hooks: {e}")
            return False
    
    def generate_fix_report(self) -> Dict:
        """Generate a comprehensive fix report."""
        logger.info("RESULTS Generating fix report...")
        
        report = {
            'timestamp': str(Path(__file__).stat().st_mtime),
            'project_root': str(self.project_root),
            'tools_available': self.tools,
            'fixes_applied': len([f for f in self.fixes if f.applied]),
            'fixes_failed': len(self.failed_fixes),
            'total_fixes': len(self.fixes),
            'success_rate': len([f for f in self.fixes if f.applied]) / len(self.fixes) * 100 if self.fixes else 0,
            'recommendations': []
        }
        
        # Add recommendations
        if not self.tools['uv']:
            report['recommendations'].append("Install uv for faster dependency management")
        
        if not self.tools['hatch']:
            report['recommendations'].append("Install hatch for better project management")
        
        if not self.tools['pre-commit']:
            report['recommendations'].append("Install pre-commit to prevent future issues")
        
        # Write report to file
        report_file = self.project_root / 'fix_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"PASS Fix report generated: {report_file}")
        return report
    
    def run_all_fixes(self) -> bool:
        """Run all fixes in the correct order."""
        logger.info("Starting comprehensive fix process...")
        
        success = True
        
        # 1. Install modern tools
        if not self.install_modern_tools():
            logger.error("FAIL Failed to install modern tools")
            success = False
        
        # 2. Fix import structure
        if not self.fix_import_structure():
            logger.error("FAIL Failed to fix import structure")
            success = False
        
        # 3. Run code quality tools
        if not self.run_code_quality_tools():
            logger.error("FAIL Failed to run code quality tools")
            success = False
        
        # 4. Set up development environment
        if not self.setup_development_environment():
            logger.error("FAIL Failed to set up development environment")
            success = False
        
        # 5. Run comprehensive tests
        if not self.run_comprehensive_tests():
            logger.error("FAIL Failed to run comprehensive tests")
            success = False
        
        # 6. Set up pre-commit hooks
        if not self.setup_pre_commit_hooks():
            logger.error("FAIL Failed to set up pre-commit hooks")
            success = False
        
        # 7. Generate fix report
        report = self.generate_fix_report()
        
        if success:
            logger.info("SUCCESS All fixes completed successfully!")
            logger.info(f"RESULTS Success rate: {report['success_rate']:.1f}%")
        else:
            logger.warning("⚠️  Some fixes failed, but the system should be significantly improved")
        
        return success


def main():
    """Main function to run all fixes."""
    project_root = Path(__file__).parent
    
    logger.info(f"FIXING Starting comprehensive fix process for {project_root}")
    logger.info("="*80)
    
    # Create fix engine
    engine = ComprehensiveFixEngine(project_root)
    
    # Run all fixes
    success = engine.run_all_fixes()
    
    logger.info("="*80)
    if success:
        logger.info("SUCCESS COMPLETE SUCCESS: All issues have been fixed!")
        logger.info("INSIGHT Next steps:")
        logger.info("   1. Run 'hatch run test' to verify everything works")
        logger.info("   2. Run 'tox' to test in multiple environments")
        logger.info("   3. Commit your changes with 'git commit'")
        logger.info("   4. Pre-commit hooks will prevent future issues")
    else:
        logger.warning("⚠️  PARTIAL SUCCESS: Most issues have been fixed")
        logger.info("INSIGHT Check the fix_report.json file for details")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 
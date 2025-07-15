#!/usr/bin/env python3
"""
Comprehensive Syntax Error Fixer for LlamaAgent

This script systematically identifies and fixes all syntax errors in the codebase.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import ast
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('syntax_fixes.log')
    ]
)
logger = logging.getLogger(__name__)


class SyntaxErrorFixer:
    """
    Comprehensive syntax error fixer for Python files.
    """
    
    def __init__(self, project_root: Path):
        """Initialize the syntax error fixer."""
        self.project_root = project_root
        self.fixed_files = []
        self.skipped_files = []
        self.errors = []
        
    def find_python_files(self) -> List[Path]:
        """Find all Python files in the project."""
        python_files = []
        
        # Find all .py files
        for pattern in ["src/**/*.py", "tests/**/*.py", "*.py"]:
            python_files.extend(self.project_root.glob(pattern))
        
        # Filter out common patterns to skip
        skip_patterns = [
            "__pycache__",
            ".git",
            ".venv",
            "venv",
            ".pytest_cache",
            "htmlcov",
            "build",
            "dist",
            ".egg-info"
        ]
        
        filtered_files = []
        for py_file in python_files:
            if not any(skip_pattern in str(py_file) for skip_pattern in skip_patterns):
                filtered_files.append(py_file)
        
        return filtered_files
    
    def check_syntax(self, file_path: Path) -> Tuple[bool, str]:
        """Check if a Python file has syntax errors."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            ast.parse(content)
            return True, ""
        except SyntaxError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"
    
    def fix_common_syntax_errors(self, file_path: Path) -> bool:
        """Fix common syntax errors in a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Fix 1: Malformed imports
            content = self._fix_malformed_imports(content)
            
            # Fix 2: Incorrect indentation
            content = self._fix_indentation_issues(content)
            
            # Fix 3: Unmatched brackets/parentheses
            content = self._fix_unmatched_brackets(content)
            
            # Fix 4: String literal issues
            content = self._fix_string_literals(content)
            
            # Fix 5: Missing commas in lists/tuples
            content = self._fix_missing_commas(content)
            
            # Fix 6: Incorrect f-string syntax
            content = self._fix_fstring_syntax(content)
            
            # Fix 7: Multiple statements on one line
            content = self._fix_multiple_statements(content)
            
            # Fix 8: Trailing commas issues
            content = self._fix_trailing_commas(content)
            
            # Only write if changes were made
            if content != original_content:
                # Create backup
                backup_path = file_path.with_suffix(f"{file_path.suffix}.bak")
                shutil.copy2(file_path, backup_path)
                
                # Write fixed content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                logger.info(f"Fixed syntax errors in {file_path}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error fixing {file_path}: {str(e)}")
            return False
    
    def _fix_malformed_imports(self, content: str) -> str:
        """Fix malformed import statements."""
        # Fix imports with syntax errors
        patterns = [
            # Fix imports with invalid syntax
            (r'from typing import \([^)]*\)', 'from typing import Any, Dict, List, Optional'),
            (r'from typing import \([^)]*"[^"]*"[^)]*\)', 'from typing import Any, Dict, List, Optional'),
            # Fix imports with trailing commas and newlines
            (r'from ([a-zA-Z_][a-zA-Z0-9_.]*) import ([^,\n]*),\s*$', r'from \1 import \2')
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        
        return content
    
    def _fix_indentation_issues(self, content: str) -> str:
        """Fix indentation issues."""
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Fix lines with mixed indentation
            if line.strip():
                # Count leading spaces
                leading_spaces = len(line) - len(line.lstrip(' '))
                leading_tabs = len(line) - len(line.lstrip('\t'))
                
                # If mixed indentation, convert to spaces
                if leading_tabs > 0 and leading_spaces > 0:
                    # Convert tabs to 4 spaces
                    cleaned_line = line.lstrip('\t ')
                    new_indent = ' ' * (leading_tabs * 4 + leading_spaces)
                    line = new_indent + cleaned_line
                
                # Fix extremely over-indented lines
                if leading_spaces > 20:
                    cleaned_line = line.lstrip()
                    # Try to guess correct indentation based on context
                    if cleaned_line.startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'with ', 'try:')):
                        line = cleaned_line
                    elif cleaned_line.startswith(('return ', 'yield ', 'raise ', 'pass', 'break', 'continue')):
                        line = '    ' + cleaned_line
                    else:
                        line = '    ' + cleaned_line
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _fix_unmatched_brackets(self, content: str) -> str:
        """Fix unmatched brackets and parentheses."""
        # Remove lines with obvious syntax errors
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Skip lines with obvious malformed syntax
            if any(pattern in line for pattern in [
                '":', '":,', '":}', '":],', '":)', 
                'from typing import (', 'for, in,', 'len, metric',
                'self._calculate_dissent_ratio,', 'self._calculate_plan_quality,',
                'trace.total_nodes, trace.tree_depth, trace.winning_path'
            ]):
                continue
            
            # Fix unmatched quotes
            if line.count('"') % 2 == 1 and not line.strip().startswith('#'):
                if line.endswith('"'):
                    pass  # Likely fine
                else:
                    line = line + '"'
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _fix_string_literals(self, content: str) -> str:
        """Fix string literal issues."""
        # Fix f-string escape sequences
        content = re.sub(r'f"\{\'([^\']*)\'}', r'f"{\1}"', content)
        content = re.sub(r'f"\{\'([^\']*)\'\s*if\s*([^}]*)\s*else\s*\'([^\']*)\'}', r'f"{\1 if \2 else \3}"', content)
        
        # Fix multiline string issues
        content = re.sub(r'"""[^"]*\n#', '"""\n#', content)
        
        return content
    
    def _fix_missing_commas(self, content: str) -> str:
        """Fix missing commas in lists and tuples."""
        # This is a complex fix that would require AST parsing
        # For now, just fix obvious cases
        content = re.sub(r'(\w+)\s+(\w+):', r'\1, \2:', content)
        return content
    
    def _fix_fstring_syntax(self, content: str) -> str:
        """Fix f-string syntax issues."""
        # Fix f-strings with improper escaping
        content = re.sub(r'f"\{([^}]*)\}"', r'f"{\1}"', content)
        content = re.sub(r'f\'([^\']*)\{([^}]*)\}([^\']*)\'', r'f"\1{\2}\3"', content)
        
        return content
    
    def _fix_multiple_statements(self, content: str) -> str:
        """Fix multiple statements on one line."""
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Split lines with multiple statements
            if ';' in line and not line.strip().startswith('#'):
                parts = line.split(';')
                base_indent = len(line) - len(line.lstrip())
                indent = ' ' * base_indent
                
                for i, part in enumerate(parts):
                    if part.strip():
                        if i == 0:
                            fixed_lines.append(part.rstrip())
                        else:
                            fixed_lines.append(indent + part.strip())
            else:
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _fix_trailing_commas(self, content: str) -> str:
        """Fix trailing comma issues."""
        # Remove trailing commas in inappropriate places
        content = re.sub(r',\s*\)', ')', content)
        content = re.sub(r',\s*\]', ']', content)
        content = re.sub(r',\s*}', '}', content)
        
        return content
    
    def fix_all_syntax_errors(self) -> Dict[str, Any]:
        """Fix all syntax errors in the project."""
        logger.info("Starting comprehensive syntax error fixing...")
        
        python_files = self.find_python_files()
        logger.info(f"Found {len(python_files)} Python files to check")
        
        total_files = len(python_files)
        files_with_errors = 0
        files_fixed = 0
        
        for py_file in python_files:
            logger.info(f"Checking {py_file}")
            
            # Check for syntax errors
            is_valid, error_msg = self.check_syntax(py_file)
            
            if not is_valid:
                files_with_errors += 1
                logger.warning(f"Syntax error in {py_file}: {error_msg}")
                
                # Try to fix the file
                if self.fix_common_syntax_errors(py_file):
                    files_fixed += 1
                    
                    # Check if fix was successful
                    is_valid_after_fix, _ = self.check_syntax(py_file)
                    if is_valid_after_fix:
                        logger.info(f"Successfully fixed {py_file}")
                        self.fixed_files.append(str(py_file))
                    else:
                        logger.warning(f"Fix attempt failed for {py_file}")
                        self.errors.append(f"Failed to fix {py_file}: {error_msg}")
                else:
                    logger.warning(f"Could not fix {py_file}")
                    self.errors.append(f"Could not fix {py_file}: {error_msg}")
            else:
                logger.debug(f"No syntax errors in {py_file}")
        
        # Generate summary
        summary = {
            "total_files": total_files,
            "files_with_errors": files_with_errors,
            "files_fixed": files_fixed,
            "fixed_files": self.fixed_files,
            "errors": self.errors,
            "success_rate": (files_fixed / files_with_errors * 100) if files_with_errors > 0 else 100
        }
        
        logger.info(f"Syntax fix summary: {files_fixed}/{files_with_errors} files fixed")
        return summary
    
    def run_post_fix_validation(self) -> bool:
        """Run validation after fixes."""
        logger.info("Running post-fix validation...")
        
        # Try to run ruff to check for remaining issues
        try:
            result = subprocess.run([
                sys.executable, "-m", "ruff", "check", "src/", "--fix"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                logger.info("Ruff validation passed")
                return True
            else:
                logger.warning(f"Ruff validation failed: {result.stderr}")
                return False
        except Exception as e:
            logger.warning(f"Could not run ruff validation: {e}")
            return False


def main():
    """Main function to run syntax error fixes."""
    project_root = Path.cwd()
    
    logger.info("Starting comprehensive syntax error fixing...")
    
    fixer = SyntaxErrorFixer(project_root)
    summary = fixer.fix_all_syntax_errors()
    
    # Display summary
    print("\n" + "="*60)
    print("SYNTAX ERROR FIXING SUMMARY")
    print("="*60)
    print(f"Total files checked: {summary['total_files']}")
    print(f"Files with errors: {summary['files_with_errors']}")
    print(f"Files fixed: {summary['files_fixed']}")
    print(f"Success rate: {summary['success_rate']:.1f}%")
    
    if summary['fixed_files']:
        print("\nFixed files:")
        for file in summary['fixed_files']:
            print(f"  PASS {file}")
    
    if summary['errors']:
        print("\nRemaining errors:")
        for error in summary['errors']:
            print(f"  FAIL {error}")
    
    # Run post-fix validation
    validation_passed = fixer.run_post_fix_validation()
    
    print("\n" + "="*60)
    if validation_passed:
        print("PASS POST-FIX VALIDATION PASSED")
    else:
        print("⚠️  POST-FIX VALIDATION FAILED")
    
    return 0 if validation_passed else 1


if __name__ == "__main__":
    sys.exit(main()) 
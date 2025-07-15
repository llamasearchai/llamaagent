#!/usr/bin/env python3
"""
Focused Syntax Fixer for LlamaAgent Core Files
This script identifies and fixes syntax errors in the core codebase only.
"""

import ast
import os
import re
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class ErrorInfo:
    file_path: str
    line_number: int
    error_type: str
    error_message: str
    code_snippet: str = ""

class FocusedSyntaxFixer:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.errors: List[ErrorInfo] = []
        self.fixed_files: List[str] = []
        self.manual_files: List[str] = []
        
        # Directories to exclude
        self.exclude_dirs = {
            '.venv', 'venv', '__pycache__', '.git', 'node_modules',
            'build', 'dist', 'htmlcov', '.pytest_cache', '.mypy_cache',
            '.ruff_cache', 'site-packages'
        }
        
        # Focus on core source files
        self.target_dirs = [
            'src/llamaagent',
            'tests',
            'examples'
        ]
        
        # Priority files to fix first
        self.priority_patterns = [
            'src/llamaagent/cli/*.py',
            'src/llamaagent/agents/*.py',
            'src/llamaagent/integration/*.py',
            'src/llamaagent/cache/*.py',
            'src/llamaagent/security/*.py',
            'src/llamaagent/tools/*.py',
            'src/llamaagent/llm/*.py',
            'src/llamaagent/api/*.py'
        ]

    def should_exclude(self, path: Path) -> bool:
        """Check if a path should be excluded."""
        parts = path.parts
        return any(excluded in parts for excluded in self.exclude_dirs)

    def find_target_files(self) -> List[Path]:
        """Find Python files in target directories."""
        python_files = []
        
        # Find files in target directories
        for target_dir in self.target_dirs:
            target_path = self.root_dir / target_dir
            if target_path.exists():
                for file_path in target_path.rglob('*.py'):
                    if not self.should_exclude(file_path):
                        python_files.append(file_path)
        
        # Also include root level Python files
        for file_path in self.root_dir.glob('*.py'):
            if file_path.name not in ['setup.py', 'conftest.py'] and not file_path.name.startswith('test_'):
                python_files.append(file_path)
        
        # Sort by priority
        priority_files = []
        other_files = []
        
        for file_path in python_files:
            rel_path = str(file_path.relative_to(self.root_dir))
            is_priority = any(
                Path(rel_path).match(pattern) for pattern in self.priority_patterns
            )
            if is_priority:
                priority_files.append(file_path)
            else:
                other_files.append(file_path)
        
        return priority_files + other_files

    def analyze_syntax_error(self, file_path: Path) -> Optional[ErrorInfo]:
        """Analyze a file for syntax errors using AST."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            try:
                ast.parse(content)
                return None  # No syntax error
            except SyntaxError as e:
                lines = content.split('\n')
                snippet = ""
                if e.lineno and 0 < e.lineno <= len(lines):
                    start = max(0, e.lineno - 3)
                    end = min(len(lines), e.lineno + 2)
                    snippet = '\n'.join(f"{i+1:4d}: {lines[i]}" for i in range(start, end))
                
                return ErrorInfo(
                    file_path=str(file_path.relative_to(self.root_dir)),
                    line_number=e.lineno or 0,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    code_snippet=snippet
                )
        except Exception as e:
            return ErrorInfo(
                file_path=str(file_path.relative_to(self.root_dir)),
                line_number=0,
                error_type="ReadError",
                error_message=str(e)
            )

    def apply_common_fixes(self, content: str) -> str:
        """Apply common syntax fixes to content."""
        lines = content.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            fixed_line = line
            
            # Fix missing colons
            patterns = [
                (r'^(\s*)(def\s+\w+\s*\([^)]*\))\s*$', r'\1\2:'),
                (r'^(\s*)(class\s+\w+(?:\([^)]*\))?)\s*$', r'\1\2:'),
                (r'^(\s*)(if\s+[^:]+)\s*$', r'\1\2:'),
                (r'^(\s*)(elif\s+[^:]+)\s*$', r'\1\2:'),
                (r'^(\s*)(else)\s*$', r'\1\2:'),
                (r'^(\s*)(for\s+[^:]+)\s*$', r'\1\2:'),
                (r'^(\s*)(while\s+[^:]+)\s*$', r'\1\2:'),
                (r'^(\s*)(try)\s*$', r'\1\2:'),
                (r'^(\s*)(except[^:]*)\s*$', r'\1\2:'),
                (r'^(\s*)(finally)\s*$', r'\1\2:'),
                (r'^(\s*)(with\s+[^:]+)\s*$', r'\1\2:'),
                (r'^(\s*)(async\s+def\s+\w+\s*\([^)]*\))\s*$', r'\1\2:'),
            ]
            
            for pattern, replacement in patterns:
                if re.match(pattern, fixed_line):
                    fixed_line = re.sub(pattern, replacement, fixed_line)
                    break
            
            # Fix unclosed parentheses on single lines
            if fixed_line.count('(') > fixed_line.count(')'):
                # Only add closing parentheses if it's likely a simple case
                if fixed_line.strip().endswith(',') or fixed_line.strip().endswith('('):
                    fixed_line = fixed_line.rstrip() + ')' * (fixed_line.count('(') - fixed_line.count(')'))
            
            fixed_lines.append(fixed_line)
        
        return '\n'.join(fixed_lines)

    def fix_file(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """Attempt to fix a file. Returns (success, error_message)."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Apply fixes
            fixed_content = self.apply_common_fixes(original_content)
            
            # Check if fixes resolved the syntax error
            try:
                ast.parse(fixed_content)
                
                # Write the fixed content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                
                return True, None
            except SyntaxError as e:
                # Revert to original
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                return False, str(e)
                
        except Exception as e:
            return False, str(e)

    def run(self):
        """Main execution method."""
        print("TARGET Focused Syntax Fixer for LlamaAgent")
        print("=" * 50)
        
        # Find target files
        print("\n📁 Finding target files...")
        target_files = self.find_target_files()
        print(f"Found {len(target_files)} Python files in core directories")
        
        # Analyze files
        print("\nScanning Analyzing syntax errors...")
        files_with_errors = []
        
        for file_path in target_files:
            error_info = self.analyze_syntax_error(file_path)
            if error_info:
                files_with_errors.append((file_path, error_info))
                self.errors.append(error_info)
        
        print(f"Found {len(files_with_errors)} files with syntax errors")
        
        # Group errors by type
        error_types = defaultdict(int)
        for error in self.errors:
            error_types[error.error_type] += 1
        
        print("\nRESULTS Error Distribution:")
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {error_type}: {count}")
        
        # Attempt fixes
        print("\nFIXING Attempting automated fixes...")
        for file_path, error_info in files_with_errors:
            success, error_msg = self.fix_file(file_path)
            
            if success:
                print(f"  PASS Fixed: {error_info.file_path}")
                self.fixed_files.append(error_info.file_path)
            else:
                print(f"  FAIL Manual fix needed: {error_info.file_path}")
                if error_msg:
                    print(f"     Error: {error_msg}")
                self.manual_files.append(error_info.file_path)
        
        # Generate detailed report
        self.generate_report()
        
        # Run validation
        self.validate_core_modules()

    def generate_report(self):
        """Generate a detailed report."""
        report = {
            'summary': {
                'total_files_analyzed': len(self.find_target_files()),
                'files_with_errors': len(self.errors),
                'files_fixed': len(self.fixed_files),
                'files_need_manual_fix': len(self.manual_files)
            },
            'fixed_files': sorted(self.fixed_files),
            'manual_intervention_needed': sorted(self.manual_files),
            'error_details': []
        }
        
        # Add error details for manual files
        for error in self.errors:
            if error.file_path in self.manual_files:
                report['error_details'].append({
                    'file': error.file_path,
                    'line': error.line_number,
                    'type': error.error_type,
                    'message': error.error_message,
                    'snippet': error.code_snippet
                })
        
        with open('focused_syntax_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\n📄 Detailed report saved to: focused_syntax_report.json")

    def validate_core_modules(self):
        """Validate that core modules can be imported."""
        print("\nAnalyzing Validating core modules...")
        
        core_modules = [
            'src.llamaagent',
            'src.llamaagent.cli',
            'src.llamaagent.agents',
            'src.llamaagent.tools',
            'src.llamaagent.llm'
        ]
        
        for module in core_modules:
            try:
                result = subprocess.run(
                    [sys.executable, '-c', f'import {module}'],
                    capture_output=True,
                    text=True,
                    cwd=self.root_dir
                )
                if result.returncode == 0:
                    print(f"  PASS {module}")
                else:
                    print(f"  FAIL {module}: {result.stderr.strip()}")
            except Exception as e:
                print(f"  FAIL {module}: {str(e)}")

def main():
    """Main entry point."""
    root_dir = Path(__file__).parent
    fixer = FocusedSyntaxFixer(root_dir)
    fixer.run()

if __name__ == "__main__":
    main()
#\!/usr/bin/env python3
"""Comprehensive syntax error fixer for llamaagent codebase."""

import ast
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional

class SyntaxFixer:
    def __init__(self):
        self.fixes_applied = 0
        
    def fix_file(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """Fix syntax errors in a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try multiple fix strategies
            fixed_content = content
            
            # Strategy 1: Fix unclosed parentheses
            fixed_content = self.fix_unclosed_parentheses(fixed_content)
            
            # Strategy 2: Fix missing colons
            fixed_content = self.fix_missing_colons(fixed_content)
            
            # Strategy 3: Fix indentation issues
            fixed_content = self.fix_indentation(fixed_content)
            
            # Strategy 4: Fix incomplete statements
            fixed_content = self.fix_incomplete_statements(fixed_content)
            
            # Check if fixes worked
            try:
                ast.parse(fixed_content)
                if fixed_content != content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(fixed_content)
                    self.fixes_applied += 1
                    return True, None
                return True, None
            except SyntaxError as e:
                return False, f"Line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, str(e)
    
    def fix_unclosed_parentheses(self, content: str) -> str:
        """Fix unclosed parentheses."""
        lines = content.split('\n')
        fixed_lines = []
        
        paren_stack = []
        
        for i, line in enumerate(lines):
            # Count parentheses
            for j, char in enumerate(line):
                if char == '(':
                    paren_stack.append((i, j))
                elif char == ')' and paren_stack:
                    paren_stack.pop()
            
            # Check for specific patterns
            stripped = line.strip()
            
            # Pattern: function call missing closing paren
            if re.match(r'.*\w+\([^)]*$', stripped) and not stripped.endswith(':'):
                if i + 1 < len(lines) and not lines[i + 1].strip().startswith(')'):
                    line = line.rstrip() + ')'
            
            # Pattern: asyncio.create_task missing closing paren
            if 'asyncio.create_task(' in line and line.count('(') > line.count(')'):
                missing = line.count('(') - line.count(')')
                line = line.rstrip() + ')' * missing
            
            fixed_lines.append(line)
        
        # Fix any remaining unclosed parens at end of file
        if paren_stack and fixed_lines:
            last_line_idx = len(fixed_lines) - 1
            while fixed_lines[last_line_idx].strip() == '' and last_line_idx > 0:
                last_line_idx -= 1
            fixed_lines[last_line_idx] += ')' * len(paren_stack)
        
        return '\n'.join(fixed_lines)
    
    def fix_missing_colons(self, content: str) -> str:
        """Fix missing colons after function/class definitions."""
        lines = content.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Check for function/class definitions missing colons
            if re.match(r'^(class|def|if|elif|else|try|except|finally|with|for|while)\s+.*[^:]$', stripped):
                if not stripped.endswith((':', '\\', ',')):
                    line = line.rstrip() + ':'
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def fix_indentation(self, content: str) -> str:
        """Fix indentation issues."""
        lines = content.split('\n')
        fixed_lines = []
        
        expected_indent = 0
        indent_stack = [0]
        
        for i, line in enumerate(lines):
            if not line.strip():
                fixed_lines.append(line)
                continue
            
            # Get current indentation
            current_indent = len(line) - len(line.lstrip())
            
            # Check if we need to dedent
            if current_indent < indent_stack[-1]:
                while indent_stack and current_indent < indent_stack[-1]:
                    indent_stack.pop()
            
            # Check for expected indent after colon
            if i > 0 and lines[i-1].strip().endswith(':'):
                expected_indent = indent_stack[-1] + 4
                if current_indent == indent_stack[-1]:
                    # Add proper indentation
                    line = ' ' * expected_indent + line.lstrip()
                indent_stack.append(expected_indent)
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def fix_incomplete_statements(self, content: str) -> str:
        """Fix incomplete statements."""
        lines = content.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Fix incomplete return statements
            if stripped == 'return' and i < len(lines) - 1:
                line = line.rstrip() + ' None'
            
            # Fix incomplete pass statements
            if i > 0 and lines[i-1].strip().endswith(':') and not stripped:
                fixed_lines.append(lines[i-1])
                fixed_lines.append(' ' * (len(lines[i-1]) - len(lines[i-1].lstrip()) + 4) + 'pass')
                continue
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)

def main():
    """Main function."""
    src_dir = Path("src/llamaagent")
    fixer = SyntaxFixer()
    
    if not src_dir.exists():
        print(f"Error: {src_dir} does not exist")
        sys.exit(1)
    
    # Get all Python files
    python_files = list(src_dir.rglob("*.py"))
    total_errors = 0
    remaining_errors = []
    
    print(f"Checking {len(python_files)} Python files...")
    
    for file_path in python_files:
        success, error = fixer.fix_file(file_path)
        if not success:
            total_errors += 1
            remaining_errors.append((file_path, error))
    
    print(f"\nFixed {fixer.fixes_applied} files")
    print(f"Remaining errors: {len(remaining_errors)}")
    
    if remaining_errors:
        print("\nFiles with remaining errors:")
        for file_path, error in remaining_errors[:10]:
            print(f"  {file_path}: {error}")

if __name__ == "__main__":
    main()
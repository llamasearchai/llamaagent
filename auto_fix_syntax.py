#!/usr/bin/env python3
"""
Automated syntax fixer for the llamaagent project.
This script attempts to fix common syntax errors automatically.
"""

import os
import re
import ast
import sys
from pathlib import Path
from typing import List, Tuple

class SyntaxFixer:
    def __init__(self, src_dir: str = "src"):
        self.src_dir = Path(src_dir)
        self.fixed_files = []
        self.failed_files = []
        
    def fix_file(self, filepath: Path) -> Tuple[bool, str]:
        """Fix syntax errors in a single file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Skip empty files
            if not original_content.strip():
                return True, "Empty file"
            
            fixed_content = original_content
            
            # Fix common patterns
            fixed_content = self.fix_missing_parentheses(fixed_content)
            fixed_content = self.fix_fstring_errors(fixed_content)
            fixed_content = self.fix_trailing_commas(fixed_content)
            fixed_content = self.fix_indentation_errors(fixed_content)
            
            # Try to parse the fixed content
            try:
                ast.parse(fixed_content)
                
                # If parsing succeeds and content changed, write back
                if fixed_content != original_content:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(fixed_content)
                    return True, "Fixed successfully"
                else:
                    return True, "No fixes needed"
                    
            except SyntaxError as e:
                # Try line-by-line fixing for remaining errors
                fixed_content = self.fix_specific_line_errors(fixed_content, str(e))
                
                try:
                    ast.parse(fixed_content)
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(fixed_content)
                    return True, "Fixed with line-specific fixes"
                except:
                    return False, f"Still has syntax errors: {str(e)}"
                    
        except Exception as e:
            return False, f"Error processing file: {str(e)}"
    
    def fix_missing_parentheses(self, content: str) -> str:
        """Fix missing closing parentheses."""
        lines = content.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            # Count parentheses in the line
            open_parens = line.count('(')
            close_parens = line.count(')')
            
            # Fix missing closing parentheses at end of line
            if open_parens > close_parens:
                # Check if it's likely a function call or list comprehension
                if re.search(r'\([^)]*$', line) and not line.strip().endswith(','):
                    missing = open_parens - close_parens
                    line = line.rstrip() + ')' * missing
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def fix_fstring_errors(self, content: str) -> str:
        """Fix f-string formatting errors."""
        # Fix f-strings with missing closing braces
        content = re.sub(r'(f["\'][^"\']*\{[^}]*)\{([^}]*["\'])', r'\1}\2', content)
        
        # Fix f-strings with unmatched brackets
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            if 'f"' in line or "f'" in line:
                # Count braces within f-strings
                in_fstring = False
                quote_char = None
                brace_count = 0
                
                i = 0
                while i < len(line):
                    if i < len(line) - 1 and line[i] == 'f' and line[i+1] in '"\'':
                        in_fstring = True
                        quote_char = line[i+1]
                        i += 2
                        continue
                    
                    if in_fstring:
                        if line[i] == quote_char and (i == 0 or line[i-1] != '\\'):
                            if brace_count > 0:
                                # Add missing closing braces
                                line = line[:i] + '}' * brace_count + line[i:]
                                i += brace_count
                            in_fstring = False
                            quote_char = None
                            brace_count = 0
                        elif line[i] == '{':
                            brace_count += 1
                        elif line[i] == '}':
                            brace_count = max(0, brace_count - 1)
                    
                    i += 1
                
                # If still in f-string at end of line, close it
                if in_fstring and brace_count > 0:
                    line = line.rstrip() + '}' * brace_count
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def fix_trailing_commas(self, content: str) -> str:
        """Fix trailing comma issues."""
        # Remove extra commas before closing parentheses
        content = re.sub(r',\s*\)', ')', content)
        
        # Fix list() with trailing comma
        content = re.sub(r'list\(([^,)]+),\)', r'list(\1)', content)
        
        return content
    
    def fix_indentation_errors(self, content: str) -> str:
        """Fix common indentation errors."""
        lines = content.split('\n')
        fixed_lines = []
        expected_indent = 0
        
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            
            # Skip empty lines
            if not stripped:
                fixed_lines.append(line)
                continue
            
            # Calculate current indentation
            current_indent = len(line) - len(stripped)
            
            # Adjust expected indentation based on previous line
            if i > 0:
                prev_line = lines[i-1].rstrip()
                if prev_line.endswith(':'):
                    expected_indent += 4
                elif stripped.startswith(('return', 'pass', 'continue', 'break')):
                    # These can dedent
                    pass
                elif stripped.startswith(('except', 'elif', 'else', 'finally')):
                    expected_indent = max(0, expected_indent - 4)
            
            # Fix obvious indentation errors
            if current_indent > expected_indent + 8:
                # Likely an error
                line = ' ' * expected_indent + stripped
            
            fixed_lines.append(line)
            
            # Update expected indent for next line
            if line.rstrip().endswith(':'):
                pass  # Already handled above
            elif stripped.startswith(('return', 'pass', 'continue', 'break')):
                expected_indent = max(0, expected_indent - 4)
        
        return '\n'.join(fixed_lines)
    
    def fix_specific_line_errors(self, content: str, error_msg: str) -> str:
        """Fix specific line errors based on error message."""
        lines = content.split('\n')
        
        # Extract line number from error message
        line_match = re.search(r'line (\d+)', error_msg)
        if line_match:
            line_num = int(line_match.group(1)) - 1
            
            if 0 <= line_num < len(lines):
                line = lines[line_num]
                
                # Fix specific patterns based on common errors
                # Missing colon after if/for/while/def/class
                if re.match(r'\s*(if|for|while|def|class|elif|else|try|except|finally)\s+.*[^:]$', line):
                    lines[line_num] = line + ':'
                
                # Fix incomplete list comprehensions
                if '[' in line and line.count('[') > line.count(']'):
                    lines[line_num] = line + ']' * (line.count('[') - line.count(']'))
                
                # Fix incomplete dict literals
                if '{' in line and line.count('{') > line.count('}'):
                    lines[line_num] = line + '}' * (line.count('{') - line.count('}'))
        
        return '\n'.join(lines)
    
    def process_directory(self):
        """Process all Python files in the directory."""
        python_files = list(self.src_dir.rglob("*.py"))
        
        print(f"Found {len(python_files)} Python files to check...")
        
        for filepath in python_files:
            print(f"Checking {filepath}...", end=" ")
            success, message = self.fix_file(filepath)
            
            if success:
                if "Fixed" in message:
                    self.fixed_files.append((filepath, message))
                    print(f"PASS {message}")
                else:
                    print(f"✓ {message}")
            else:
                self.failed_files.append((filepath, message))
                print(f"FAIL {message}")
        
        # Summary
        print("\n" + "="*60)
        print(f"Total files processed: {len(python_files)}")
        print(f"Files fixed: {len(self.fixed_files)}")
        print(f"Files still with errors: {len(self.failed_files)}")
        
        if self.fixed_files:
            print("\nFixed files:")
            for filepath, message in self.fixed_files[:10]:
                print(f"  - {filepath}")
            if len(self.fixed_files) > 10:
                print(f"  ... and {len(self.fixed_files) - 10} more")
        
        if self.failed_files:
            print("\nFiles still needing manual fixes:")
            for filepath, message in self.failed_files[:10]:
                print(f"  - {filepath}: {message}")
            if len(self.failed_files) > 10:
                print(f"  ... and {len(self.failed_files) - 10} more")

if __name__ == "__main__":
    fixer = SyntaxFixer("src")
    fixer.process_directory()
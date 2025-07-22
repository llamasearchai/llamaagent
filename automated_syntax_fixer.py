#!/usr/bin/env python3
"""
Automated syntax fixer for common patterns in the LlamaAgent codebase.
"""

import re
import os
from pathlib import Path
from typing import List, Tuple, Optional


class SyntaxFixer:
    """Automated syntax error fixer."""
    
    def __init__(self):
        self.fixes_applied = 0
        self.files_fixed = []
        
    def fix_file(self, file_path: Path) -> bool:
        """Fix common syntax errors in a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            original_content = content
            
            # Apply various fixes
            content = self.fix_missing_closing_parens(content)
            content = self.fix_missing_colons(content)
            content = self.fix_indentation_after_except(content)
            content = self.fix_unclosed_brackets(content)
            content = self.fix_incomplete_function_calls(content)
            
            if content != original_content:
                # Write back the fixed content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.files_fixed.append(str(file_path))
                return True
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
        return False
    
    def fix_missing_closing_parens(self, content: str) -> str:
        """Fix missing closing parentheses."""
        lines = content.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            # Count parentheses
            open_parens = line.count('(')
            close_parens = line.count(')')
            
            # Simple heuristic: if we have more open than close at end of line
            if open_parens > close_parens and not line.strip().endswith('\\'):
                # Check if next line exists and starts with whitespace
                if i + 1 < len(lines) and lines[i + 1].strip() == '':
                    # Add missing closing parentheses
                    line = line.rstrip() + ')' * (open_parens - close_parens)
                    self.fixes_applied += 1
                    
            fixed_lines.append(line)
            
        return '\n'.join(fixed_lines)
    
    def fix_missing_colons(self, content: str) -> str:
        """Fix missing colons after function/class definitions."""
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Check for function/class definitions without colons
            if (stripped.startswith(('def ', 'class ', 'if ', 'elif ', 'else', 'while ', 'for ', 'try', 'except', 'finally', 'with ')) 
                and not stripped.endswith(':') 
                and not stripped.endswith('\\')
                and ')' in stripped):
                
                # Add missing colon
                line = line.rstrip() + ':'
                self.fixes_applied += 1
                
            fixed_lines.append(line)
            
        return '\n'.join(fixed_lines)
    
    def fix_indentation_after_except(self, content: str) -> str:
        """Fix incorrect indentation after except blocks."""
        lines = content.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            if i > 0 and lines[i-1].strip().startswith('except') and lines[i-1].strip().endswith(':'):
                # Check if current line has proper indentation
                if line and not line[0].isspace() and line.strip() != '':
                    # Get indentation from except line
                    except_indent = len(lines[i-1]) - len(lines[i-1].lstrip())
                    # Add proper indentation
                    line = ' ' * (except_indent + 4) + line
                    self.fixes_applied += 1
                    
            fixed_lines.append(line)
            
        return '\n'.join(fixed_lines)
    
    def fix_unclosed_brackets(self, content: str) -> str:
        """Fix unclosed brackets in dictionary/list definitions."""
        lines = content.split('\n')
        fixed_lines = []
        
        bracket_stack = []
        
        for i, line in enumerate(lines):
            # Track opening brackets
            for char in line:
                if char in '[{(':
                    bracket_stack.append(char)
                elif char in ']})':
                    if bracket_stack:
                        bracket_stack.pop()
            
            # If we're at end of a logical block and have unclosed brackets
            if (i + 1 < len(lines) and 
                lines[i + 1].strip() == '' and 
                bracket_stack and 
                not line.strip().endswith(',')):
                
                # Add closing brackets
                closing = ''
                for bracket in reversed(bracket_stack):
                    if bracket == '[':
                        closing += ']'
                    elif bracket == '{':
                        closing += '}'
                    elif bracket == '(':
                        closing += ')'
                
                if closing:
                    line = line.rstrip() + closing
                    bracket_stack.clear()
                    self.fixes_applied += 1
                    
            fixed_lines.append(line)
            
        return '\n'.join(fixed_lines)
    
    def fix_incomplete_function_calls(self, content: str) -> str:
        """Fix incomplete function calls (missing closing parenthesis)."""
        # Pattern: function_name(args... but no closing )
        pattern = r'(\w+\([^)]*?)(\n\s*(?:def|class|if|elif|else|while|for|try|except|finally|with)\s)'
        
        def replacer(match):
            self.fixes_applied += 1
            return match.group(1) + ')' + match.group(2)
        
        return re.sub(pattern, replacer, content, flags=re.MULTILINE)


def main():
    """Main function to fix syntax errors."""
    fixer = SyntaxFixer()
    
    # Read the list of files with errors
    if not os.path.exists('syntax_errors.txt'):
        print("No syntax_errors.txt found. Run fix_all_syntax.py first.")
        return 1
    
    with open('syntax_errors.txt', 'r') as f:
        error_files = [line.split('|')[0] for line in f.readlines()]
    
    print(f"Found {len(error_files)} files with syntax errors")
    
    for file_path in error_files:
        file_path = Path(file_path.strip())
        if file_path.exists():
            print(f"Fixing {file_path}...")
            if fixer.fix_file(file_path):
                print(f"   Fixed")
            else:
                print(f"  - No automatic fixes applied")
    
    print(f"\nSummary:")
    print(f"Files processed: {len(error_files)}")
    print(f"Files fixed: {len(fixer.files_fixed)}")
    print(f"Total fixes applied: {fixer.fixes_applied}")
    
    if fixer.files_fixed:
        print(f"\nFixed files:")
        for file in fixer.files_fixed:
            print(f"  - {file}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
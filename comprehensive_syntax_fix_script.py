#!/usr/bin/env python3
"""
Comprehensive Syntax Fix Script

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import re
import ast
import subprocess
from pathlib import Path
from typing import List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveSyntaxFixer:
    """Comprehensive syntax error fixer for the entire codebase."""
    
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.fixes_applied: List[str] = []
        self.errors_encountered: List[str] = []
        self.files_with_errors: List[Path] = []
        
    def find_syntax_errors(self) -> List[Path]:
        """Find all Python files with syntax errors."""
        error_files: List[Path] = []
        
        # Get all Python files
        python_files = list(self.root_path.rglob("src/**/*.py"))
        
        for py_file in python_files:
            try:
                result = subprocess.run(
                    ["python3", "-m", "py_compile", str(py_file)],
                    capture_output=True,
                    text=True,
                    cwd=self.root_path
                )
                if result.returncode != 0:
                    error_files.append(py_file)
                    logger.warning(f"Syntax error in {py_file}")
            except Exception as e:
                logger.error(f"Error checking {py_file}: {e}")
                
        return error_files
    
    def fix_unmatched_parentheses(self, content: str) -> str:
        """Fix unmatched parentheses and brackets."""
        # Fix common patterns
        fixes: List[Tuple[str, str]] = [
            # Fix unmatched closing parentheses
            (r'\)\s*$', ')'),
            # Fix missing closing parentheses in function calls
            (r'raise HTTPException\(.*detail=str\(e\)(?!\))', r'\g<0>)'),
            # Fix unmatched opening parentheses
            (r'\.append\(\)\s*\{', '.append({'),
            # Fix missing opening parentheses
            (r'\.append\(\s*\{', '.append({'),
        ]
        
        for pattern, replacement in fixes:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        
        return content
    
    def fix_missing_colons(self, content: str) -> str:
        """Fix missing colons after function definitions, if statements, etc."""
        # Fix function definitions without colons
        content = re.sub(r'(def\s+\w+\([^)]*\))\s*$', r'\1:', content, flags=re.MULTILINE)
        
        # Fix class definitions without colons
        content = re.sub(r'(class\s+\w+(?:\([^)]*\))?)\s*$', r'\1:', content, flags=re.MULTILINE)
        
        # Fix if statements without colons
        content = re.sub(r'(if\s+[^:]+?)\s*$', r'\1:', content, flags=re.MULTILINE)
        
        # Fix for loops without colons
        content = re.sub(r'(for\s+[^:]+?)\s*$', r'\1:', content, flags=re.MULTILINE)
        
        # Fix while loops without colons
        content = re.sub(r'(while\s+[^:]+?)\s*$', r'\1:', content, flags=re.MULTILINE)
        
        # Fix try statements without colons
        content = re.sub(r'(try)\s*$', r'\1:', content, flags=re.MULTILINE)
        
        # Fix except statements without colons
        content = re.sub(r'(except[^:]*?)\s*$', r'\1:', content, flags=re.MULTILINE)
        
        return content
    
    def fix_string_quotes(self, content: str) -> str:
        """Fix string quote issues."""
        # Fix unterminated strings (basic cases)
        content = re.sub(r'"""([^"]*?)$', r'"""\1"""', content, flags=re.MULTILINE)
        content = re.sub(r"'''([^']*?)$", r"'''\1'''", content, flags=re.MULTILINE)
        
        return content
    
    def fix_import_statements(self, content: str) -> str:
        """Fix malformed import statements."""
        # Fix imports with missing quotes
        content = re.sub(r'from typing import\s+""', 'from typing import Dict, List, Optional, Any', content)
        
        # Fix imports with extra commas
        content = re.sub(r'import\s+,', 'import ', content)
        
        return content
    
    def fix_common_syntax_patterns(self, content: str) -> str:
        """Fix common syntax patterns."""
        # Fix missing opening parentheses in function calls
        content = re.sub(r'\.append\(\)\s*\{', '.append({', content)
        
        # Fix mismatched brackets/parentheses
        content = re.sub(r'description="([^"]*)")', r'description="\1"', content)
        
        # Fix extra parentheses
        content = re.sub(r'datetime\.now\(timezone\.utc\)\)', 'datetime.now(timezone.utc)', content)
        
        # Fix function calls with missing parentheses
        content = re.sub(r'return\s+StructuredLogger\(name\)\)', 'return StructuredLogger(name)', content)
        
        # Fix malformed dictionary/list syntax
        content = re.sub(r'placeholders = \["todo", ""\)\.lower\(\)', 'placeholders = ["todo"]', content)
        
        # Fix closing parenthesis that doesn't match opening bracket
        content = re.sub(r'context: Optional\[Dict\[str, description="([^"]*)")', r'context: Optional[Dict[str, str]] = None  # \1', content)
        
        return content
    
    def fix_indentation_issues(self, content: str) -> str:
        """Fix basic indentation issues."""
        lines: List[str] = content.split('\n')
        fixed_lines: List[str] = []
        
        for i, line in enumerate(lines):
            # Skip empty lines
            if not line.strip():
                fixed_lines.append(line)
                continue
            
            # Fix unexpected indentation (basic cases)
            if line.startswith('    ') and i > 0:
                prev_line = lines[i-1].strip()
                if prev_line and not prev_line.endswith(':') and not prev_line.endswith(',') and not prev_line.endswith('('):
                    # Check if this might be a continued line
                    if not any(prev_line.endswith(c) for c in ['=', '+', '-', '*', '/', '(', '[', '{']):
                        # Remove excessive indentation
                        fixed_lines.append(line.lstrip())
                    else:
                        fixed_lines.append(line)
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def fix_file(self, file_path: Path) -> bool:
        """Fix syntax errors in a single file."""
        logger.info(f"Fixing {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Apply various fixes
            content = self.fix_unmatched_parentheses(content)
            content = self.fix_missing_colons(content)
            content = self.fix_string_quotes(content)
            content = self.fix_import_statements(content)
            content = self.fix_common_syntax_patterns(content)
            content = self.fix_indentation_issues(content)
            
            # Check if content changed
            if content != original_content:
                # Validate syntax before writing
                try:
                    ast.parse(content)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    self.fixes_applied.append(str(file_path))
                    logger.info(f"Fixed: {file_path}")
                    return True
                except SyntaxError as e:
                    logger.warning(f"Still has syntax errors after fix: {file_path} - {e}")
                    self.errors_encountered.append(f"{file_path}: {e}")
                    return False
            else:
                logger.info(f"No changes needed for: {file_path}")
                return True
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            self.errors_encountered.append(f"{file_path}: {e}")
            return False
    
    def fix_all_files(self) -> None:
        """Fix all files with syntax errors."""
        logger.info("Starting comprehensive syntax error fixing...")
        
        # Find files with syntax errors
        error_files = self.find_syntax_errors()
        logger.info(f"Found {len(error_files)} files with syntax errors")
        
        # Fix each file
        for file_path in error_files:
            self.fix_file(file_path)
        
        # Report results
        logger.info(f"\nSyntax Fixing Complete:")
        logger.info(f"  Files fixed: {len(self.fixes_applied)}")
        logger.info(f"  Errors encountered: {len(self.errors_encountered)}")
        
        if self.fixes_applied:
            logger.info("\nFixed files:")
            for file_path in self.fixes_applied:
                logger.info(f"  PASS {file_path}")
        
        if self.errors_encountered:
            logger.info("\nRemaining errors:")
            for error in self.errors_encountered:
                logger.info(f"  FAIL {error}")

def main():
    """Main execution function."""
    fixer = ComprehensiveSyntaxFixer()
    fixer.fix_all_files()

if __name__ == "__main__":
    main() 
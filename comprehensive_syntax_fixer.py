#\!/usr/bin/env python3
"""
Comprehensive Python Syntax Error Fixer
Fixes common syntax errors in Python files with backup and reporting capabilities.
"""

import ast
import os
import re
import shutil
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import difflib
import traceback

class SyntaxFixer:
    """Comprehensive syntax error fixer with pattern-based fixes."""
    
    def __init__(self, backup_dir: str = "syntax_backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        self.report = {
            "start_time": datetime.now().isoformat(),
            "files_processed": 0,
            "files_fixed": 0,
            "errors_fixed": 0,
            "failed_files": [],
            "fixes_applied": []
        }
        
        # Define fix patterns
        self.fix_patterns = [
            # Missing closing parenthesis/brackets
            (r'(\([^)]*$)', self._fix_missing_closing_paren),
            (r'(\[[^\]]*$)', self._fix_missing_closing_bracket),
            (r'(\{[^}]*$)', self._fix_missing_closing_brace),
            
            # F-string errors
            (r'f"([^"]*)\{([^}]*)\}"', self._fix_fstring_quotes),
            (r"f'([^']*)\{([^}]*)\}'", self._fix_fstring_quotes_single),
            
            # Missing colons
            (r'^(\s*)(if|elif|else|for|while|def|class|try|except|finally|with)\s+[^:]+$', self._fix_missing_colon),
            
            # Indentation errors
            (r'^( *)(\S)', self._fix_indentation),
            
            # Unmatched quotes
            (r'^([^"\']*)"([^"]*)"?([^"\']*)$', self._fix_quotes),
            (r"^([^\"']*)'([^']*)'?([^\"']*)$", self._fix_quotes_single),
            
            # Extra/missing commas
            (r',\s*\)', self._fix_trailing_comma),
            (r'}\s*{', self._fix_missing_comma_between_dicts),
            
            # Invalid syntax in comparisons
            (r'if\s+(\w+)\s*=\s*([^=])', self._fix_assignment_in_if),
            
            # Type hints issues
            (r':\s*0"\s*,\s*100"\s*,', self._fix_type_hint_corruption),
            
            # Empty blocks
            (r'^(\s*)(if|elif|else|for|while|def|class|try|except|finally).*:\s*$', self._fix_empty_block),
        ]

    def _backup_file(self, filepath: Path) -> Path:
        """Create a backup of the file before modifying."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"{filepath.name}.{timestamp}.bak"
        shutil.copy2(filepath, backup_path)
        return backup_path

    def _fix_missing_closing_paren(self, match, line: str) -> str:
        """Fix missing closing parenthesis."""
        open_count = line.count('(')
        close_count = line.count(')')
        if open_count > close_count:
            return line + ')' * (open_count - close_count)
        return line

    def _fix_missing_closing_bracket(self, match, line: str) -> str:
        """Fix missing closing bracket."""
        open_count = line.count('[')
        close_count = line.count(']')
        if open_count > close_count:
            return line + ']' * (open_count - close_count)
        return line

    def _fix_missing_closing_brace(self, match, line: str) -> str:
        """Fix missing closing brace."""
        open_count = line.count('{')
        close_count = line.count('}')
        if open_count > close_count:
            return line + '}' * (open_count - close_count)
        return line

    def _fix_fstring_quotes(self, match, line: str) -> str:
        """Fix f-string with mismatched quotes."""
        # Count braces inside the f-string
        content = match.group(0)
        if content.count('{') != content.count('}'):
            # Try to fix by adding missing braces
            fixed = re.sub(r'f"([^"]*)\{([^}]*)$', r'f"\1{\2}"', line)
            return fixed
        return line

    def _fix_fstring_quotes_single(self, match, line: str) -> str:
        """Fix f-string with single quotes."""
        content = match.group(0)
        if content.count('{') != content.count('}'):
            fixed = re.sub(r"f'([^']*)\{([^}]*)$", r"f'\1{\2}'", line)
            return fixed
        return line

    def _fix_missing_colon(self, match, line: str) -> str:
        """Add missing colon at end of control structure."""
        if not line.rstrip().endswith(':'):
            return line.rstrip() + ':'
        return line

    def _fix_indentation(self, match, line: str) -> str:
        """Fix indentation to use 4 spaces."""
        indent = match.group(1)
        # Convert tabs to spaces
        indent = indent.replace('\t', '    ')
        # Ensure multiple of 4 spaces
        space_count = len(indent)
        if space_count % 4 != 0:
            new_count = round(space_count / 4) * 4
            indent = ' ' * new_count
        return indent + line.lstrip()

    def _fix_quotes(self, match, line: str) -> str:
        """Fix unmatched double quotes."""
        if line.count('"') % 2 != 0:
            # Add missing quote at the end if needed
            if not line.rstrip().endswith('"'):
                return line.rstrip() + '"'
        return line

    def _fix_quotes_single(self, match, line: str) -> str:
        """Fix unmatched single quotes."""
        if line.count("'") % 2 != 0:
            if not line.rstrip().endswith("'"):
                return line.rstrip() + "'"
        return line

    def _fix_trailing_comma(self, match, line: str) -> str:
        """Remove trailing comma before closing parenthesis."""
        return re.sub(r',\s*\)', ')', line)

    def _fix_missing_comma_between_dicts(self, match, line: str) -> str:
        """Add missing comma between dict literals."""
        return re.sub(r'}\s*{', '}, {', line)

    def _fix_assignment_in_if(self, match, line: str) -> str:
        """Fix assignment operator in if statement."""
        return re.sub(r'if\s+(\w+)\s*=\s*([^=])', r'if \1 == \2', line)

    def _fix_type_hint_corruption(self, match, line: str) -> str:
        """Fix corrupted type hints."""
        # Fix pattern like: from typing import 0", 100", "", "<, "==
        if 'from typing import' in line and ('0"' in line or '100"' in line):
            return "from typing import Dict, List, Any, Optional, Union, Tuple"
        return line

    def _fix_empty_block(self, match, line: str) -> str:
        """Add pass statement to empty blocks."""
        return line

    def _apply_pattern_fixes(self, content: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Apply all pattern-based fixes to the content."""
        lines = content.split('\n')
        fixed_lines = []
        fixes = []
        
        for i, line in enumerate(lines):
            original_line = line
            fixed_line = line
            
            for pattern, fixer in self.fix_patterns:
                match = re.search(pattern, fixed_line)
                if match:
                    try:
                        new_line = fixer(match, fixed_line)
                        if new_line \!= fixed_line:
                            fixed_line = new_line
                            fixes.append({
                                "line": i + 1,
                                "original": original_line,
                                "fixed": fixed_line,
                                "pattern": pattern
                            })
                    except Exception as e:
                        # Skip this fix if it fails
                        pass
            
            fixed_lines.append(fixed_line)
        
        # Post-process for empty blocks
        result_lines = []
        for i, line in enumerate(fixed_lines):
            result_lines.append(line)
            # Check if this line ends a block header without content
            if re.match(r'^(\s*)(if|elif|else|for|while|def|class|try|except|finally).*:$', line):
                # Check if next line is at same or lower indentation
                if i + 1 < len(fixed_lines):
                    current_indent = len(re.match(r'^(\s*)', line).group(1))
                    next_line = fixed_lines[i + 1]
                    next_indent = len(re.match(r'^(\s*)', next_line).group(1)) if next_line.strip() else 0
                    
                    if next_indent <= current_indent and next_line.strip():
                        # Insert pass statement
                        result_lines.append(' ' * (current_indent + 4) + 'pass')
                        fixes.append({
                            "line": i + 1,
                            "original": "",
                            "fixed": ' ' * (current_indent + 4) + 'pass',
                            "pattern": "empty_block"
                        })
                elif i + 1 == len(fixed_lines):
                    # Last line is a block header
                    current_indent = len(re.match(r'^(\s*)', line).group(1))
                    result_lines.append(' ' * (current_indent + 4) + 'pass')
                    fixes.append({
                        "line": i + 1,
                        "original": "",
                        "fixed": ' ' * (current_indent + 4) + 'pass',
                        "pattern": "empty_block"
                    })
        
        return '\n'.join(result_lines), fixes

    def _validate_syntax(self, content: str) -> Tuple[bool, Optional[str]]:
        """Validate Python syntax using ast.parse."""
        try:
            ast.parse(content)
            return True, None
        except SyntaxError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"

    def fix_file(self, filepath: Path) -> Dict[str, Any]:
        """Fix syntax errors in a single file."""
        result = {
            "path": str(filepath),
            "success": False,
            "backed_up": False,
            "fixes": [],
            "error": None
        }
        
        try:
            # Read the file
            with open(filepath, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Check if file has syntax errors
            valid, error = self._validate_syntax(original_content)
            if valid:
                result["success"] = True
                result["error"] = "No syntax errors found"
                return result
            
            # Create backup
            backup_path = self._backup_file(filepath)
            result["backed_up"] = True
            result["backup_path"] = str(backup_path)
            
            # Apply fixes
            fixed_content, fixes = self._apply_pattern_fixes(original_content)
            result["fixes"] = fixes
            
            # Validate fixed content
            valid, error = self._validate_syntax(fixed_content)
            
            if valid:
                # Write fixed content
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                result["success"] = True
                self.report["errors_fixed"] += len(fixes)
            else:
                # Try more aggressive fixes
                fixed_content = self._apply_aggressive_fixes(original_content, error)
                valid, error = self._validate_syntax(fixed_content)
                
                if valid:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(fixed_content)
                    result["success"] = True
                    result["fixes"].append({"type": "aggressive", "description": "Applied aggressive fixes"})
                else:
                    # Restore from backup
                    shutil.copy2(backup_path, filepath)
                    result["error"] = f"Could not fix syntax errors: {error}"
                    self.report["failed_files"].append(str(filepath))
        
        except Exception as e:
            result["error"] = f"Exception during processing: {str(e)}\n{traceback.format_exc()}"
            self.report["failed_files"].append(str(filepath))
        
        return result

    def _apply_aggressive_fixes(self, content: str, error: str) -> str:
        """Apply more aggressive fixes based on specific error messages."""
        lines = content.split('\n')
        
        # Extract line number from error if possible
        line_match = re.search(r'line (\d+)', error)
        if line_match:
            error_line = int(line_match.group(1)) - 1
            
            if 0 <= error_line < len(lines):
                line = lines[error_line]
                
                # Fix specific patterns based on error
                if 'invalid syntax' in error:
                    # Try to fix common invalid syntax patterns
                    # Remove trailing operators
                    line = re.sub(r'[+\-*/=]\s*$', '', line)
                    # Fix incomplete function calls
                    if '(' in line and ')' not in line:
                        line += ')'
                    # Fix incomplete string literals
                    if (line.count('"') % 2 \!= 0):
                        line += '"'
                    if (line.count("'") % 2 \!= 0):
                        line += "'"
                
                elif 'unexpected indent' in error:
                    # Fix indentation
                    line = line.lstrip()
                    # Guess indentation based on previous lines
                    if error_line > 0:
                        prev_indent = len(lines[error_line - 1]) - len(lines[error_line - 1].lstrip())
                        if lines[error_line - 1].rstrip().endswith(':'):
                            line = ' ' * (prev_indent + 4) + line
                        else:
                            line = ' ' * prev_indent + line
                
                lines[error_line] = line
        
        return '\n'.join(lines)

    def fix_directory(self, directory: str, batch_size: int = 10) -> None:
        """Fix all Python files in a directory in batches."""
        python_files = []
        
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        
        total_files = len(python_files)
        print(f"Found {total_files} Python files to process")
        
        # Process in batches
        for i in range(0, total_files, batch_size):
            batch = python_files[i:i + batch_size]
            print(f"\nProcessing batch {i//batch_size + 1}/{(total_files + batch_size - 1)//batch_size}")
            
            for filepath in batch:
                print(f"Processing: {filepath}")
                result = self.fix_file(filepath)
                
                self.report["files_processed"] += 1
                if result["success"] and result["fixes"]:
                    self.report["files_fixed"] += 1
                    self.report["fixes_applied"].append({
                        "file": str(filepath),
                        "fixes": result["fixes"]
                    })
                    print(f"   Fixed {len(result['fixes'])} issues")
                elif result["success"]:
                    print(f"   No syntax errors found")
                else:
                    print(f"   Failed: {result['error']}")

    def generate_report(self, output_file: str = "syntax_fix_report.json") -> None:
        """Generate a detailed report of all fixes."""
        self.report["end_time"] = datetime.now().isoformat()
        
        # Calculate summary statistics
        self.report["summary"] = {
            "total_files": self.report["files_processed"],
            "files_with_errors": self.report["files_fixed"],
            "total_errors_fixed": self.report["errors_fixed"],
            "failed_files_count": len(self.report["failed_files"]),
            "success_rate": (self.report["files_processed"] - len(self.report["failed_files"])) / self.report["files_processed"] * 100 if self.report["files_processed"] > 0 else 0
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(self.report, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print("SYNTAX FIX SUMMARY")
        print("="*50)
        print(f"Total files processed: {self.report['summary']['total_files']}")
        print(f"Files with errors fixed: {self.report['summary']['files_with_errors']}")
        print(f"Total errors fixed: {self.report['summary']['total_errors_fixed']}")
        print(f"Failed files: {self.report['summary']['failed_files_count']}")
        print(f"Success rate: {self.report['summary']['success_rate']:.2f}%")
        print(f"\nDetailed report saved to: {output_file}")
        print(f"Backups saved in: {self.backup_dir}/")

def main():
    """Main function to run the syntax fixer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix Python syntax errors comprehensively")
    parser.add_argument("directory", nargs="?", default="src", help="Directory to process (default: src)")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of files to process in each batch")
    parser.add_argument("--backup-dir", default="syntax_backups", help="Directory for file backups")
    parser.add_argument("--report", default="syntax_fix_report.json", help="Output report file")
    
    args = parser.parse_args()
    
    # Create fixer instance
    fixer = SyntaxFixer(backup_dir=args.backup_dir)
    
    # Fix all files
    fixer.fix_directory(args.directory, batch_size=args.batch_size)
    
    # Generate report
    fixer.generate_report(args.report)

if __name__ == "__main__":
    main()
EOF < /dev/null
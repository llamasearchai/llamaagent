#!/usr/bin/env python3
"""
Comprehensive Python Syntax Error Fixer v2
Enhanced version with more robust pattern matching and specific error handling.
"""

import ast
import os
import re
import shutil
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import traceback

class ComprehensiveSyntaxFixer:
    """Enhanced syntax error fixer with comprehensive pattern-based fixes."""
    
    def __init__(self, backup_dir: str = "syntax_backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        self.report = {
            "start_time": datetime.now().isoformat(),
            "files_processed": 0,
            "files_fixed": 0,
            "errors_fixed": 0,
            "failed_files": [],
            "fixes_applied": [],
            "specific_errors": {}
        }

    def _backup_file(self, filepath: Path) -> Path:
        """Create a backup of the file before modifying."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create subdirectory based on original path
        relative_path = filepath.relative_to(Path('src')) if filepath.is_relative_to(Path('src')) else filepath
        backup_subdir = self.backup_dir / relative_path.parent
        backup_subdir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_subdir / f"{filepath.name}.{timestamp}.bak"
        shutil.copy2(filepath, backup_path)
        return backup_path

    def _fix_specific_file_errors(self, filepath: Path, content: str) -> str:
        """Fix errors specific to certain files based on the scan results."""
        filename = filepath.name
        lines = content.split('\n')
        
        # Fix specific known errors
        if filename == "prompt_templates.py" and len(lines) > 568:
            # Line 569: self.add_template(custom_template)
            if lines[568].strip() == "self.add_template(custom_template)":
                # This line might be missing proper indentation or context
                lines[568] = "            self.add_template(custom_template)"
        
        elif filename == "optimization.py" and len(lines) > 256:
            # Line 257: PromptCandidate(
            if "PromptCandidate(" in lines[256]:
                # Fix unexpected indent by adjusting to proper level
                lines[256] = "        return PromptCandidate("
        
        elif filename == "_openai_stub.py" and len(lines) > 360:
            # Line 361: missing closing parenthesis
            if 'if api_key and ("test-key" in str(api_key) or "test_api" in str(api_key):' in lines[360]:
                lines[360] = '        if api_key and ("test-key" in str(api_key) or "test_api" in str(api_key)):'
        
        elif filename == "performance.py" and len(lines) > 230:
            # Line 231: return await future
            if lines[230].strip() == "return await future":
                # Ensure proper indentation
                lines[230] = "            return await future"
        
        elif filename == "multimodal_reasoning.py" and len(lines) > 260:
            # Line 261: f-string closing issue
            if 'data_summary = f"Structured data with {len(data)} fields: {list(data.keys()}"' in lines[260]:
                lines[260] = '            data_summary = f"Structured data with {len(data)} fields: {list(data.keys())}"'
        
        elif filename == "enhanced_shell_cli.py" and len(lines) > 325:
            # Line 326: TaskInput(
            if "TaskInput(" in lines[325]:
                lines[325] = "            task = TaskInput("
        
        elif filename == "enhanced_cli.py" and len(lines) > 332:
            # Line 333: missing closing parenthesis
            if 'config_table.add_row("Max Tokens", str(self.config.llm.max_tokens)' in lines[332]:
                lines[332] = '            config_table.add_row("Max Tokens", str(self.config.llm.max_tokens))'
        
        elif filename == "inference_engine.py" and len(lines) > 171:
            # Line 172: unmatched ')'
            if 'self.performance_stats: Dict[str, Any] = {}' in lines[171]:
                lines[171] = '        self.performance_stats: Dict[str, Any] = {}'
        
        elif filename == "adaptive_learning.py" and len(lines) > 122:
            # Line 123: for py_file in py_files[:10]:  # Limit for demo
            if "for py_file in py_files[:10]:  # Limit for demo" in lines[122]:
                lines[122] = "        for py_file in py_files[:10]:  # Limit for demo"
        
        elif filename == "frontier_evaluation.py" and len(lines) > 107:
            # Line 108: EvaluationTask()
            if "EvaluationTask()" in lines[107]:
                lines[107] = "            return EvaluationTask("
        
        elif filename == "gaia_benchmark.py" and len(lines) > 126:
            # Line 127: for task in self.tasks:
            if "for task in self.tasks:" in lines[126]:
                lines[126] = "        for task in self.tasks:"
        
        elif filename == "task_analyzer.py" and len(lines) > 351:
            # Line 352: if task.count("\n") > 5 or len(task.split() > 200:
            if 'if task.count("\\n") > 5 or len(task.split() > 200:' in lines[351]:
                lines[351] = '        if task.count("\\n") > 5 or len(task.split()) > 200:'
        
        elif filename == "advanced_monitoring.py" and len(lines) > 26:
            # Line 27: from typing import 0", 100", "", "<, "==
            if 'from typing import 0", 100", "", "<, "==' in lines[26]:
                lines[26] = 'from typing import Dict, List, Any, Optional, Union, Tuple, Set'
        
        elif filename == "model_comparison.py" and len(lines) > 283:
            # Line 284: if isinstance(value, (int, float):)
            if 'if isinstance(value, (int, float):)' in lines[283]:
                lines[283] = '            if isinstance(value, (int, float)):'
        
        elif filename == "base.py" and len(lines) > 35:
            # Line 36: unmatched ')'
            if 'node_id: str = field(default_factory=lambda: str(uuid.uuid4())' in content:
                for i, line in enumerate(lines):
                    if 'node_id: str = field(default_factory=lambda: str(uuid.uuid4())' in line:
                        lines[i] = '    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))'
        
        return '\n'.join(lines)

    def _apply_comprehensive_fixes(self, content: str, filepath: Path) -> Tuple[str, List[Dict[str, Any]]]:
        """Apply comprehensive pattern-based fixes to content."""
        fixes = []
        lines = content.split('\n')
        fixed_lines = []
        
        # First, apply file-specific fixes
        content = self._fix_specific_file_errors(filepath, content)
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            original_line = line
            fixed_line = line
            
            # Fix missing closing parentheses
            paren_count = fixed_line.count('(') - fixed_line.count(')')
            if paren_count > 0:
                fixed_line += ')' * paren_count
                if fixed_line != original_line:
                    fixes.append({
                        "line": i + 1,
                        "type": "missing_closing_paren",
                        "original": original_line,
                        "fixed": fixed_line
                    })
            
            # Fix missing closing brackets
            bracket_count = fixed_line.count('[') - fixed_line.count(']')
            if bracket_count > 0:
                fixed_line += ']' * bracket_count
                if fixed_line != original_line:
                    fixes.append({
                        "line": i + 1,
                        "type": "missing_closing_bracket",
                        "original": original_line,
                        "fixed": fixed_line
                    })
            
            # Fix missing closing braces
            brace_count = fixed_line.count('{') - fixed_line.count('}')
            if brace_count > 0:
                fixed_line += '}' * brace_count
                if fixed_line != original_line:
                    fixes.append({
                        "line": i + 1,
                        "type": "missing_closing_brace",
                        "original": original_line,
                        "fixed": fixed_line
                    })
            
            # Fix f-string issues
            if 'f"' in fixed_line or "f'" in fixed_line:
                # Fix unclosed f-strings
                if fixed_line.count('f"') > fixed_line.count('"', fixed_line.find('f"') + 2):
                    fixed_line += '"'
                    fixes.append({
                        "line": i + 1,
                        "type": "unclosed_fstring",
                        "original": original_line,
                        "fixed": fixed_line
                    })
                
                # Fix f-string brace mismatch
                fstring_match = re.search(r'f["\']([^"\']*)', fixed_line)
                if fstring_match:
                    fstring_content = fstring_match.group(1)
                    if fstring_content.count('{') != fstring_content.count('}'):
                        if fstring_content.count('{') > fstring_content.count('}'):
                            fixed_line = fixed_line.rstrip('"\'') + '}' * (fstring_content.count('{') - fstring_content.count('}')) + fixed_line[-1] if fixed_line and fixed_line[-1] in '"\'?' else '"'
                            fixes.append({
                                "line": i + 1,
                                "type": "fstring_brace_mismatch",
                                "original": original_line,
                                "fixed": fixed_line
                            })
            
            # Fix missing colons
            if re.match(r'^\s*(if|elif|else|for|while|def|class|try|except|finally|with)\s+.*[^:]$', fixed_line):
                fixed_line += ':'
                fixes.append({
                    "line": i + 1,
                    "type": "missing_colon",
                    "original": original_line,
                    "fixed": fixed_line
                })
            
            # Fix assignment in if statement
            if re.search(r'\bif\s+\w+\s*=\s*[^=]', fixed_line):
                fixed_line = re.sub(r'\bif\s+(\w+)\s*=\s*([^=])', r'if \1 == \2', fixed_line)
                fixes.append({
                    "line": i + 1,
                    "type": "assignment_in_if",
                    "original": original_line,
                    "fixed": fixed_line
                })
            
            # Fix trailing comma before closing parenthesis
            if re.search(r',\s*\)', fixed_line):
                fixed_line = re.sub(r',\s*\)', ')', fixed_line)
                fixes.append({
                    "line": i + 1,
                    "type": "trailing_comma",
                    "original": original_line,
                    "fixed": fixed_line
                })
            
            # Fix indentation (ensure multiples of 4 spaces)
            if fixed_line.strip():
                indent = len(fixed_line) - len(fixed_line.lstrip())
                if indent % 4 != 0:
                    new_indent = round(indent / 4) * 4
                    fixed_line = ' ' * new_indent + fixed_line.lstrip()
                    fixes.append({
                        "line": i + 1,
                        "type": "indentation",
                        "original": original_line,
                        "fixed": fixed_line
                    })
            
            fixed_lines.append(fixed_line)
        
        # Post-process for empty blocks
        result_lines = []
        for i, line in enumerate(fixed_lines):
            result_lines.append(line)
            
            # Check if this line ends a block header without content
            if re.match(r'^\s*(if|elif|else|for|while|def|class|try|except|finally|with).*:$', line):
                # Check if next line is empty or at same/lower indentation
                if i + 1 < len(fixed_lines):
                    current_indent = len(line) - len(line.lstrip())
                    next_line = fixed_lines[i + 1] if i + 1 < len(fixed_lines) else ""
                    next_indent = len(next_line) - len(next_line.lstrip()) if next_line.strip() else 0
                    
                    if not next_line.strip() or next_indent <= current_indent:
                        # Insert pass statement
                        pass_line = ' ' * (current_indent + 4) + 'pass'
                        result_lines.append(pass_line)
                        fixes.append({
                            "line": i + 2,
                            "type": "empty_block",
                            "original": "",
                            "fixed": pass_line
                        })
                else:
                    # Last line is a block header
                    current_indent = len(line) - len(line.lstrip())
                    pass_line = ' ' * (current_indent + 4) + 'pass'
                    result_lines.append(pass_line)
                    fixes.append({
                        "line": i + 2,
                        "type": "empty_block",
                        "original": "",
                        "fixed": pass_line
                    })
        
        return '\n'.join(result_lines), fixes

    def _validate_syntax(self, content: str) -> Tuple[bool, Optional[str], Optional[int]]:
        """Validate Python syntax using ast.parse."""
        try:
            ast.parse(content)
            return True, None, None
        except SyntaxError as e:
            return False, str(e), e.lineno
        except Exception as e:
            return False, f"Unexpected error: {str(e)}", None

    def fix_file(self, filepath: Path) -> Dict[str, Any]:
        """Fix syntax errors in a single file."""
        result = {
            "path": str(filepath),
            "success": False,
            "backed_up": False,
            "fixes": [],
            "error": None,
            "iterations": 0
        }
        
        try:
            # Read the file
            with open(filepath, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Check if file has syntax errors
            valid, error, line_no = self._validate_syntax(original_content)
            if valid:
                result["success"] = True
                result["error"] = "No syntax errors found"
                return result
            
            # Store the specific error
            if filepath.name not in self.report["specific_errors"]:
                self.report["specific_errors"][filepath.name] = []
            self.report["specific_errors"][filepath.name].append({
                "error": error,
                "line": line_no
            })
            
            # Create backup
            backup_path = self._backup_file(filepath)
            result["backed_up"] = True
            result["backup_path"] = str(backup_path)
            
            # Try to fix with multiple iterations
            fixed_content = original_content
            max_iterations = 5
            
            for iteration in range(max_iterations):
                result["iterations"] = iteration + 1
                
                # Apply comprehensive fixes
                fixed_content, fixes = self._apply_comprehensive_fixes(fixed_content, filepath)
                result["fixes"].extend(fixes)
                
                # Validate fixed content
                valid, error, line_no = self._validate_syntax(fixed_content)
                
                if valid:
                    # Write fixed content
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(fixed_content)
                    result["success"] = True
                    self.report["errors_fixed"] += len(result["fixes"])
                    break
                else:
                    # Try to fix the specific error
                    if line_no and line_no > 0:
                        lines = fixed_content.split('\n')
                        if line_no <= len(lines):
                            error_line = lines[line_no - 1]
                            
                            # Apply specific fixes based on error message
                            if 'invalid syntax' in error:
                                # Remove trailing operators
                                error_line = re.sub(r'[+\-*/=]\s*$', '', error_line)
                                # Ensure quotes are closed
                                if error_line.count('"') % 2 != 0:
                                    error_line += '"'
                                if error_line.count("'") % 2 != 0:
                                    error_line += "'"
                                lines[line_no - 1] = error_line
                                fixed_content = '\n'.join(lines)
                            elif 'unexpected indent' in error:
                                # Fix indentation
                                lines[line_no - 1] = lines[line_no - 1].lstrip()
                                if line_no > 1 and lines[line_no - 2].rstrip().endswith(':'):
                                    prev_indent = len(lines[line_no - 2]) - len(lines[line_no - 2].lstrip())
                                    lines[line_no - 1] = ' ' * (prev_indent + 4) + lines[line_no - 1]
                                fixed_content = '\n'.join(lines)
            
            if not result["success"]:
                # Restore from backup if we couldn't fix it
                shutil.copy2(backup_path, filepath)
                result["error"] = f"Could not fix syntax errors after {max_iterations} iterations: {error}"
                self.report["failed_files"].append(str(filepath))
        
        except Exception as e:
            result["error"] = f"Exception during processing: {str(e)}\n{traceback.format_exc()}"
            self.report["failed_files"].append(str(filepath))
        
        return result

    def fix_directory(self, directory: str, batch_size: int = 10) -> None:
        """Fix all Python files in a directory in batches."""
        python_files = []
        
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py') and not file.startswith('__pycache__'):
                    python_files.append(Path(root) / file)
        
        # Sort files for consistent processing
        python_files.sort()
        
        total_files = len(python_files)
        print(f"Found {total_files} Python files to process")
        
        # Process in batches
        for i in range(0, total_files, batch_size):
            batch = python_files[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_files + batch_size - 1) // batch_size
            
            print(f"\n{'='*50}")
            print(f"Processing batch {batch_num}/{total_batches}")
            print(f"{'='*50}")
            
            for filepath in batch:
                print(f"\nProcessing: {filepath}")
                result = self.fix_file(filepath)
                
                self.report["files_processed"] += 1
                if result["success"] and result["fixes"]:
                    self.report["files_fixed"] += 1
                    self.report["fixes_applied"].append({
                        "file": str(filepath),
                        "fixes": result["fixes"],
                        "iterations": result["iterations"]
                    })
                    print(f"   Fixed {len(result['fixes'])} issues in {result['iterations']} iteration(s)")
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
        
        # Save full report
        with open(output_file, 'w') as f:
            json.dump(self.report, f, indent=2)
        
        # Save failed files list
        if self.report["failed_files"]:
            with open("failed_files.txt", 'w') as f:
                for file in self.report["failed_files"]:
                    f.write(f"{file}\n")
        
        # Print summary
        print("\n" + "="*70)
        print("COMPREHENSIVE SYNTAX FIX SUMMARY")
        print("="*70)
        print(f"Total files processed: {self.report['summary']['total_files']}")
        print(f"Files with errors fixed: {self.report['summary']['files_with_errors']}")
        print(f"Total errors fixed: {self.report['summary']['total_errors_fixed']}")
        print(f"Failed files: {self.report['summary']['failed_files_count']}")
        print(f"Success rate: {self.report['summary']['success_rate']:.2f}%")
        print(f"\nDetailed report saved to: {output_file}")
        print(f"Backups saved in: {self.backup_dir}/")
        
        if self.report["failed_files"]:
            print(f"\nFailed files list saved to: failed_files.txt")
            print("\nFailed files:")
            for file in self.report["failed_files"][:10]:
                print(f"  - {file}")
            if len(self.report["failed_files"]) > 10:
                print(f"  ... and {len(self.report['failed_files']) - 10} more")

def main():
    """Main function to run the comprehensive syntax fixer."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Comprehensive Python Syntax Error Fixer v2",
        epilog="This tool will create backups of all modified files and generate a detailed report."
    )
    parser.add_argument(
        "directory", 
        nargs="?", 
        default="src", 
        help="Directory to process (default: src)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=10, 
        help="Number of files to process in each batch (default: 10)"
    )
    parser.add_argument(
        "--backup-dir", 
        default="syntax_backups", 
        help="Directory for file backups (default: syntax_backups)"
    )
    parser.add_argument(
        "--report", 
        default="syntax_fix_report.json", 
        help="Output report file (default: syntax_fix_report.json)"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("COMPREHENSIVE PYTHON SYNTAX ERROR FIXER v2")
    print("="*70)
    print(f"Directory: {args.directory}")
    print(f"Batch size: {args.batch_size}")
    print(f"Backup directory: {args.backup_dir}")
    print(f"Report file: {args.report}")
    print("="*70)
    
    # Create fixer instance
    fixer = ComprehensiveSyntaxFixer(backup_dir=args.backup_dir)
    
    # Fix all files
    fixer.fix_directory(args.directory, batch_size=args.batch_size)
    
    # Generate report
    fixer.generate_report(args.report)

if __name__ == "__main__":
    main()
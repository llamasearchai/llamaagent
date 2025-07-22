#!/usr/bin/env python3
"""
Final comprehensive syntax fixer for remaining 21 files with errors.
This script targets specific known issues in each file.
"""

import ast
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Map of files to their specific fixes
FILE_SPECIFIC_FIXES = {
    "optimization.py": {
        "line": 326,
        "pattern": r"(\s+)return\s+{([^}]+)$",
        "replacement": r"\1return {\2}"
    },
    "performance.py": {
        "line": 80,
        "search": "metrics.append(",
        "fix": "add closing parenthesis to append call"
    },
    "memory_manager.py": {
        "line": 55,
        "error": "unmatched ')'",
        "fix": "remove extra closing parenthesis"
    },
    "chain_engine.py": {
        "line": 54,
        "fix": "add missing colon after method definition"
    },
    "inference_engine.py": {
        "line": 135,
        "fix": "fix f-string or parenthesis mismatch"
    },
    "adaptive_learning.py": {
        "line": 116,
        "fix": "fix field definition syntax"
    },
    "frontier_evaluation.py": {
        "line": 305,
        "fix": "fix append or method call syntax"
    },
    "master_diagnostics.py": {
        "line": 552,
        "fix": "fix complex expression or return statement"
    },
    "system_validator.py": {
        "line": 66,
        "error": "unexpected indent",
        "fix": "fix indentation"
    },
    "code_analyzer.py": {
        "line": 196,
        "error": "unexpected indent",
        "fix": "fix indentation after except block"
    }
}

def fix_specific_file(filepath: Path, error_line: int, error_type: str) -> bool:
    """Fix a specific file based on known error patterns."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if error_line > len(lines):
            return False
        
        original_lines = lines.copy()
        idx = error_line - 1  # Convert to 0-based index
        
        # Apply specific fixes based on error type and context
        current_line = lines[idx]
        
        # Fix 1: Missing closing parenthesis in append() or function calls
        if "append(" in current_line and current_line.count('(') > current_line.count(')'):
            # Add missing closing parenthesis
            lines[idx] = current_line.rstrip() + ')\n'
        
        # Fix 2: Unmatched closing parenthesis
        elif "unmatched ')'" in error_type:
            # Remove extra closing parenthesis
            lines[idx] = re.sub(r'\)+(\s*[,;:\n])', r')\1', current_line)
        
        # Fix 3: Missing colon after control structures
        elif re.match(r'\s*(def|class|if|elif|else|for|while|try|except|finally|async def)\s+.*[^:]$', current_line):
            lines[idx] = current_line.rstrip() + ':\n'
        
        # Fix 4: Indentation errors
        elif "indent" in error_type:
            # Fix indentation based on context
            stripped = current_line.lstrip()
            if idx > 0:
                # Find previous non-empty line
                prev_idx = idx - 1
                while prev_idx >= 0 and not lines[prev_idx].strip():
                    prev_idx -= 1
                
                if prev_idx >= 0:
                    prev_line = lines[prev_idx]
                    prev_indent = len(prev_line) - len(prev_line.lstrip())
                    
                    # If previous line ends with colon, indent more
                    if prev_line.rstrip().endswith(':'):
                        new_indent = prev_indent + 4
                    else:
                        new_indent = prev_indent
                    
                    lines[idx] = ' ' * new_indent + stripped
        
        # Fix 5: F-string errors
        elif 'f"' in current_line or "f'" in current_line:
            # Fix missing closing braces in f-strings
            if current_line.count('{') > current_line.count('}'):
                # Add missing closing braces
                missing = current_line.count('{') - current_line.count('}')
                # Find the f-string end quote and add braces before it
                lines[idx] = re.sub(r'(["\'])\s*$', r'}' * missing + r'\1\n', current_line)
        
        # Fix 6: Dict/list literal issues
        elif '{' in current_line and current_line.strip().startswith('return'):
            # Ensure return statement with dict is properly formatted
            if not current_line.strip().endswith(('{', '}')):
                lines[idx] = current_line.rstrip() + ' }\n'
        
        # Write back if changed
        if lines != original_lines:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False

def main():
    """Main function to fix all remaining syntax errors."""
    # List of files with known syntax errors
    error_files = [
        ("src/llamaagent/prompting/optimization.py", 326, "invalid syntax"),
        ("src/llamaagent/optimization/performance.py", 80, "invalid syntax"),
        ("src/llamaagent/reasoning/memory_manager.py", 55, "unmatched ')'"),
        ("src/llamaagent/reasoning/chain_engine.py", 54, "invalid syntax"),
        ("src/llamaagent/ml/inference_engine.py", 135, "invalid syntax"),
        ("src/llamaagent/evolution/adaptive_learning.py", 116, "invalid syntax"),
        ("src/llamaagent/benchmarks/frontier_evaluation.py", 305, "invalid syntax"),
        ("src/llamaagent/diagnostics/master_diagnostics.py", 552, "invalid syntax"),
        ("src/llamaagent/diagnostics/system_validator.py", 66, "unexpected indent"),
        ("src/llamaagent/diagnostics/code_analyzer.py", 196, "unexpected indent"),
        ("src/llamaagent/cli/enhanced_shell_cli.py", 326, "invalid syntax"),
        ("src/llamaagent/cli/enhanced_cli.py", 333, "invalid syntax"),
        ("src/llamaagent/cli/openai_cli.py", 179, "invalid syntax"),
        ("src/llamaagent/cli/role_manager.py", 377, "invalid syntax"),
        ("src/llamaagent/cli/code_generator.py", 246, "invalid syntax"),
        ("src/llamaagent/cli/diagnostics_cli.py", 156, "invalid syntax"),
        ("src/llamaagent/monitoring/alerting.py", 154, "invalid syntax"),
        ("src/llamaagent/monitoring/advanced_monitoring.py", 27, "invalid syntax"),
        ("src/llamaagent/monitoring/middleware.py", 167, "invalid syntax"),
        ("src/llamaagent/monitoring/metrics_collector.py", 599, "unexpected indent"),
        ("src/llamaagent/knowledge/knowledge_generator.py", 685, "invalid syntax"),
    ]
    
    print(f"Attempting to fix {len(error_files)} files with syntax errors...")
    print("=" * 60)
    
    fixed_count = 0
    still_failing = []
    
    for filepath, line_num, error_type in error_files:
        path = Path(filepath)
        if not path.exists():
            print(f"FAIL File not found: {filepath}")
            continue
        
        print(f"\nProcessing: {filepath} (line {line_num})")
        
        # Try to fix the file
        if fix_specific_file(path, line_num, error_type):
            # Verify the fix
            try:
                with open(path, 'r') as f:
                    ast.parse(f.read())
                print(f"  PASS Fixed successfully!")
                fixed_count += 1
            except SyntaxError as e:
                print(f"  WARNING:  Still has errors after fix: {e}")
                still_failing.append((filepath, str(e)))
        else:
            print(f"  FAIL No changes made")
            still_failing.append((filepath, error_type))
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Fixed {fixed_count} out of {len(error_files)} files")
    
    if still_failing:
        print(f"\n{len(still_failing)} files still need manual fixes:")
        for filepath, error in still_failing[:10]:
            print(f"  - {filepath}: {error}")
        
        if len(still_failing) > 10:
            print(f"  ... and {len(still_failing) - 10} more")

if __name__ == "__main__":
    main()
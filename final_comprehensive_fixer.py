#!/usr/bin/env python3
"""Final comprehensive syntax fixer for remaining issues."""

import re
import os

def fix_syntax_issues(content: str, filepath: str) -> str:
    """Apply targeted fixes for specific file issues."""
    
    # Fix comprehensive_diagnostic_system.py line 607
    if "comprehensive_diagnostic_system.py" in filepath:
        # Fix missing closing parenthesis on line around 607
        lines = content.split('\n')
        for i, line in enumerate(lines):
            # Fix missing parenthesis in function calls
            if 'append(' in line and line.count('(') > line.count(')'):
                # Count tabs/spaces for proper indentation
                indent = len(line) - len(line.lstrip())
                # Check if there's a dict literal on next line
                if i + 1 < len(lines) and '{' in lines[i + 1]:
                    # Fix append with dict on next line
                    lines[i] = line.rstrip() + '({'
                    if i + 1 < len(lines):
                        lines[i + 1] = lines[i + 1].replace('{', '', 1)
        content = '\n'.join(lines)
    
    # Fix system_validator.py indentation
    if "system_validator.py" in filepath:
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'issues.append()' in line:
                lines[i] = line.replace('append()', 'append(')
                # Fix the dict on next line
                if i + 1 < len(lines) and '{' in lines[i + 1]:
                    lines[i + 1] = lines[i + 1].lstrip()
                    indent = len(lines[i]) - len(lines[i].lstrip())
                    lines[i + 1] = ' ' * (indent + 4) + lines[i + 1]
        content = '\n'.join(lines)
    
    # Fix master_diagnostics.py line 552
    if "master_diagnostics.py" in filepath:
        # Fix missing closing parenthesis
        content = re.sub(r'(\w+)\s*=\s*field\(default_factory=lambda:\s*datetime\.now\(timezone\.utc\)$', 
                         r'\1 = field(default_factory=lambda: datetime.now(timezone.utc))', 
                         content, flags=re.MULTILINE)
    
    # Fix benchmark_engine.py line 423
    if "benchmark_engine.py" in filepath:
        # Fix missing closing brackets/parenthesis
        lines = content.split('\n')
        for i, line in enumerate(lines):
            # Count open and close brackets
            if line.count('[') > line.count(']'):
                # Look for list comprehension pattern
                if 'for' in line:
                    lines[i] = line.rstrip() + ']'
            elif line.count('(') > line.count(')'):
                lines[i] = line.rstrip() + ')'
        content = '\n'.join(lines)
    
    # Fix golden_dataset.py line 125
    if "golden_dataset.py" in filepath:
        # Fix missing closing parenthesis
        content = re.sub(r'return list\(self\.samples\.keys\(\)$', 
                         r'return list(self.samples.keys())', 
                         content, flags=re.MULTILINE)
    
    # Fix model_comparison.py line 102
    if "model_comparison.py" in filepath:
        # Fix missing closing parenthesis in field definition
        content = re.sub(r'(\w+):\s*datetime\s*=\s*field\(default_factory=lambda:\s*datetime\.now\(timezone\.utc\)$',
                         r'\1: datetime = field(default_factory=lambda: datetime.now(timezone.utc))',
                         content, flags=re.MULTILINE)
    
    # Fix evolution/adaptive_learning.py line 66
    if "adaptive_learning.py" in filepath:
        # Fix missing closing parenthesis
        content = re.sub(r'created_at:\s*datetime\s*=\s*field\(default_factory=lambda:\s*datetime\.now\(timezone\.utc\)$',
                         r'created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))',
                         content, flags=re.MULTILINE)
    
    # Fix ml/inference_engine.py line 135
    if "inference_engine.py" in filepath:
        # Fix missing closing parenthesis in list methods
        content = re.sub(r'\.append\(\s*$', '.append(', content, flags=re.MULTILINE)
        content = re.sub(r'return\s+list\(([^)]+)$', r'return list(\1)', content, flags=re.MULTILINE)
    
    # Fix monitoring files
    if "monitoring/" in filepath:
        # Fix metrics_collector.py line 375
        if "metrics_collector.py" in filepath:
            content = re.sub(r'total_requests\s*=\s*sum\(m\.request_count\s+for\s+m\s+in\s+self\.metrics\.values\(\)$',
                             r'total_requests = sum(m.request_count for m in self.metrics.values())',
                             content, flags=re.MULTILINE)
        
        # Fix middleware.py line 91
        if "middleware.py" in filepath:
            content = re.sub(r'async\s+def\s+\w+\([^)]*\)\s*=\s*Depends\([^)]+\)$',
                             lambda m: m.group(0) + '):', content, flags=re.MULTILINE)
    
    # Fix performance.py line 80
    if "performance.py" in filepath:
        content = re.sub(r'timestamp:\s*datetime\s*=\s*field\(default_factory=lambda:\s*datetime\.now\(timezone\.utc\)$',
                         r'timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))',
                         content, flags=re.MULTILINE)
    
    # Fix reasoning files
    if "reasoning/" in filepath:
        # Fix common issues with missing parentheses
        content = re.sub(r'\.extend\(([^)]+)$', r'.extend(\1)', content, flags=re.MULTILINE)
        content = re.sub(r'results\.append\(\(([^,]+),\s*([^)]+)\)$', r'results.append((\1, \2))', content, flags=re.MULTILINE)
    
    # Fix spawning files
    if "spawning/" in filepath:
        # Fix missing closing parentheses in function calls
        content = re.sub(r'agents\.append\(([^)]+)$', r'agents.append(\1)', content, flags=re.MULTILINE)
    
    # General fixes for all files
    # Fix incomplete list comprehensions
    content = re.sub(r'\[\s*(\w+)\s+for\s+(\w+)\s+in\s+([^]]+)$', r'[\1 for \2 in \3]', content, flags=re.MULTILINE)
    
    # Fix missing closing parenthesis in return statements
    content = re.sub(r'return\s+(\w+)\(([^)]+)$', r'return \1(\2)', content, flags=re.MULTILINE)
    
    # Fix backslash escapes in comparisons
    content = content.replace(r'\!=', '!=')
    content = content.replace(r'\==', '==')
    
    return content

def main():
    """Process all Python files with syntax errors."""
    # Find files with syntax errors
    import ast
    
    error_files = []
    for root, dirs, files in os.walk('.'):
        # Skip venv and other non-source directories
        if any(skip in root for skip in ['venv', '__pycache__', '.git', 'node_modules']):
            continue
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    ast.parse(content)
                except SyntaxError as e:
                    error_files.append((filepath, e.lineno, e.msg))
    
    print(f"Found {len(error_files)} files with syntax errors")
    
    # Fix each file
    fixed_count = 0
    for filepath, lineno, msg in error_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Apply fixes
            fixed_content = fix_syntax_issues(content, filepath)
            
            if fixed_content != content:
                # Verify the fix
                try:
                    ast.parse(fixed_content)
                    # Write back if syntax is valid
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(fixed_content)
                    print(f"Fixed: {filepath} (line {lineno}: {msg})")
                    fixed_count += 1
                except SyntaxError:
                    print(f"Failed to fix: {filepath} (line {lineno}: {msg})")
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    print(f"\nFixed {fixed_count} files")
    
    # Re-check for remaining errors
    remaining_errors = 0
    for root, dirs, files in os.walk('.'):
        if any(skip in root for skip in ['venv', '__pycache__', '.git', 'node_modules']):
            continue
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    ast.parse(content)
                except SyntaxError:
                    remaining_errors += 1
    
    print(f"Remaining syntax errors: {remaining_errors}")

if __name__ == "__main__":
    main()
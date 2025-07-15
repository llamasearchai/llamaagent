#!/usr/bin/env python3
"""Batch fix common syntax errors"""

import ast
import re
import os

def fix_file(filepath):
    """Fix common syntax errors in a file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        
        # Fix common patterns
        # 1. Fix backslash escapes in comparisons
        content = re.sub(r'(\s+)if\s+(.+?)\s*\\!=\s*(.+?):', r'\1if \2 != \3:', content)
        content = re.sub(r'(\s+)if\s+(.+?)\s*\\==\s*(.+?):', r'\1if \2 == \3:', content)
        
        # 2. Fix missing closing parentheses in function calls
        content = re.sub(r'(\w+)\(([^)]+)\)(\))', r'\1(\2)', content)
        
        # 3. Fix unmatched parentheses at end of lines
        lines = content.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            # Count parentheses
            open_count = line.count('(')
            close_count = line.count(')')
            
            # If more closes than opens and line ends with )
            if close_count > open_count and line.rstrip().endswith(')'):
                # Check if it's likely an extra closing paren
                if re.search(r'\)\)$', line.rstrip()) and not re.search(r'\(\(', line):
                    line = line.rstrip()[:-1] + '\n'
            
            fixed_lines.append(line)
        
        content = '\n'.join(fixed_lines)
        
        # Write back if changed
        if content != original:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed: {filepath}")
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

# List of files to fix
files_to_fix = [
    "./src/llamaagent/api/openai_comprehensive_api.py",
    "./src/llamaagent/api/production_app.py",
    "./src/llamaagent/benchmarks/frontier_evaluation.py",
    "./src/llamaagent/benchmarks/gaia_benchmark.py",
    "./src/llamaagent/diagnostics/code_analyzer.py",
    "./src/llamaagent/diagnostics/dependency_checker.py",
    "./src/llamaagent/diagnostics/master_diagnostics.py",
    "./src/llamaagent/diagnostics/system_validator.py",
    "./src/llamaagent/evaluation/benchmark_engine.py",
    "./src/llamaagent/evaluation/golden_dataset.py",
    "./src/llamaagent/evaluation/model_comparison.py",
    "./src/llamaagent/evolution/adaptive_learning.py",
    "./src/llamaagent/ml/inference_engine.py",
    "./src/llamaagent/monitoring/alerting.py",
    "./src/llamaagent/monitoring/metrics_collector.py",
    "./src/llamaagent/monitoring/middleware.py",
    "./src/llamaagent/optimization/performance.py",
    "./src/llamaagent/orchestration/adaptive_orchestra.py",
    "./src/llamaagent/prompting/optimization.py",
    "./src/llamaagent/reasoning/chain_engine.py",
    "./src/llamaagent/reasoning/memory_manager.py",
    "./src/llamaagent/routing/metrics.py",
    "./src/llamaagent/routing/provider_registry.py",
    "./src/llamaagent/routing/strategies.py",
    "./src/llamaagent/routing/task_analyzer.py"
]

if __name__ == "__main__":
    fixed = 0
    for filepath in files_to_fix:
        if os.path.exists(filepath):
            if fix_file(filepath):
                fixed += 1
    
    print(f"\nFixed {fixed} files")
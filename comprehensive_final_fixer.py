#!/usr/bin/env python3
"""Comprehensive syntax fixer for remaining issues"""

import ast
import re
import os

def fix_specific_issues(filepath):
    """Fix specific known issues in files"""
    
    specific_fixes = {
        "./src/llamaagent/api/openai_comprehensive_api.py": [
            (r'tools\[tool_type\] = create_openai_tool\(tool_type, get_integration\(\)\s*$', 
             'tools[tool_type] = create_openai_tool(tool_type, get_integration())'),
            (r'return list\(OPENAI_TOOLS\.keys\(\),\s*$', 
             'return list(OPENAI_TOOLS.keys())'),
            (r'return APIResponse\(success=False, error=str\(e\)\s*$',
             'return APIResponse(success=False, error=str(e))'),
            (r'return APIResponse\(success=True, data=response, usage=response\.get\("usage"\)\s*$',
             'return APIResponse(success=True, data=response, usage=response.get("usage"))'),
        ],
        "./src/llamaagent/api/production_app.py": [
            (r'content = json\.loads\(response\.body\.decode\(\)\s*$',
             'content = json.loads(response.body.decode())'),
            (r'app_state\["tools"\]\.register_tool\(CalculatorTool\(\)\s*$',
             'app_state["tools"].register_tool(CalculatorTool())'),
            (r'app_state\["tools"\]\.register_tool\(PythonREPLTool\(\)\s*$',
             'app_state["tools"].register_tool(PythonREPLTool())'),
            (r'file_id = str\(uuid\.uuid4\(\)\s*$',
             'file_id = str(uuid.uuid4())'),
            (r'task_id = str\(uuid\.uuid4\(\)\s*$',
             'task_id = str(uuid.uuid4())'),
            (r'agent_id = str\(uuid\.uuid4\(\)\s*$',
             'agent_id = str(uuid.uuid4())'),
        ],
        "./src/llamaagent/benchmarks/frontier_evaluation.py": [
            (r'timestamp: datetime = field\(default_factory=lambda: datetime\.now\(timezone\.utc\)\s*\n@dataclass',
             'timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))\n\n@dataclass'),
        ],
        "./src/llamaagent/benchmarks/gaia_benchmark.py": [
            (r'pred_num = float\(pred_norm\.replace\("%", ""\)\s*$',
             'pred_num = float(pred_norm.replace("%", ""))'),
            (r'correct_num = float\(correct_norm\.replace\("%", ""\)\s*$',
             'correct_num = float(correct_norm.replace("%", ""))'),
        ],
        "./src/llamaagent/diagnostics/code_analyzer.py": [
            (r'"line_count": len\(content\.splitlines\(\),\s*$',
             '"line_count": len(content.splitlines()),'),
        ],
        "./src/llamaagent/diagnostics/dependency_checker.py": [
            (r'max_len = max\(len\(parts1\), len\(parts2\)\s*$',
             'max_len = max(len(parts1), len(parts2))'),
            (r'parts1\.extend\(\[0\] \* \(max_len - len\(parts1\)\s*$',
             'parts1.extend([0] * (max_len - len(parts1)))'),
            (r'parts2\.extend\(\[0\] \* \(max_len - len\(parts2\)\s*$',
             'parts2.extend([0] * (max_len - len(parts2)))'),
        ],
        "./src/llamaagent/diagnostics/master_diagnostics.py": [
            (r'importlib\.import_module\(dep\.replace\("-", "_"\)\s*$',
             'importlib.import_module(dep.replace("-", "_"))'),
            (r'python_files = list\(self\.project_root\.rglob\("\*\.py"\)\s*$',
             'python_files = list(self.project_root.rglob("*.py"))'),
        ],
        "./src/llamaagent/evaluation/benchmark_engine.py": [
            (r'benchmark_id=f"{benchmark_id}_{model_name}_{int\(started_at\.timestamp\(\)}\",',
             'benchmark_id=f"{benchmark_id}_{model_name}_{int(started_at.timestamp())}",'),
        ],
        "./src/llamaagent/evaluation/golden_dataset.py": [
            (r'created_at: datetime = field\(default_factory=lambda: datetime\.now\(timezone\.utc\)\s*\n@classmethod',
             'created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))\n\n    @classmethod'),
            (r'available_tags = list\(patterns\["tag_patterns"\]\.keys\(\)\s*$',
             'available_tags = list(patterns["tag_patterns"].keys())'),
            (r'random\.sample\(available_tags, min\(3, len\(available_tags\)\s*$',
             'random.sample(available_tags, min(3, len(available_tags)))'),
        ],
        "./src/llamaagent/evaluation/model_comparison.py": [
            (r'generated_at: datetime = field\(default_factory=lambda: datetime\.now\(timezone\.utc\)\s*\ndef to_dict',
             'generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))\n\n    def to_dict'),
            (r'if isinstance\(value, \(int, float\)\s*:',
             'if isinstance(value, (int, float)):'),
            (r'task_counts\.append\(result\.get\(\'total_tasks\', 1\)\s*$',
             'task_counts.append(result.get(\'total_tasks\', 1))'),
            (r'execution_times\.append\(result\.get\(\'avg_execution_time\', 1\.0\)\s*$',
             'execution_times.append(result.get(\'avg_execution_time\', 1.0))'),
            (r'if len\(set\(task_counts\)\s*<\s*2:',
             'if len(set(task_counts)) < 2:'),
            (r'scalability = max\(0\.0, 1\.0 - abs\(correlation\)\s*$',
             'scalability = max(0.0, 1.0 - abs(correlation))'),
        ],
    }
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        
        # Apply specific fixes for this file
        if filepath in specific_fixes:
            for pattern, replacement in specific_fixes[filepath]:
                content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        
        # General fixes
        # Fix missing closing parentheses at end of line
        content = re.sub(r'(\w+)\(([^)]+)\)\s*$', r'\1(\2)', content, flags=re.MULTILINE)
        
        # Fix dataclass field definitions missing closing paren
        content = re.sub(
            r'(field\(default_factory=lambda: [^)]+)\s*\n(\s*)(@\w+|def\s+)',
            r'\1)\n\n\2\3',
            content,
            flags=re.MULTILINE
        )
        
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


# List of files that still have errors
files_to_fix = [
    "./src/llamaagent/api/openai_comprehensive_api.py",
    "./src/llamaagent/api/production_app.py", 
    "./src/llamaagent/benchmarks/frontier_evaluation.py",
    "./src/llamaagent/benchmarks/gaia_benchmark.py",
    "./src/llamaagent/diagnostics/code_analyzer.py",
    "./src/llamaagent/diagnostics/dependency_checker.py",
    "./src/llamaagent/diagnostics/master_diagnostics.py",
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
            if fix_specific_issues(filepath):
                fixed += 1
    
    print(f"\nFixed {fixed} files")
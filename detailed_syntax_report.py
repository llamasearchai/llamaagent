#!/usr/bin/env python3
import os
import ast
import sys
from pathlib import Path

def check_syntax_detailed(file_path):
    """Check if a Python file has valid syntax and return detailed error info."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.splitlines()
        ast.parse(content)
        return True, None, None
    except SyntaxError as e:
        # Get context lines around the error
        context = []
        if e.lineno:
            start = max(0, e.lineno - 3)
            end = min(len(lines), e.lineno + 2)
            for i in range(start, end):
                prefix = ">>> " if i == e.lineno - 1 else "    "
                context.append(f"{prefix}{i+1}: {lines[i] if i < len(lines) else ''}")
        
        error_info = {
            'line': e.lineno,
            'msg': e.msg,
            'text': e.text,
            'offset': e.offset,
            'context': '\n'.join(context)
        }
        return False, error_info, None
    except Exception as e:
        return False, None, str(e)

def main():
    src_dir = Path('src')
    
    # Get files with syntax errors from previous check
    error_files = [
        'src/llamaagent/agents/multimodal_reasoning.py',
        'src/llamaagent/api/openai_comprehensive_api.py',
        'src/llamaagent/api/production_app.py',
        'src/llamaagent/api/shell_endpoints.py',
        'src/llamaagent/api/simon_ecosystem_api.py',
        'src/llamaagent/benchmarks/frontier_evaluation.py',
        'src/llamaagent/benchmarks/gaia_benchmark.py',
        'src/llamaagent/benchmarks/spre_evaluator.py',
        'src/llamaagent/cli/code_generator.py',
        'src/llamaagent/cli/diagnostics_cli.py',
        'src/llamaagent/cli/enhanced_cli.py',
        'src/llamaagent/cli/enhanced_shell_cli.py',
        'src/llamaagent/cli/openai_cli.py',
        'src/llamaagent/cli/role_manager.py',
        'src/llamaagent/data_generation/agentic_pipelines.py',
        'src/llamaagent/data_generation/base.py',
        'src/llamaagent/data_generation/gdt.py',
        'src/llamaagent/diagnostics/code_analyzer.py',
        'src/llamaagent/diagnostics/dependency_checker.py',
        'src/llamaagent/diagnostics/master_diagnostics.py',
        'src/llamaagent/diagnostics/system_validator.py',
        'src/llamaagent/evaluation/benchmark_engine.py',
        'src/llamaagent/evaluation/golden_dataset.py',
        'src/llamaagent/evaluation/model_comparison.py',
        'src/llamaagent/evolution/adaptive_learning.py',
        'src/llamaagent/integration/_openai_stub.py',
        'src/llamaagent/knowledge/knowledge_generator.py',
        'src/llamaagent/ml/inference_engine.py',
        'src/llamaagent/monitoring/advanced_monitoring.py',
        'src/llamaagent/monitoring/alerting.py',
        'src/llamaagent/monitoring/metrics_collector.py',
        'src/llamaagent/monitoring/middleware.py',
        'src/llamaagent/optimization/performance.py',
        'src/llamaagent/orchestration/adaptive_orchestra.py',
        'src/llamaagent/prompting/optimization.py',
        'src/llamaagent/prompting/prompt_templates.py',
        'src/llamaagent/reasoning/chain_engine.py',
        'src/llamaagent/reasoning/memory_manager.py',
        'src/llamaagent/routing/metrics.py',
        'src/llamaagent/routing/provider_registry.py',
        'src/llamaagent/routing/strategies.py',
        'src/llamaagent/routing/task_analyzer.py'
    ]
    
    print("DETAILED PYTHON SYNTAX ERROR REPORT")
    print("=" * 80)
    print(f"Found {len(error_files)} files with syntax errors\n")
    
    for i, file_path in enumerate(error_files[:10], 1):  # Show first 10 in detail
        print(f"\n{i}. {file_path}")
        print("-" * 80)
        
        is_valid, error_info, other_error = check_syntax_detailed(file_path)
        
        if error_info:
            print(f"   Line {error_info['line']}: {error_info['msg']}")
            if error_info['text']:
                print(f"   Problem text: {error_info['text'].strip()}")
            if error_info['offset']:
                print(f"   Error position: column {error_info['offset']}")
            print(f"\n   Context:\n{error_info['context']}")
        elif other_error:
            print(f"   Error: {other_error}")
        
    print("\n" + "=" * 80)
    print("Note: Showing detailed errors for first 10 files only.")
    print("Full list of files with syntax errors:")
    for file_path in error_files:
        print(f"  - {file_path}")

if __name__ == "__main__":
    main()
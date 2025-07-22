#!/usr/bin/env python3
"""Quick syntax check for Python files."""

import os
import py_compile
import sys
from pathlib import Path

def check_file(file_path):
    """Check if a file compiles without syntax errors."""
    try:
        py_compile.compile(file_path, doraise=True)
        return True, None
    except py_compile.PyCompileError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

def main():
    """Check syntax for key files."""
    files_to_check = [
        "src/llamaagent/monitoring/tracing.py",
        "src/llamaagent/monitoring/metrics.py",
        "src/llamaagent/monitoring/health.py",
        "src/llamaagent/llm/providers/cohere.py",
        "src/llamaagent/integration/simon_tools.py",
        "src/llamaagent/integration/_openai_stub.py",
        "src/llamaagent/integration/openai_agents.py",
        "src/llamaagent/integration/openai_agents_complete.py",
        "src/llamaagent/data_generation/spre.py",
        "src/llamaagent/data_generation/agentic_pipelines.py",
        "src/llamaagent/data_generation/gdt.py",
        "src/llamaagent/evaluation/benchmark_engine.py",
        "src/llamaagent/evaluation/golden_dataset.py",
        "src/llamaagent/api/main.py",
        "src/llamaagent/api/production_app.py",
        "src/llamaagent/api/premium_endpoints.py",
        "src/llamaagent/api/shell_endpoints.py",
        "src/llamaagent/agents/reasoning_chains.py",
        "src/llamaagent/agents/multimodal_reasoning.py",
        "src/llamaagent/cli/main.py",
        "src/llamaagent/cli/enhanced_cli.py",
        "src/llamaagent/cli/diagnostics_cli.py",
        "src/llamaagent/cli/interactive.py",
        "src/llamaagent/cli/openai_cli.py",
        "src/llamaagent/cli/llm_cmd.py",
        "src/llamaagent/cli/code_generator.py",
        "src/llamaagent/cli/function_manager.py",
        "src/llamaagent/cli/config_manager.py",
        "src/llamaagent/cli/role_manager.py",
        "src/llamaagent/cli/enhanced_shell_cli.py",
        "src/llamaagent/planning/execution_engine.py",
        "src/llamaagent/planning/strategies.py",
        "src/llamaagent/planning/task_planner.py",
        "src/llamaagent/planning/monitoring.py",
        "src/llamaagent/planning/optimization.py",
        "src/llamaagent/spawning/agent_pool.py",
        "src/llamaagent/spawning/agent_spawner.py",
        "src/llamaagent/spawning/communication.py",
        "src/llamaagent/optimization/performance.py",
        "src/llamaagent/optimization/prompt_optimizer.py",
        "src/llamaagent/security/validator.py",
        "src/llamaagent/security/rate_limiter.py",
        "src/llamaagent/security/manager.py",
        "src/llamaagent/routing/ai_router.py",
        "src/llamaagent/routing/metrics.py",
        "src/llamaagent/routing/strategies.py",
        "src/llamaagent/routing/task_analyzer.py",
        "src/llamaagent/routing/provider_registry.py",
        "src/llamaagent/reasoning/chain_engine.py",
        "src/llamaagent/reasoning/context_sharing.py",
        "src/llamaagent/reasoning/memory_manager.py",
        "src/llamaagent/knowledge/knowledge_generator.py",
        "src/llamaagent/ml/inference_engine.py",
        "src/llamaagent/evolution/adaptive_learning.py",
        "src/llamaagent/evolution/reflection.py",
        "src/llamaagent/evolution/knowledge_base.py",
        "src/llamaagent/benchmarks/frontier_evaluation.py",
        "src/llamaagent/benchmarks/spre_evaluator.py",
        "src/llamaagent/diagnostics/code_analyzer.py",
        "src/llamaagent/diagnostics/dependency_checker.py",
        "src/llamaagent/diagnostics/system_validator.py",
        "src/llamaagent/diagnostics/master_diagnostics.py",
        "src/llamaagent/orchestration/adaptive_orchestra.py",
        "src/llamaagent/cache/advanced_cache.py",
        "src/llamaagent/cache/cache_manager.py",
        "src/llamaagent/cache/llm_cache.py",
        "src/llamaagent/cache/query_optimizer.py",
        "src/llamaagent/cache/result_cache.py",
        "src/llamaagent/prompting/optimization.py",
        "src/llamaagent/prompting/prompt_templates.py",
        "src/llamaagent/prompting/dspy_optimizer.py",
        "src/llamaagent/core/agent.py",
        "src/llamaagent/core/error_handling.py",
        "src/llamaagent/core/message_bus.py",
        "src/llamaagent/core/orchestrator.py",
        "src/llamaagent/core/service_mesh.py",
        "src/llamaagent/llm/simon_ecosystem.py",
        "src/llamaagent/monitoring/advanced_monitoring.py",
        "src/llamaagent/monitoring/alerting.py",
        "src/llamaagent/monitoring/middleware.py",
        "src/llamaagent/monitoring/metrics_collector.py",
        "src/llamaagent/data/gdt.py",
        "src/llamaagent/evaluation/model_comparison.py",
        "src/llamaagent/api/simon_ecosystem_api.py",
        "src/llamaagent/api/openai_comprehensive_api.py",
        "advanced_performance_optimizer.py",
        "clean_fastapi_app.py",
        "comprehensive_fixer.py",
        "complete_spre_demo.py",
        "comprehensive_diagnostic_system.py",
        "fastapi_app.py",
        "demo_complete_system.py",
        "quickstart.py",
        "simple_diagnostic_system.py",
        "production_demo.py",
        "llm/factory.py",
        "tests/test_comprehensive_integration.py",
        "tests/test_gaia_benchmark_comprehensive.py",
        "tests/test_advanced_features.py",
        "examples/spre_usage.py",
        "src/llamaagent/experiment_runner.py",
        "src/llamaagent/types.py",
        "src/llamaagent/visualization.py",
        "src/llamaagent/api.py",
        "src/llamaagent/report_generator.py",
        "src/llamaagent/orchestrator.py",
        "src/llamaagent/statistical_analysis.py",
    ]

    print("Checking syntax for key files...")
    print("=" * 60)
    
    errors = []
    for file_path in files_to_check:
        if os.path.exists(file_path):
            success, error = check_file(file_path)
            if success:
                print(f" {file_path}")
            else:
                print(f" {file_path}: {error}")
                errors.append((file_path, error))
        else:
            print(f"? {file_path} (not found)")

    print("\n" + "=" * 60)
    print(f"Total files checked: {len([f for f in files_to_check if os.path.exists(f)])}")
    print(f"Syntax errors found: {len(errors)}")
    
    if errors:
        print("\nFiles with syntax errors:")
        for file_path, error in errors:
            print(f"  - {file_path}: {error}")
        return 1
    else:
        print("\nAll files have valid syntax!")
        return 0

if __name__ == "__main__":
    sys.exit(main()) 
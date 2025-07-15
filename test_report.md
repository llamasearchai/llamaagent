
# LlamaAgent Test Report

Generated: 2025-06-21 07:35:39
Total Test Time: 7.52 seconds

## Summary

- Core Functionality: PASS
- Integration Tests: FAIL
- Performance: PASS

## Detailed Results

### Import Tests

- success_rate: 1.0
- results: {'llm.factory': True, 'llm.models': True, 'llm.providers.mock_provider': True, 'llm.providers.ollama_provider': True, 'llm.providers.openai_provider': True, 'src.llamaagent.cache': True, 'src.llamaagent.security': True, 'src.llamaagent.monitoring': True, 'src.llamaagent.config.settings': True, 'src.llamaagent.api': True}
- total_modules: 10

### Core Functionality

- success: True

### Unit Tests

- overall_success: True
- suites: {'suite_1': {'success': True, 'command': 'python -m pytest tests/test_basic.py -v --tb=short', 'output': '============================= test session starts ==============================\nplatform darwin -- Python 3.13.4, pytest-8.4.1, pluggy-1.6.0 -- /Users/nemesis/llamaagent/.venv/bin/python\ncachedir: .pytest_cache\nrootdir: /Users/nemesis/llamaagent\nconfigfile: pyproject.toml\nplugins: anyio-4.9.0, cov-6.2.1, asyncio-1.0.0\nasyncio: mode=Mode.AUTO, asyncio_default_fixture_loop_scope=function, asyncio_default_test_loop_scope=function\ncollecting ... collected 15 items\n\ntests/test_basic.py::test_agent_basic_execution PASSED                   [  6%]\ntests/test_basic.py::test_agent_with_tools PASSED                        [ 13%]\ntests/test_basic.py::test_agent_with_memory PASSED                       [ 20%]\ntests/test_basic.py::test_calculator_tool PASSED                         [ 26%]\ntests/test_basic.py::test_python_repl_tool PASSED                        [ 33%]\ntests/test_basic.py::test_memory_operations PASSED                       [ 40%]\ntests/test_basic.py::test_tool_registry PASSED                           [ 46%]\ntests/test_basic.py::test_base_tool_abstract_methods PASSED              [ 53%]\ntests/test_basic.py::test_tool_compatibility_alias PASSED                [ 60%]\ntests/test_basic.py::test_tool_registry_comprehensive PASSED             [ 66%]\ntests/test_basic.py::test_llm_provider PASSED                            [ 73%]\ntests/test_basic.py::test_agent_config PASSED                            [ 80%]\ntests/test_basic.py::test_agent_trace PASSED                             [ 86%]\ntests/test_basic.py::test_memory_entry PASSED                            [ 93%]\ntests/test_basic.py::test_agent_error_handling PASSED                    [100%]\n\n================================ tests coverage ================================\n_______________ coverage: platform darwin, python 3.13.4-final-0 _______________\n\nName                                           Stmts   Miss  Cover   Missing\n----------------------------------------------------------------------------\nsrc/llamaagent/_version.py                         2      0   100%\nsrc/llamaagent/agents/__init__.py                  4      0   100%\nsrc/llamaagent/benchmarks/__init__.py              5      5     0%   1-14\nsrc/llamaagent/benchmarks/baseline_agents.py      60     60     0%   1-151\nsrc/llamaagent/benchmarks/gaia_benchmark.py      188    188     0%   1-554\nsrc/llamaagent/benchmarks/spre_evaluator.py      186    186     0%   1-379\nsrc/llamaagent/cache/__init__.py                 208    208     0%   14-352\nsrc/llamaagent/config/__init__.py                118    118     0%   5-214\nsrc/llamaagent/config/settings.py                263    263     0%   14-428\nsrc/llamaagent/memory/base.py                     14      0   100%\nsrc/llamaagent/monitoring/__init__.py            157    157     0%   14-292\nsrc/llamaagent/security/__init__.py              165    165     0%   14-350\nsrc/llamaagent/storage/__init__.py                 4      0   100%\nsrc/llamaagent/storage/database.py                49     19    61%   36, 58-68, 77, 81-82, 86-87, 91-92, 96-97, 106\nsrc/llamaagent/storage/vector_memory.py           36     20    44%   47-50, 54-56, 59-62, 66-71, 76-79\nsrc/llamaagent/tools/__init__.py                   9      0   100%\nsrc/llamaagent/tools/base.py                      25      0   100%\nsrc/llamaagent/tools/dynamic.py                    0      0   100%\n----------------------------------------------------------------------------\nTOTAL                                           1493   1389     7%\nCoverage HTML written to dir htmlcov\nCoverage XML written to file coverage.xml\n============================== 15 passed in 0.28s ==============================\n'}, 'suite_2': {'success': True, 'command': 'python -m pytest tests/test_llm_providers.py::TestMockProvider -v --tb=short', 'output': '============================= test session starts ==============================\nplatform darwin -- Python 3.13.4, pytest-8.4.1, pluggy-1.6.0 -- /Users/nemesis/llamaagent/.venv/bin/python\ncachedir: .pytest_cache\nrootdir: /Users/nemesis/llamaagent\nconfigfile: pyproject.toml\nplugins: anyio-4.9.0, cov-6.2.1, asyncio-1.0.0\nasyncio: mode=Mode.AUTO, asyncio_default_fixture_loop_scope=function, asyncio_default_test_loop_scope=function\ncollecting ... collected 8 items\n\ntests/test_llm_providers.py::TestMockProvider::test_basic_completion PASSED [ 12%]\ntests/test_llm_providers.py::TestMockProvider::test_math_response_context PASSED [ 25%]\ntests/test_llm_providers.py::TestMockProvider::test_programming_response_context PASSED [ 37%]\ntests/test_llm_providers.py::TestMockProvider::test_planning_response_context PASSED [ 50%]\ntests/test_llm_providers.py::TestMockProvider::test_predefined_responses PASSED [ 62%]\ntests/test_llm_providers.py::TestMockProvider::test_failure_simulation PASSED [ 75%]\ntests/test_llm_providers.py::TestMockProvider::test_call_counting PASSED [ 87%]\ntests/test_llm_providers.py::TestMockProvider::test_health_check PASSED  [100%]\n\n================================ tests coverage ================================\n_______________ coverage: platform darwin, python 3.13.4-final-0 _______________\n\nName                                           Stmts   Miss  Cover   Missing\n----------------------------------------------------------------------------\nsrc/llamaagent/_version.py                         2      2     0%   4-5\nsrc/llamaagent/agents/__init__.py                  4      4     0%   8-18\nsrc/llamaagent/benchmarks/__init__.py              5      5     0%   1-14\nsrc/llamaagent/benchmarks/baseline_agents.py      60     60     0%   1-151\nsrc/llamaagent/benchmarks/gaia_benchmark.py      188    188     0%   1-554\nsrc/llamaagent/benchmarks/spre_evaluator.py      186    186     0%   1-379\nsrc/llamaagent/cache/__init__.py                 208    208     0%   14-352\nsrc/llamaagent/config/__init__.py                118    118     0%   5-214\nsrc/llamaagent/config/settings.py                263    263     0%   14-428\nsrc/llamaagent/memory/base.py                     14     14     0%   1-23\nsrc/llamaagent/monitoring/__init__.py            157    157     0%   14-292\nsrc/llamaagent/security/__init__.py              165    165     0%   14-350\nsrc/llamaagent/storage/__init__.py                 4      4     0%   1-17\nsrc/llamaagent/storage/database.py                49     49     0%   1-106\nsrc/llamaagent/storage/vector_memory.py           36     36     0%   1-79\nsrc/llamaagent/tools/__init__.py                   9      9     0%   3-23\nsrc/llamaagent/tools/base.py                      25     25     0%   3-77\nsrc/llamaagent/tools/dynamic.py                    0      0   100%\n----------------------------------------------------------------------------\nTOTAL                                           1493   1493     0%\nCoverage HTML written to dir htmlcov\nCoverage XML written to file coverage.xml\n============================== 8 passed in 2.55s ===============================\n'}}
- total_suites: 2

### Integration Tests

- success: False
- partial_success: False

### Code Quality

- syntax_check: True
- type_check: False

### Performance

- success: True
- data: {'total_requests': 10, 'total_time': 0.4721388816833496, 'avg_time': 0.04721388816833496, 'throughput': 21.180208595289386}


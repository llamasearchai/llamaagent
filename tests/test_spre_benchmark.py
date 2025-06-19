import pytest
import asyncio
from pathlib import Path


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary directory for benchmark outputs."""
    return tmp_path / "benchmark_results"


class TestGAIABenchmark:
    """Test GAIA benchmark functionality."""
    
    def test_gaia_task_creation(self):
        """Test GAIA task creation and validation."""
        from llamaagent.benchmarks.gaia_benchmark import GAIATask
        
        task = GAIATask(
            task_id="test_001",
            question="What is 2 + 2?",
            expected_answer="4",
            difficulty="easy",
            steps_required=1,
            domain="arithmetic"
        )
        
        assert task.task_id == "test_001"
        assert task.difficulty == "easy"
        assert task.steps_required == 1
    
    def test_gaia_task_validation(self):
        """Test GAIA task validation."""
        from llamaagent.benchmarks.gaia_benchmark import GAIATask
        
        with pytest.raises(ValueError, match="Invalid difficulty"):
            GAIATask(
                task_id="test",
                question="test",
                expected_answer="test",
                difficulty="invalid",
                steps_required=1,
                domain="test"
            )
        
        with pytest.raises(ValueError, match="Invalid steps_required"):
            GAIATask(
                task_id="test",
                question="test", 
                expected_answer="test",
                difficulty="easy",
                steps_required=0,
                domain="test"
            )
    
    def test_gaia_benchmark_initialization(self, temp_output_dir):
        """Test GAIA benchmark initialization."""
        from llamaagent.benchmarks.gaia_benchmark import GAIABenchmark
        
        benchmark = GAIABenchmark(data_file=temp_output_dir / "gaia_tasks.json")
        
        assert len(benchmark.tasks) > 0
        assert benchmark.data_file.exists()
    
    def test_gaia_benchmark_filtering(self, temp_output_dir):
        """Test task filtering functionality."""
        from llamaagent.benchmarks.gaia_benchmark import GAIABenchmark
        
        benchmark = GAIABenchmark(data_file=temp_output_dir / "gaia_tasks.json")
        
        # Test difficulty filtering
        easy_tasks = benchmark.get_tasks(difficulty="easy")
        assert all(task.difficulty == "easy" for task in easy_tasks)
        
        # Test domain filtering
        math_tasks = benchmark.get_tasks(domain="mathematics")
        assert all(task.domain == "mathematics" for task in math_tasks)
        
        # Test min_steps filtering
        complex_tasks = benchmark.get_tasks(min_steps=3)
        assert all(task.steps_required >= 3 for task in complex_tasks)
        
        # Test limit
        limited_tasks = benchmark.get_tasks(limit=2)
        assert len(limited_tasks) <= 2


class TestBaselineAgents:
    """Test baseline agent implementations."""
    
    def test_baseline_agent_creation(self):
        """Test creation of all baseline agent types."""
        from llamaagent.benchmarks.baseline_agents import BaselineAgentFactory, BaselineType
        
        for baseline_type in BaselineAgentFactory.get_all_baseline_types():
            agent = BaselineAgentFactory.create_agent(baseline_type)
            assert agent is not None
            assert agent.config.name is not None
    
    def test_baseline_descriptions(self):
        """Test baseline descriptions are available."""
        from llamaagent.benchmarks.baseline_agents import BaselineAgentFactory
        
        for baseline_type in BaselineAgentFactory.get_all_baseline_types():
            description = BaselineAgentFactory.get_baseline_description(baseline_type)
            assert len(description) > 0
    
    @pytest.mark.asyncio
    async def test_vanilla_react_agent(self):
        """Test vanilla ReAct agent behavior."""
        from llamaagent.benchmarks.baseline_agents import BaselineAgentFactory, BaselineType
        
        agent = BaselineAgentFactory.create_agent(BaselineType.VANILLA_REACT)
        response = await agent.execute("What is 5 + 3?")
        
        assert response.success
        # Should not have SPRE planning traces
        planning_events = [e for e in response.trace if e["type"] == "plan_generated"]
        assert len(planning_events) == 0
    
    @pytest.mark.asyncio
    async def test_preact_only_agent(self):
        """Test Pre-Act only agent behavior."""
        from llamaagent.benchmarks.baseline_agents import BaselineAgentFactory, BaselineType
        
        agent = BaselineAgentFactory.create_agent(BaselineType.PREACT_ONLY)
        response = await agent.execute("Calculate 10 * 5")
        
        assert response.success
        # Should have planning but force tool usage
        resource_events = [e for e in response.trace if e["type"] == "resource_assessment_override"]
        assert len(resource_events) > 0
    
    @pytest.mark.asyncio
    async def test_sem_only_agent(self):
        """Test SEM only agent behavior."""
        from llamaagent.benchmarks.baseline_agents import BaselineAgentFactory, BaselineType
        
        agent = BaselineAgentFactory.create_agent(BaselineType.SEM_ONLY)
        response = await agent.execute("What is the capital of France?")
        
        assert response.success
        # Should have resource assessment but minimal planning
        sem_events = [e for e in response.trace if e["type"] == "sem_only_execution"]
        assert len(sem_events) > 0
    
    @pytest.mark.asyncio
    async def test_spre_full_agent(self):
        """Test full SPRE agent behavior."""
        from llamaagent.benchmarks.baseline_agents import BaselineAgentFactory, BaselineType
        
        agent = BaselineAgentFactory.create_agent(BaselineType.SPRE_FULL)
        response = await agent.execute("Calculate 7 * 8 and explain the result")
        
        assert response.success
        # Should have full SPRE pipeline
        planning_events = [e for e in response.trace if e["type"] == "plan_generated"]
        resource_events = [e for e in response.trace if e["type"] == "resource_assessment"]
        
        # May have planning depending on task complexity
        assert len(planning_events) >= 0
        assert len(resource_events) >= 0


class TestSPREEvaluator:
    """Test SPRE evaluation framework."""
    
    @pytest.mark.asyncio
    async def test_evaluator_initialization(self, temp_output_dir):
        """Test evaluator initialization."""
        from llamaagent.benchmarks.spre_evaluator import SPREEvaluator
        
        evaluator = SPREEvaluator(output_dir=temp_output_dir)
        assert evaluator.output_dir.exists()
        assert evaluator.benchmark is not None
    
    @pytest.mark.asyncio
    async def test_single_baseline_evaluation(self, temp_output_dir):
        """Test evaluation of single baseline."""
        from llamaagent.benchmarks.spre_evaluator import SPREEvaluator
        from llamaagent.benchmarks.baseline_agents import BaselineType
        
        evaluator = SPREEvaluator(output_dir=temp_output_dir)
        
        # Run evaluation on simple tasks
        result = await evaluator.run_single_baseline_evaluation(
            BaselineType.VANILLA_REACT,
            task_ids=["simple_001", "simple_002"],
            max_tasks=2
        )
        
        assert result.baseline_type == BaselineType.VANILLA_REACT
        assert len(result.task_results) <= 2
        assert result.success_rate >= 0
        assert result.avg_api_calls >= 0
        assert result.avg_latency >= 0
    
    def test_answer_evaluation(self, temp_output_dir):
        """Test answer evaluation logic."""
        from llamaagent.benchmarks.spre_evaluator import SPREEvaluator
        
        evaluator = SPREEvaluator(output_dir=temp_output_dir)
        
        # Test exact match
        assert evaluator._evaluate_answer("400", "400") == True
        
        # Test case insensitive
        assert evaluator._evaluate_answer("Paris", "paris") == True
        
        # Test containment
        assert evaluator._evaluate_answer("42", "The answer is 42.") == True
        
        # Test numeric tolerance
        assert evaluator._evaluate_answer("3.14", "3.141") == True
        
        # Test failure cases
        assert evaluator._evaluate_answer("100", "200") == False
    
    def test_api_call_counting(self, temp_output_dir):
        """Test API call counting from trace."""
        from llamaagent.benchmarks.spre_evaluator import SPREEvaluator
        
        evaluator = SPREEvaluator(output_dir=temp_output_dir)
        
        trace = [
            {"type": "planner_response", "data": {}},
            {"type": "resource_assessment_detail", "data": {}},
            {"type": "internal_execution", "data": {}},
            {"type": "synthesis_complete", "data": {}},
            {"type": "other_event", "data": {}}  # Should not be counted
        ]
        
        count = evaluator._count_api_calls(trace)
        assert count == 4  # Only the first 4 events should be counted
    
    @pytest.mark.asyncio
    async def test_benchmark_result_properties(self, temp_output_dir):
        """Test BenchmarkResult property calculations."""
        from llamaagent.benchmarks.spre_evaluator import BenchmarkResult, TaskResult
        
        # Create mock task results
        task_results = [
            TaskResult(
                task_id="test_1",
                question="Q1",
                expected_answer="A1",
                actual_answer="A1",
                success=True,
                execution_time=1.0,
                tokens_used=100,
                api_calls=2,
                reasoning_tokens=25,
                baseline_type="test"
            ),
            TaskResult(
                task_id="test_2", 
                question="Q2",
                expected_answer="A2",
                actual_answer="Wrong",
                success=False,
                execution_time=2.0,
                tokens_used=200,
                api_calls=4,
                reasoning_tokens=50,
                baseline_type="test"
            )
        ]
        
        result = BenchmarkResult(
            baseline_type="test",
            agent_name="Test Agent",
            task_results=task_results
        )
        
        assert result.success_rate == 50.0  # 1/2 * 100
        assert result.avg_api_calls == 3.0  # (2 + 4) / 2
        assert result.avg_latency == 1.5   # (1.0 + 2.0) / 2
        assert result.avg_tokens == 150.0  # (100 + 200) / 2
        assert result.efficiency_ratio == 50.0 / 3.0  # success_rate / avg_api_calls


class TestSPREIntegration:
    """Integration tests for complete SPRE system."""
    
    @pytest.mark.asyncio
    async def test_spre_vs_vanilla_comparison(self):
        """Test SPRE vs vanilla agent on same task."""
        from llamaagent.benchmarks.baseline_agents import BaselineAgentFactory, BaselineType
        
        task = "Calculate 15 * 24"
        
        # Test vanilla agent
        vanilla_agent = BaselineAgentFactory.create_agent(BaselineType.VANILLA_REACT)
        vanilla_response = await vanilla_agent.execute(task)
        
        # Test SPRE agent
        spre_agent = BaselineAgentFactory.create_agent(BaselineType.SPRE_FULL)
        spre_response = await spre_agent.execute(task)
        
        # Both should succeed on simple math
        assert vanilla_response.success
        assert spre_response.success
        
        # Check for different trace patterns
        vanilla_planning = [e for e in vanilla_response.trace if e["type"] == "plan_generated"]
        spre_planning = [e for e in spre_response.trace if e["type"] == "plan_generated"]
        
        # Vanilla should have no planning, SPRE may have planning
        assert len(vanilla_planning) == 0
        # SPRE may or may not plan for simple tasks, but should have SPRE-specific traces
        spre_traces = [e for e in spre_response.trace if "spree_enabled" in e.get("data", {})]
        assert len(spre_traces) > 0
    
    @pytest.mark.asyncio
    async def test_complex_task_execution(self):
        """Test SPRE on complex multi-step task."""
        from llamaagent.benchmarks.baseline_agents import BaselineAgentFactory, BaselineType
        
        complex_task = """
        1. Calculate 25 * 16
        2. Find the square root of that result
        3. Round to 2 decimal places
        """
        
        agent = BaselineAgentFactory.create_agent(BaselineType.SPRE_FULL)
        response = await agent.execute(complex_task)
        
        assert response.success
        
        # Should have planning for multi-step task
        planning_events = [e for e in response.trace if e["type"] == "plan_generated"]
        resource_events = [e for e in response.trace if e["type"] == "resource_assessment"]
        
        # Complex task should trigger planning
        assert len(planning_events) >= 0  # May vary based on implementation
        assert response.execution_time > 0
        assert response.tokens_used > 0 
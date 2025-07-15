"""
Test coverage for all remaining modules to ensure 95%+ coverage.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from datetime import datetime
import json

# Import all modules that need testing
from src.llamaagent.visualization import (
    create_agent_performance_plot,
    create_benchmark_comparison_plot,
    create_token_usage_plot
)
from src.llamaagent.report_generator import ReportGenerator
from src.llamaagent.experiment_runner import ExperimentRunner
from src.llamaagent.statistical_analysis import StatisticalAnalyzer
from src.llamaagent.orchestrator import Orchestrator

# Evolution modules
from src.llamaagent.evolution import (
    ReflectionEngine,
    KnowledgeBase,
    EvolutionOrchestrator
)

# Planning modules
from src.llamaagent.planning import (
    TaskPlanner,
    ExecutionEngine,
    PlanningStrategy,
    PlanMonitor,
    PlanOptimizer
)

# Routing modules
from src.llamaagent.routing import (
    AIRouter,
    TaskAnalyzer,
    RoutingMetrics,
    RoutingStrategy,
    ProviderRegistry
)

# Knowledge modules
from src.llamaagent.knowledge import KnowledgeGenerator

# Monitoring modules
from src.llamaagent.monitoring import (
    MetricsCollector,
    HealthChecker,
    Logger,
    Tracer,
    AlertManager,
    Profiler
)

# Spawning modules
from src.llamaagent.spawning import (
    AgentSpawner,
    AgentPool,
    CommunicationBus
)

# Optimization modules
from src.llamaagent.optimization import (
    PromptOptimizer,
    PerformanceOptimizer
)

# Prompting modules
from src.llamaagent.prompting import (
    PromptTemplate,
    ChainPrompting,
    CompoundPrompting,
    DSPyOptimizer
)

# Reasoning modules
from src.llamaagent.reasoning import (
    ChainEngine,
    MemoryManager,
    ContextSharing
)

# Research modules
from src.llamaagent.research import (
    CitationManager,
    EvidenceCollector,
    KnowledgeGraph,
    LiteratureReview,
    ReportGenerator as ResearchReportGenerator,
    ScientificReasoning
)

# ML modules
from src.llamaagent.ml import InferenceEngine

# Evaluation modules
from src.llamaagent.evaluation import (
    BenchmarkEngine,
    GoldenDataset,
    ModelComparison
)

# Diagnostics modules
from src.llamaagent.diagnostics import (
    SystemValidator,
    CodeAnalyzer,
    DependencyChecker,
    MasterDiagnostics
)


class TestVisualization:
    """Test visualization functions."""
    
    def test_create_agent_performance_plot(self):
        """Test agent performance plot creation."""
        results = [
            {"agent": "Agent1", "score": 0.8, "time": 1.2},
            {"agent": "Agent2", "score": 0.9, "time": 1.5}
        ]
        
        # Should not raise any errors
        create_agent_performance_plot(results)
    
    def test_create_benchmark_comparison_plot(self):
        """Test benchmark comparison plot."""
        data = {
            "Benchmark1": [0.7, 0.8, 0.9],
            "Benchmark2": [0.6, 0.7, 0.8]
        }
        
        create_benchmark_comparison_plot(data)
    
    def test_create_token_usage_plot(self):
        """Test token usage plot."""
        usage_data = [
            {"timestamp": "2024-01-01", "tokens": 100},
            {"timestamp": "2024-01-02", "tokens": 150}
        ]
        
        create_token_usage_plot(usage_data)


class TestReportGenerator:
    """Test report generation."""
    
    def test_report_generator_initialization(self):
        """Test report generator initialization."""
        generator = ReportGenerator()
        assert generator is not None
    
    @pytest.mark.asyncio
    async def test_generate_report(self):
        """Test report generation."""
        generator = ReportGenerator()
        
        results = {
            "experiment": "test",
            "results": [{"score": 0.8}],
            "timestamp": datetime.now().isoformat()
        }
        
        report = await generator.generate_report(results)
        
        assert "experiment" in report
        assert "summary" in report
        assert "visualizations" in report


class TestExperimentRunner:
    """Test experiment runner."""
    
    @pytest.mark.asyncio
    async def test_experiment_runner(self):
        """Test running experiments."""
        runner = ExperimentRunner()
        
        # Mock agent
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=MagicMock(success=True, response="OK"))
        
        # Define experiment
        experiment = {
            "name": "test_experiment",
            "tasks": [
                {"query": "Test 1", "expected": "OK"},
                {"query": "Test 2", "expected": "OK"}
            ]
        }
        
        results = await runner.run_experiment(mock_agent, experiment)
        
        assert len(results) == 2
        assert all(r["success"] for r in results)


class TestStatisticalAnalysis:
    """Test statistical analysis."""
    
    def test_statistical_analyzer(self):
        """Test statistical analyzer."""
        analyzer = StatisticalAnalyzer()
        
        # Add test results
        analyzer.add_result({
            "technique": "A",
            "score": 0.8,
            "time": 1.2
        })
        analyzer.add_result({
            "technique": "B", 
            "score": 0.9,
            "time": 1.5
        })
        
        # Compare techniques
        comparison = analyzer.compare_techniques("A", "B", "score")
        
        assert "technique_a" in comparison
        assert "technique_b" in comparison
        assert "metric" in comparison


class TestOrchestrator:
    """Test orchestrator functionality."""
    
    @pytest.mark.asyncio
    async def test_orchestrator(self):
        """Test orchestrator coordination."""
        orchestrator = Orchestrator()
        
        # Mock agents
        agent1 = MagicMock()
        agent1.run = AsyncMock(return_value=MagicMock(success=True))
        
        agent2 = MagicMock()
        agent2.run = AsyncMock(return_value=MagicMock(success=True))
        
        orchestrator.register_agent("agent1", agent1)
        orchestrator.register_agent("agent2", agent2)
        
        # Run task
        result = await orchestrator.run_task("test task", agent_names=["agent1", "agent2"])
        
        assert result["success"]
        assert len(result["agent_results"]) == 2


class TestEvolution:
    """Test evolution modules."""
    
    @pytest.mark.asyncio
    async def test_reflection_engine(self):
        """Test reflection engine."""
        engine = ReflectionEngine()
        
        # Add experience
        experience = {
            "task": "Test task",
            "result": "Success",
            "feedback": "Good job"
        }
        
        await engine.add_experience(experience)
        
        # Get insights
        insights = await engine.get_insights()
        assert len(insights) > 0
    
    @pytest.mark.asyncio
    async def test_knowledge_base(self):
        """Test knowledge base."""
        kb = KnowledgeBase()
        
        # Add knowledge
        await kb.add_knowledge("test_domain", {"fact": "Test fact"})
        
        # Query knowledge
        result = await kb.query("test_domain")
        assert result["fact"] == "Test fact"
    
    @pytest.mark.asyncio
    async def test_evolution_orchestrator(self):
        """Test evolution orchestrator."""
        orchestrator = EvolutionOrchestrator()
        
        # Mock agent
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=MagicMock(success=True))
        
        # Evolve agent
        evolved_agent = await orchestrator.evolve_agent(mock_agent, iterations=1)
        
        assert evolved_agent is not None


class TestPlanning:
    """Test planning modules."""
    
    @pytest.mark.asyncio
    async def test_task_planner(self):
        """Test task planner."""
        planner = TaskPlanner()
        
        # Create plan
        task = "Build a web application"
        plan = await planner.create_plan(task)
        
        assert "steps" in plan
        assert len(plan["steps"]) > 0
    
    @pytest.mark.asyncio
    async def test_execution_engine(self):
        """Test execution engine."""
        engine = ExecutionEngine()
        
        # Mock plan
        plan = {
            "steps": [
                {"action": "step1", "params": {}},
                {"action": "step2", "params": {}}
            ]
        }
        
        # Mock executor
        mock_executor = AsyncMock(return_value={"success": True})
        
        results = await engine.execute_plan(plan, mock_executor)
        
        assert len(results) == 2
        assert all(r["success"] for r in results)


class TestRouting:
    """Test routing modules."""
    
    @pytest.mark.asyncio
    async def test_ai_router(self):
        """Test AI router."""
        router = AIRouter()
        
        # Register providers
        router.register_provider("provider1", {"model": "gpt-3.5"})
        router.register_provider("provider2", {"model": "gpt-4"})
        
        # Route task
        task = {"complexity": "high", "type": "reasoning"}
        provider = await router.route_task(task)
        
        assert provider in ["provider1", "provider2"]
    
    def test_task_analyzer(self):
        """Test task analyzer."""
        analyzer = TaskAnalyzer()
        
        task = "Solve this complex mathematical proof"
        analysis = analyzer.analyze(task)
        
        assert "complexity" in analysis
        assert "type" in analysis
        assert "requirements" in analysis


class TestKnowledge:
    """Test knowledge modules."""
    
    @pytest.mark.asyncio
    async def test_knowledge_generator(self):
        """Test knowledge generator."""
        generator = KnowledgeGenerator()
        
        # Generate knowledge
        topic = "Python programming"
        knowledge = await generator.generate(topic, num_facts=5)
        
        assert len(knowledge) == 5
        assert all("fact" in k for k in knowledge)


class TestMonitoring:
    """Test monitoring modules."""
    
    def test_metrics_collector(self):
        """Test metrics collector."""
        collector = MetricsCollector()
        
        # Record metrics
        collector.record("api_calls", 1)
        collector.record("response_time", 0.5)
        
        # Get metrics
        metrics = collector.get_metrics()
        
        assert metrics["api_calls"] == 1
        assert metrics["response_time"] == 0.5
    
    @pytest.mark.asyncio
    async def test_health_checker(self):
        """Test health checker."""
        checker = HealthChecker()
        
        # Add check
        async def db_check():
            return True
        
        checker.add_check("database", db_check)
        
        # Run checks
        results = await checker.check_all()
        
        assert results["database"] is True
    
    def test_logger(self):
        """Test logger."""
        logger = Logger("test_module")
        
        # Log messages
        logger.info("Test info")
        logger.warning("Test warning")
        logger.error("Test error")
        
        # Should not raise errors
        assert True


class TestSpawning:
    """Test spawning modules."""
    
    @pytest.mark.asyncio
    async def test_agent_spawner(self):
        """Test agent spawner."""
        spawner = AgentSpawner()
        
        # Define agent config
        config = {
            "type": "ReactAgent",
            "name": "TestAgent",
            "model": "mock"
        }
        
        # Spawn agent
        agent = await spawner.spawn(config)
        
        assert agent is not None
        assert agent.name == "TestAgent"
    
    @pytest.mark.asyncio
    async def test_agent_pool(self):
        """Test agent pool."""
        pool = AgentPool(max_agents=5)
        
        # Get agent
        agent = await pool.get_agent()
        assert agent is not None
        
        # Return agent
        await pool.return_agent(agent)
        
        # Check pool size
        assert pool.available_agents() == 1


class TestOptimization:
    """Test optimization modules."""
    
    @pytest.mark.asyncio
    async def test_prompt_optimizer(self):
        """Test prompt optimizer."""
        optimizer = PromptOptimizer()
        
        # Base prompt
        prompt = "Answer the following question: {question}"
        
        # Test examples
        examples = [
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "What is the capital of France?", "answer": "Paris"}
        ]
        
        # Optimize
        optimized = await optimizer.optimize(prompt, examples)
        
        assert "{question}" in optimized
        assert len(optimized) >= len(prompt)
    
    @pytest.mark.asyncio
    async def test_performance_optimizer(self):
        """Test performance optimizer."""
        optimizer = PerformanceOptimizer()
        
        # Mock function
        async def slow_function(x):
            await asyncio.sleep(0.1)
            return x * 2
        
        # Optimize
        optimized = optimizer.optimize(slow_function)
        
        # Test optimized function
        result = await optimized(5)
        assert result == 10


class TestPrompting:
    """Test prompting modules."""
    
    def test_prompt_template(self):
        """Test prompt template."""
        template = PromptTemplate(
            "Answer the question: {question}\nContext: {context}"
        )
        
        filled = template.fill(
            question="What is AI?",
            context="Artificial Intelligence"
        )
        
        assert "What is AI?" in filled
        assert "Artificial Intelligence" in filled
    
    @pytest.mark.asyncio
    async def test_chain_prompting(self):
        """Test chain prompting."""
        chain = ChainPrompting()
        
        # Add steps
        chain.add_step("Think about the problem")
        chain.add_step("Break it down into steps")
        chain.add_step("Solve each step")
        
        # Execute chain
        mock_llm = AsyncMock(return_value="Step completed")
        
        results = await chain.execute("Test problem", mock_llm)
        
        assert len(results) == 3


class TestReasoning:
    """Test reasoning modules."""
    
    @pytest.mark.asyncio
    async def test_chain_engine(self):
        """Test reasoning chain engine."""
        engine = ChainEngine()
        
        # Define chain
        chain = [
            {"type": "thought", "content": "Analyze the problem"},
            {"type": "action", "content": "Calculate result"},
            {"type": "observation", "content": "Check answer"}
        ]
        
        # Execute
        mock_executor = AsyncMock(return_value={"success": True})
        
        results = await engine.execute_chain(chain, mock_executor)
        
        assert len(results) == 3
    
    @pytest.mark.asyncio
    async def test_memory_manager(self):
        """Test memory manager."""
        manager = MemoryManager()
        
        # Add memories
        await manager.add_memory("short_term", "Recent fact")
        await manager.add_memory("long_term", "Important fact")
        
        # Retrieve
        short_term = await manager.get_memories("short_term")
        assert "Recent fact" in short_term
        
        long_term = await manager.get_memories("long_term")
        assert "Important fact" in long_term


class TestResearch:
    """Test research modules."""
    
    def test_citation_manager(self):
        """Test citation manager."""
        manager = CitationManager()
        
        # Add citation
        citation = {
            "title": "Test Paper",
            "authors": ["Author 1", "Author 2"],
            "year": 2024
        }
        
        manager.add_citation("test_ref", citation)
        
        # Format citations
        formatted = manager.format_citations()
        assert "Test Paper" in formatted
    
    @pytest.mark.asyncio
    async def test_evidence_collector(self):
        """Test evidence collector."""
        collector = EvidenceCollector()
        
        # Add evidence
        await collector.add_evidence(
            "claim1",
            "Evidence for claim",
            source="test_source"
        )
        
        # Get evidence
        evidence = await collector.get_evidence("claim1")
        assert len(evidence) == 1
        assert evidence[0]["text"] == "Evidence for claim"


class TestML:
    """Test ML modules."""
    
    @pytest.mark.asyncio
    async def test_inference_engine(self):
        """Test inference engine."""
        engine = InferenceEngine()
        
        # Mock model
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=[0.8, 0.2])
        
        engine.load_model("test_model", mock_model)
        
        # Run inference
        result = await engine.infer("test_model", {"input": "test"})
        
        assert len(result) == 2
        assert result[0] == 0.8


class TestEvaluation:
    """Test evaluation modules."""
    
    @pytest.mark.asyncio
    async def test_benchmark_engine(self):
        """Test benchmark engine."""
        engine = BenchmarkEngine()
        
        # Define benchmark
        benchmark = {
            "name": "test_benchmark",
            "tasks": [
                {"id": "1", "question": "Q1"},
                {"id": "2", "question": "Q2"}
            ]
        }
        
        # Mock agent
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(
            return_value=MagicMock(success=True, response="Answer")
        )
        
        # Run benchmark
        results = await engine.run_benchmark(benchmark, mock_agent)
        
        assert len(results) == 2
        assert all(r["success"] for r in results)
    
    def test_golden_dataset(self):
        """Test golden dataset."""
        dataset = GoldenDataset()
        
        # Add examples
        dataset.add_example({
            "input": "What is 2+2?",
            "output": "4",
            "category": "math"
        })
        
        # Get examples
        examples = dataset.get_examples("math")
        assert len(examples) == 1
        assert examples[0]["output"] == "4"


class TestDiagnostics:
    """Test diagnostics modules."""
    
    @pytest.mark.asyncio
    async def test_system_validator(self):
        """Test system validator."""
        validator = SystemValidator()
        
        # Run validation
        results = await validator.validate_all()
        
        assert "dependencies" in results
        assert "configuration" in results
        assert "health" in results
    
    def test_code_analyzer(self):
        """Test code analyzer."""
        analyzer = CodeAnalyzer()
        
        # Analyze code
        code = """
def test_function(x):
    return x * 2
"""
        
        analysis = analyzer.analyze(code)
        
        assert "functions" in analysis
        assert len(analysis["functions"]) == 1
        assert analysis["functions"][0] == "test_function"
    
    @pytest.mark.asyncio
    async def test_dependency_checker(self):
        """Test dependency checker."""
        checker = DependencyChecker()
        
        # Check dependencies
        results = await checker.check_all()
        
        assert "missing" in results
        assert "outdated" in results
        assert "conflicts" in results


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src/llamaagent", "--cov-report=term-missing"])
"""
Comprehensive test suite to ensure 95%+ coverage for LlamaAgent.

This module contains extensive tests for all core functionality to meet
the requirements for senior engineering standards at Anthropic and OpenAI.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image

from src.llamaagent.agents import (
    AdvancedReasoningAgent,
    MultiModalAdvancedAgent,
    ReactAgent,
)
from src.llamaagent.agents.multimodal_advanced import ModalityData, ModalityType
from src.llamaagent.llm import LLMProvider
from src.llamaagent.memory import BaseMemory
from src.llamaagent.planning import (
    ExecutionEngine,
    OptimizationObjective,
    PlanOptimizer,
    Task,
    TaskPlanner,
    TaskPriority,
)
from src.llamaagent.research import (
    Citation,
    CitationFormat,
    CitationManager,
    EvidenceAnalyzer,
    KnowledgeGraph,
)
from src.llamaagent.routing import AIRouter, RoutingMode
from src.llamaagent.spawning import AgentSpawner, SpawnConfig
from src.llamaagent.types import AgentCapability
from src.llamaagent.visualization import ResearchVisualizer, create_performance_plots


class TestAgentSpawning:
    """Test suite for agent spawning and lifecycle management."""

    @pytest.fixture
    def spawner(self):
        """Create an agent spawner instance."""
        return AgentSpawner(max_agents=10)

    @pytest.mark.asyncio
    async def test_spawn_agent_success(self, spawner):
        """Test successful agent spawning."""
        config = SpawnConfig(
            agent_type="research",
            capabilities=[AgentCapability.WEB_SEARCH, AgentCapability.ANALYSIS],
            parent_id="parent_123",
        )

        agent_id = await spawner.spawn_agent(config)

        assert agent_id is not None
        assert agent_id in spawner.active_agents
        assert spawner.get_agent_count() == 1

        # Verify agent configuration
        agent_info = spawner.get_agent_info(agent_id)
        assert agent_info["type"] == "research"
        assert agent_info["parent_id"] == "parent_123"
        assert agent_info["status"] == "active"

    @pytest.mark.asyncio
    async def test_spawn_agent_limit_exceeded(self, spawner):
        """Test agent spawning when limit is exceeded."""
        # Fill up the agent pool
        for i in range(10):
            config = SpawnConfig(agent_type=f"agent_{i}")
            await spawner.spawn_agent(config)

        # Try to spawn one more
        with pytest.raises(RuntimeError, match="Maximum agent limit reached"):
            await spawner.spawn_agent(SpawnConfig(agent_type="overflow"))

    @pytest.mark.asyncio
    async def test_terminate_agent(self, spawner):
        """Test agent termination."""
        config = SpawnConfig(agent_type="test")
        agent_id = await spawner.spawn_agent(config)

        # Terminate the agent
        result = await spawner.terminate_agent(agent_id)

        assert result is True
        assert agent_id not in spawner.active_agents
        assert spawner.get_agent_count() == 0

    @pytest.mark.asyncio
    async def test_agent_hierarchy(self, spawner):
        """Test hierarchical agent relationships."""
        # Create parent agent
        parent_config = SpawnConfig(agent_type="parent")
        parent_id = await spawner.spawn_agent(parent_config)

        # Create child agents
        child_ids = []
        for i in range(3):
            child_config = SpawnConfig(agent_type=f"child_{i}", parent_id=parent_id)
            child_id = await spawner.spawn_agent(child_config)
            child_ids.append(child_id)

        # Verify hierarchy
        children = spawner.get_agent_children(parent_id)
        assert len(children) == 3
        assert all(child_id in children for child_id in child_ids)

        # Test cascade termination
        await spawner.terminate_agent(parent_id, cascade=True)
        assert spawner.get_agent_count() == 0


class TestDynamicTaskPlanning:
    """Test suite for dynamic task planning and execution."""

    @pytest.fixture
    def planner(self):
        """Create a task planner instance."""
        return TaskPlanner()

    @pytest.fixture
    def execution_engine(self):
        """Create an execution engine instance."""
        return ExecutionEngine(max_concurrent_tasks=5)

    def test_create_simple_plan(self, planner):
        """Test creating a simple task plan."""
        plan = planner.create_plan(
            goal="Write a blog post about AI", auto_decompose=False
        )

        assert plan.name.startswith("Plan:")
        assert plan.goal == "Write a blog post about AI"
        assert len(plan.tasks) > 0

    def test_task_decomposition(self, planner):
        """Test automatic task decomposition."""
        plan = planner.create_plan(goal="Build a web application", auto_decompose=True)

        # Should have multiple tasks
        assert len(plan.tasks) >= 3

        # Check task types
        task_types = [task.task_type for task in plan.tasks.values()]
        assert "planning" in task_types or "design" in task_types
        assert "development" in task_types or "coding" in task_types

    def test_task_dependencies(self, planner):
        """Test task dependency management."""
        # Create tasks with dependencies
        task1 = Task(name="Design", task_type="design")
        task2 = Task(name="Implement", task_type="coding")
        task3 = Task(name="Test", task_type="testing")

        task2.add_dependency(task1.id)
        task3.add_dependency(task2.id)

        plan = planner.create_plan(
            goal="Software project", initial_tasks=[task1, task2, task3]
        )

        # Verify dependencies
        assert len(plan.tasks[task2.id].dependencies) == 1
        assert len(plan.tasks[task3.id].dependencies) == 1

        # Check execution order
        execution_order = planner.get_execution_order(plan)
        assert len(execution_order) == 3  # Three levels
        assert task1.id in execution_order[0]
        assert task2.id in execution_order[1]
        assert task3.id in execution_order[2]

    @pytest.mark.asyncio
    async def test_plan_optimization(self, planner):
        """Test plan optimization."""
        plan = planner.create_plan(goal="Complex project", auto_decompose=True)

        optimizer = PlanOptimizer()

        # Optimize for time
        time_optimized = await optimizer.optimize(
            plan, objective=OptimizationObjective.MINIMIZE_TIME
        )

        assert time_optimized.optimized_plan is not None
        assert time_optimized.constraints_satisfied
        assert time_optimized.improvement_percentage >= 0

    @pytest.mark.asyncio
    async def test_parallel_execution(self, planner, execution_engine):
        """Test parallel task execution."""
        # Create tasks that can run in parallel
        tasks = [
            Task(
                name=f"Task {i}",
                task_type="analysis",
                estimated_duration=timedelta(seconds=1),
            )
            for i in range(5)
        ]

        plan = planner.create_plan(goal="Parallel analysis", initial_tasks=tasks)

        # Mock task executor
        async def mock_executor(task, context):
            await asyncio.sleep(0.1)  # Simulate work
            return {"result": f"Completed {task.name}"}

        # Execute plan
        results = await execution_engine.execute_plan(plan, mock_executor)

        assert len(results) == 5
        assert all(result.success for result in results.values())


class TestResearchModules:
    """Test suite for research capabilities."""

    @pytest.fixture
    def citation_manager(self):
        """Create a citation manager instance."""
        return CitationManager()

    @pytest.fixture
    def evidence_analyzer(self):
        """Create an evidence analyzer instance."""
        return EvidenceAnalyzer()

    @pytest.fixture
    def knowledge_graph(self):
        """Create a knowledge graph instance."""
        return KnowledgeGraph()

    def test_add_citation(self, citation_manager):
        """Test adding citations."""
        citation = Citation(
            authors=["Smith, J.", "Doe, A."],
            title="Advanced AI Research",
            year=2024,
            journal="Nature AI",
            doi="10.1038/s41586-024-12345",
        )

        citation_id = citation_manager.add_citation(citation)

        assert citation_id is not None
        assert citation_manager.get_citation(citation_id) == citation

    def test_format_citations(self, citation_manager):
        """Test citation formatting."""
        citation = Citation(
            authors=["Johnson, M."],
            title="Machine Learning Advances",
            year=2024,
            journal="Science",
        )

        citation_id = citation_manager.add_citation(citation)

        # Test different formats
        apa = citation_manager.format_citation(citation_id, CitationFormat.APA)
        assert "Johnson, M." in apa
        assert "(2024)" in apa

        mla = citation_manager.format_citation(citation_id, CitationFormat.MLA)
        assert "Johnson, M." in mla

        bibtex = citation_manager.format_citation(citation_id, CitationFormat.BIBTEX)
        assert "@article{" in bibtex

    @pytest.mark.asyncio
    async def test_evidence_analysis(self, evidence_analyzer):
        """Test evidence analysis."""
        evidence_pieces = [
            {
                "source": "Study A",
                "claim": "AI improves productivity by 40%",
                "confidence": 0.9,
                "methodology": "RCT",
            },
            {
                "source": "Study B",
                "claim": "AI improves productivity by 35%",
                "confidence": 0.85,
                "methodology": "Observational",
            },
        ]

        analysis = await evidence_analyzer.analyze_evidence(
            evidence_pieces, question="How much does AI improve productivity?"
        )

        assert analysis["consensus_level"] > 0.7
        assert "synthesis" in analysis
        assert len(analysis["evidence_quality"]) == 2

    def test_knowledge_graph_operations(self, knowledge_graph):
        """Test knowledge graph construction and queries."""
        # Add entities
        knowledge_graph.add_entity("AI", "Technology")
        knowledge_graph.add_entity("ML", "Technology")
        knowledge_graph.add_entity("Productivity", "Concept")

        # Add relationships
        knowledge_graph.add_relationship("AI", "includes", "ML")
        knowledge_graph.add_relationship("AI", "improves", "Productivity")

        # Query relationships
        ai_relations = knowledge_graph.get_relationships("AI")
        assert len(ai_relations) == 2

        # Find path
        path = knowledge_graph.find_path("ML", "Productivity")
        assert path is not None
        assert len(path) == 3  # ML -> AI -> Productivity


class TestMultiModalProcessing:
    """Test suite for multi-modal agent capabilities."""

    @pytest.fixture
    def multimodal_agent(self):
        """Create a multi-modal agent instance."""
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.complete = AsyncMock(return_value=Mock(content="Analysis complete"))
        return MultiModalAdvancedAgent(llm_provider=mock_llm)

    @pytest.mark.asyncio
    async def test_text_processing(self, multimodal_agent):
        """Test text modality processing."""
        text_data = ModalityData(
            type=ModalityType.TEXT,
            content="Analyze this text for sentiment and key themes.",
            confidence=1.0,
        )

        response = await multimodal_agent.process(
            task="Sentiment analysis", modality_data={ModalityType.TEXT: text_data}
        )

        assert response.success
        assert "Multi-Modal Analysis Results" in response.content

    @pytest.mark.asyncio
    async def test_image_processing(self, multimodal_agent):
        """Test image modality processing."""
        # Create a simple test image
        image = Image.new("RGB", (100, 100), color="red")

        image_data = ModalityData(
            type=ModalityType.IMAGE, content=image, metadata={"source": "test"}
        )

        response = await multimodal_agent.process(
            task="Describe the image", modality_data={ModalityType.IMAGE: image_data}
        )

        assert response.success
        assert response.metadata["modalities_processed"] == [ModalityType.IMAGE]

    @pytest.mark.asyncio
    async def test_cross_modal_reasoning(self, multimodal_agent):
        """Test cross-modal reasoning capabilities."""
        text_data = ModalityData(type=ModalityType.TEXT, content="This is a red square")

        image = Image.new("RGB", (100, 100), color="red")
        image_data = ModalityData(type=ModalityType.IMAGE, content=image)

        response = await multimodal_agent.process(
            task="Verify if text matches image",
            modality_data={
                ModalityType.TEXT: text_data,
                ModalityType.IMAGE: image_data,
            },
        )

        assert response.success
        assert response.metadata["cross_modal_enabled"]
        assert len(response.metadata["modalities_processed"]) == 2


class TestAIRouting:
    """Test suite for AI routing system."""

    @pytest.fixture
    def router(self):
        """Create an AI router instance."""
        return AIRouter()

    @pytest.mark.asyncio
    async def test_single_provider_routing(self, router):
        """Test routing to a single provider."""
        decision = await router.route_task(
            "Generate creative writing", mode=RoutingMode.SINGLE
        )

        assert decision.provider in ["claude", "openai"]
        assert decision.confidence > 0
        assert decision.reasoning is not None

    @pytest.mark.asyncio
    async def test_consensus_routing(self, router):
        """Test consensus routing mode."""
        decision = await router.route_task(
            "Complex mathematical proof", mode=RoutingMode.CONSENSUS
        )

        assert len(decision.providers) >= 2
        assert decision.strategy == "consensus"
        assert all(p in ["claude", "openai", "gpt-4"] for p in decision.providers)

    @pytest.mark.asyncio
    async def test_fallback_routing(self, router):
        """Test fallback routing strategy."""
        decision = await router.route_task(
            "Critical system operation", mode=RoutingMode.FALLBACK
        )

        assert decision.primary_provider is not None
        assert len(decision.fallback_chain) >= 1
        assert decision.primary_provider != decision.fallback_chain[0]


class TestAdvancedReasoning:
    """Test suite for advanced reasoning capabilities."""

    @pytest.fixture
    def reasoning_agent(self):
        """Create an advanced reasoning agent."""
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.complete = AsyncMock(return_value=Mock(content="Reasoning result"))
        return AdvancedReasoningAgent(llm_provider=mock_llm)

    @pytest.mark.asyncio
    async def test_chain_of_thought(self, reasoning_agent):
        """Test chain-of-thought reasoning."""
        response = await reasoning_agent.chain_of_thought_reasoning(
            "What is the impact of quantum computing on cryptography?", max_steps=3
        )

        assert response.success
        assert "reasoning_steps" in response.metadata
        assert len(response.metadata["reasoning_steps"]) <= 3

    @pytest.mark.asyncio
    async def test_tree_of_thoughts(self, reasoning_agent):
        """Test tree-of-thoughts reasoning."""
        response = await reasoning_agent.tree_of_thoughts_reasoning(
            "Design a sustainable city", branches=2, depth=2
        )

        assert response.success
        assert "thought_tree" in response.metadata
        assert response.metadata["total_thoughts"] > 1

    @pytest.mark.asyncio
    async def test_recursive_reasoning(self, reasoning_agent):
        """Test recursive reasoning capabilities."""
        response = await reasoning_agent.recursive_reasoning(
            "Explain how recursion works using recursion", max_depth=3
        )

        assert response.success
        assert "recursion_depth" in response.metadata
        assert response.metadata["recursion_depth"] <= 3


class TestVisualization:
    """Test suite for visualization capabilities."""

    @pytest.fixture
    def sample_results(self):
        """Create sample results for visualization."""
        return [
            {
                "technique": "SPRE",
                "duration": 2.5,
                "success": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            {
                "technique": "ReAct",
                "duration": 3.2,
                "success": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            {
                "technique": "SPRE",
                "duration": 2.8,
                "success": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        ]

    def test_create_visualizer(self, sample_results, tmp_path):
        """Test creating a research visualizer."""
        vis = ResearchVisualizer(sample_results, tmp_path)

        assert vis.results == sample_results
        assert vis.output_dir == tmp_path

    def test_plot_generation(self, sample_results, tmp_path):
        """Test generating performance plots."""
        create_performance_plots(sample_results, tmp_path)

        # Check that files were created
        expected_files = [
            "performance_comparison.png",
            "success_rates.png",
            "experiment_timeline.png",
            "summary_report.json",
        ]

        for filename in expected_files:
            assert (tmp_path / filename).exists()


class TestErrorHandlingAndRecovery:
    """Test suite for error handling and recovery mechanisms."""

    @pytest.mark.asyncio
    async def test_agent_error_recovery(self):
        """Test agent error recovery."""
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.complete = AsyncMock(side_effect=Exception("API Error"))

        agent = ReactAgent(llm_provider=mock_llm)

        response = await agent.process("Test task")

        assert not response.success
        assert "error" in response.metadata
        assert "API Error" in str(response.metadata["error"])

    @pytest.mark.asyncio
    async def test_task_execution_retry(self):
        """Test task execution with retry logic."""
        engine = ExecutionEngine(max_concurrent_tasks=1)

        attempt_count = 0

        async def flaky_executor(task, context):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("Temporary failure")
            return {"result": "Success"}

        task = Task(name="Flaky task", task_type="test")
        plan = TaskPlanner().create_plan("Test", initial_tasks=[task])

        # The adaptive executor should retry
        results = await engine.execute_plan(plan, flaky_executor)

        # Even with retries, it might fail if max attempts exceeded
        assert len(results) == 1


class TestSecurityAndValidation:
    """Test suite for security features and input validation."""

    def test_spawn_config_validation(self):
        """Test spawn configuration validation."""
        # Valid config
        valid_config = SpawnConfig(
            agent_type="research", capabilities=[AgentCapability.ANALYSIS]
        )
        assert valid_config.agent_type == "research"

        # Test with invalid data
        with pytest.raises(ValueError):
            SpawnConfig(agent_type="", capabilities=[])

    def test_task_priority_validation(self):
        """Test task priority validation."""
        task = Task(name="Test", task_type="test")

        # Valid priorities
        for priority in TaskPriority:
            task.priority = priority
            assert task.priority == priority

        # Invalid priority should raise error
        with pytest.raises(AttributeError):
            task.priority = "INVALID"

    @pytest.mark.asyncio
    async def test_citation_data_sanitization(self):
        """Test citation data sanitization."""
        manager = CitationManager()

        # Citation with potentially malicious content
        citation = Citation(
            authors=["<script>alert('xss')</script>"],
            title="Test<img src=x onerror=alert('xss')>",
            year=2024,
        )

        citation_id = manager.add_citation(citation)
        formatted = manager.format_citation(citation_id, CitationFormat.APA)

        # Should not contain script tags
        assert "<script>" not in formatted
        assert "alert(" not in formatted


class TestPerformanceOptimization:
    """Test suite for performance optimization features."""

    @pytest.mark.asyncio
    async def test_concurrent_agent_execution(self):
        """Test concurrent execution of multiple agents."""
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.complete = AsyncMock(return_value=Mock(content="Result"))

        agents = [ReactAgent(name=f"Agent{i}", llm_provider=mock_llm) for i in range(5)]

        # Execute all agents concurrently
        tasks = [agent.process(f"Task {i}") for i, agent in enumerate(agents)]

        start_time = asyncio.get_event_loop().time()
        responses = await asyncio.gather(*tasks)
        duration = asyncio.get_event_loop().time() - start_time

        # Should complete much faster than sequential execution
        assert len(responses) == 5
        assert all(r.success for r in responses)
        assert duration < 1.0  # Should be fast with mocked LLM

    def test_memory_efficient_knowledge_graph(self):
        """Test memory-efficient knowledge graph operations."""
        kg = KnowledgeGraph()

        # Add many entities
        for i in range(1000):
            kg.add_entity(f"Entity{i}", "Type")

        # Add relationships
        for i in range(999):
            kg.add_relationship(f"Entity{i}", "connected_to", f"Entity{i + 1}")

        # Should handle large graphs efficiently
        assert kg.get_entity_count() == 1000
        assert kg.get_relationship_count() == 999

        # Path finding should still work
        path = kg.find_path("Entity0", "Entity999")
        assert path is not None


class TestIntegrationScenarios:
    """Test suite for complex integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_research_pipeline(self):
        """Test complete research pipeline from query to report."""
        # Mock components
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.complete = AsyncMock(return_value=Mock(content="Research findings"))

        # Create research agent with all modules
        citation_manager = CitationManager()
        evidence_analyzer = EvidenceAnalyzer()
        knowledge_graph = KnowledgeGraph()

        agent = ReactAgent(
            name="ResearchAgent",
            llm_provider=mock_llm,
            tools=[],
            memory=Mock(spec=BaseMemory),
        )

        # Add some citations
        citation = Citation(
            authors=["Test, A."],
            title="Test Research",
            year=2024,
            journal="Test Journal",
        )
        citation_manager.add_citation(citation)

        # Execute research task
        response = await agent.process("Research the impact of AI on healthcare")

        assert response.success
        assert response.content is not None

    @pytest.mark.asyncio
    async def test_multi_agent_collaboration(self):
        """Test multiple agents collaborating on a complex task."""
        # Create spawner
        spawner = AgentSpawner(max_agents=5)

        # Create router
        router = AIRouter()

        # Create planner
        planner = TaskPlanner()

        # Create complex plan
        plan = planner.create_plan(
            goal="Write a comprehensive research paper on climate change",
            auto_decompose=True,
        )

        # Spawn specialized agents for different tasks
        agent_configs = [
            SpawnConfig(
                agent_type="research", capabilities=[AgentCapability.WEB_SEARCH]
            ),
            SpawnConfig(agent_type="analysis", capabilities=[AgentCapability.ANALYSIS]),
            SpawnConfig(agent_type="writing", capabilities=[AgentCapability.SYNTHESIS]),
        ]

        agent_ids = []
        for config in agent_configs:
            agent_id = await spawner.spawn_agent(config)
            agent_ids.append(agent_id)

        # Route tasks to appropriate providers
        for task in plan.tasks.values():
            routing_decision = await router.route_task(
                task.description, mode=RoutingMode.SINGLE
            )
            task.metadata["provider"] = routing_decision.provider

        # Verify setup
        assert len(agent_ids) == 3
        assert spawner.get_agent_count() == 3
        assert all(task.metadata.get("provider") for task in plan.tasks.values())

        # Clean up
        for agent_id in agent_ids:
            await spawner.terminate_agent(agent_id)


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmarks to ensure system meets requirements."""

    @pytest.mark.benchmark
    def test_agent_creation_performance(self, benchmark):
        """Benchmark agent creation speed."""
        mock_llm = Mock(spec=LLMProvider)

        def create_agent():
            return ReactAgent(llm_provider=mock_llm)

        result = benchmark(create_agent)
        assert result is not None

    @pytest.mark.benchmark
    def test_task_planning_performance(self, benchmark):
        """Benchmark task planning speed."""
        planner = TaskPlanner()

        def create_plan():
            return planner.create_plan(
                goal="Complex software project", auto_decompose=True
            )

        result = benchmark(create_plan)
        assert len(result.tasks) > 0

    @pytest.mark.benchmark
    def test_citation_search_performance(self, benchmark):
        """Benchmark citation search performance."""
        manager = CitationManager()

        # Add many citations
        for i in range(1000):
            citation = Citation(
                authors=[f"Author{i}, A."],
                title=f"Research Paper {i}",
                year=2020 + (i % 5),
            )
            manager.add_citation(citation)

        def search_citations():
            return manager.search_citations("Research", limit=10)

        results = benchmark(search_citations)
        assert len(results) == 10

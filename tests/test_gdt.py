# pyright: reportMissingImports=false

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

# type: ignore
from llamaagent.agents.base import AgentRole

# type: ignore
from llamaagent.data_generation import DebateNode, DebateTrace, GDTOrchestrator

# type: ignore
from llamaagent.llm import MockProvider


@pytest.mark.asyncio
async def test_debate_node_creation():
    """Test debate node creation."""
    node = DebateNode(
        proposal="Test proposal",
        proposing_agent_role=AgentRole.RESEARCHER,
        critique="Test critique",
        score=0.8,
    )

    assert node.proposal == "Test proposal"
    assert node.proposing_agent_role == AgentRole.RESEARCHER
    assert node.critique == "Test critique"
    assert node.score == 0.8
    assert not node.is_terminal


@pytest.mark.asyncio
async def test_debate_trace_creation():
    """Test debate trace creation."""
    trace = DebateTrace(
        original_problem="Test problem",
        final_answer="Test answer",
        full_debate_transcript=[
            {"from": "human", "value": "Test question"},
            {"from": "gpt", "value": "Test response"},
        ],
        total_nodes=5,
        tree_depth=3,
    )

    assert trace.original_problem == "Test problem"
    assert trace.final_answer == "Test answer"
    assert len(trace.full_debate_transcript) == 2
    assert trace.total_nodes == 5
    assert trace.tree_depth == 3


@pytest.mark.asyncio
async def test_gdt_orchestrator_creation():
    """Test GDT orchestrator creation."""
    orchestrator = GDTOrchestrator(llm_provider=MockProvider())

    assert orchestrator.llm is not None
    assert orchestrator.researcher is not None
    assert orchestrator.analyzer is not None
    assert orchestrator.critic is not None


@pytest.mark.asyncio
async def test_debate_trace_generation():
    """Test debate trace generation."""
    orchestrator = GDTOrchestrator(llm_provider=MockProvider())

    problem = "What is the capital of France?"
    trace = await orchestrator.generate_debate_trace(problem, max_depth=2)

    assert isinstance(trace, DebateTrace)
    assert trace.original_problem == problem
    assert isinstance(trace.final_answer, str)
    assert len(trace.full_debate_transcript) > 0
    assert trace.total_nodes > 0


@pytest.mark.asyncio
async def test_proposal_generation():
    """Test proposal generation."""
    orchestrator = GDTOrchestrator(llm_provider=MockProvider())

    # Initialize with a root node
    root = DebateNode(
        proposal="Problem: Test problem",
        proposing_agent_role=AgentRole.COORDINATOR,
    )
    orchestrator.debate_tree[root.node_id] = root

    proposals = await orchestrator._generate_proposals(root.node_id, "Test problem")

    assert isinstance(proposals, list)
    assert len(proposals) > 0

    for proposal in proposals:
        assert "content" in proposal
        assert "role" in proposal


@pytest.mark.asyncio
async def test_proposal_evaluation():
    """Test proposal evaluation."""
    orchestrator = GDTOrchestrator(llm_provider=MockProvider())

    node = DebateNode(
        proposal="Test proposal for evaluation",
        proposing_agent_role=AgentRole.RESEARCHER,
    )

    score, critique = await orchestrator._evaluate_proposal(node, "Test problem")

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert isinstance(critique, str)


@pytest.mark.asyncio
async def test_winning_path_extraction():
    """Test winning path extraction."""
    orchestrator = GDTOrchestrator()

    # Create a simple tree
    root = DebateNode(node_id="root", proposal="Root")
    child1 = DebateNode(node_id="child1", parent_id="root", proposal="Child 1")
    child2 = DebateNode(node_id="child2", parent_id="child1", proposal="Child 2")

    orchestrator.debate_tree = {
        "root": root,
        "child1": child1,
        "child2": child2,
    }

    path = orchestrator._extract_winning_path("child2")

    assert len(path) == 3
    assert path[0].node_id == "root"
    assert path[1].node_id == "child1"
    assert path[2].node_id == "child2"


@pytest.mark.asyncio
async def test_transcript_formatting():
    """Test transcript formatting."""
    orchestrator = GDTOrchestrator()

    path = [
        DebateNode(proposal="Problem: Test", proposing_agent_role=AgentRole.COORDINATOR),
        DebateNode(proposal="Research finding", proposing_agent_role=AgentRole.RESEARCHER, critique="Good", score=0.8),
        DebateNode(
            proposal="Analysis result", proposing_agent_role=AgentRole.ANALYZER, critique="Excellent", score=0.9
        ),
    ]

    transcript = orchestrator._format_transcript(path, "Test problem")

    assert isinstance(transcript, list)
    assert len(transcript) > 0
    assert transcript[0]["from"] == "human"
    assert "Test problem" in transcript[0]["value"]

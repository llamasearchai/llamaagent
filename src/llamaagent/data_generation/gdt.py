from __future__ import annotations

"""
Ground Truth Data (GDT) Generation - Compatibility Module

This module provides backward compatibility by re-exporting from the main data.gdt module
and includes additional specialized functionality for debate tree generation.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

# Re-export everything from the main GDT module
# Additional imports for specialized functionality
from ..agents.base import AgentConfig
from ..agents.react import ReactAgent
from ..llm import MockProvider


class AgentRole(str, Enum):
    """Roles for debate agents."""

    RESEARCHER = "researcher"
    ANALYZER = "analyzer"
    CRITIC = "critic"
    COORDINATOR = "coordinator"
    GENERALIST = "generalist"


@dataclass
class DebateNode:
    """Node in the debate tree."""

    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    proposal: str = ""
    proposing_agent_role: AgentRole = AgentRole.GENERALIST
    critique: str = ""
    score: float = 0.0
    is_terminal: bool = False
    children: List[str] = field(default_factory=list)


@dataclass
class DebateTrace:
    """Complete debate trace from problem to solution."""

    original_problem: str
    final_answer: str
    full_debate_transcript: List[Dict[str, str]]
    total_nodes: int
    tree_depth: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class GDTOrchestrator:
    """Orchestrates generative debate tree creation."""

    RESEARCHER_PROMPT = """You are a researcher in a debate. Given the current argument, 
find a verifiable piece of external information that either supports or refutes it. 
Provide a clear, factual statement."""

    ANALYZER_PROMPT = """You are an analyzer in a debate. Given the current argument, 
propose the next logical deduction or mathematical step required to advance the problem-solving process."""

    CRITIC_PROMPT = """You are a logical reasoner and critic. Analyze the following proposal in the context of the overall problem.

Assess:
1. Factual accuracy
2. Logical soundness
3. Relevance to the problem

Assign a score from 0.0 to 1.0 and provide a brief justification. Identify any fallacies or errors.

Format your response as:
SCORE: [0.0-1.0]
JUSTIFICATION: [brief explanation]"""

    def __init__(self, llm_provider: Optional[Any] = None) -> None:
        self.llm = llm_provider or MockProvider()
        self.debate_tree: Dict[str, DebateNode] = {}
        self.root_id: Optional[str] = None

        # Create specialized agents
        self.researcher = ReactAgent(
            config=AgentConfig(
                name="Researcher",
                description="Finds supporting evidence",
                role=AgentRole.RESEARCHER.value,
            ),
            llm_provider=self.llm,
        )

        self.analyzer = ReactAgent(
            config=AgentConfig(
                name="Analyzer",
                description="Performs logical analysis",
                role=AgentRole.ANALYZER.value,
            ),
            llm_provider=self.llm,
        )

        self.critic = ReactAgent(
            config=AgentConfig(
                name="Critic",
                description="Evaluates arguments",
                role=AgentRole.CRITIC.value,
            ),
            llm_provider=self.llm,
        )

    async def _generate_proposals(
        self, node_id: str, problem: str
    ) -> List[Dict[str, Any]]:
        """Generate proposals from different agents."""
        current_node = self.debate_tree[node_id]
        context = f"Problem: {problem}\nCurrent argument: {current_node.proposal}"

        proposals: List[Dict[str, Any]] = []

        # Researcher proposal
        try:
            researcher_response = await self.researcher.execute(
                f"{self.RESEARCHER_PROMPT}\n\nContext: {context}"
            )
            proposals.append({
                    "content": researcher_response.content,
                    "role": AgentRole.RESEARCHER.value,
                }
            )
        except Exception:
            proposals.append({
                    "content": f"Research shows that {problem} requires systematic analysis.",
                    "role": AgentRole.RESEARCHER.value,
                }
            )

        # Analyzer proposal
        try:
            analyzer_response = await self.analyzer.execute(
                f"{self.ANALYZER_PROMPT}\n\nContext: {context}"
            )
            proposals.append({"content": analyzer_response.content, "role": AgentRole.ANALYZER.value}
            )
        except Exception:
            proposals.append({
                    "content": f"Analysis suggests breaking down {problem} into components.",
                    "role": AgentRole.ANALYZER.value,
                }
            )

        return proposals

    async def _evaluate_proposal(
        self, node: DebateNode, problem: str
    ) -> tuple[float, str]:
        """Evaluate a proposal using the critic."""
        evaluation_prompt = f"""
{self.CRITIC_PROMPT}

Problem: {problem}
Proposal: {node.proposal}
"""

        try:
            response = await self.critic.execute(evaluation_prompt)
            content = response.content

            # Parse score and justification
            score = 0.5  # default
            justification = content

            lines = content.split("\n")
            for line in lines:
                if line.startswith("SCORE:"):
                    try:
                        score = float(line.split(":")[1].strip())
                    except (ValueError, IndexError):
                        pass
                elif line.startswith("JUSTIFICATION:"):
                    justification = line.split(":", 1)[1].strip()

            return score, justification
        except Exception:
            return 0.5, "Standard evaluation applied."

    async def generate_debate_trace(
        self, problem: str, max_depth: int = 5
    ) -> DebateTrace:
        """Generate a complete debate trace for a problem."""
        # Initialize root node
        root = DebateNode(
            proposal=f"Problem: {problem}", proposing_agent_role=AgentRole.COORDINATOR
        )
        self.debate_tree[root.node_id] = root
        self.root_id = root.node_id

        # Expand tree
        current_node_id = root.node_id
        depth = 0

        while depth < max_depth and not self.debate_tree[current_node_id].is_terminal:
            # Generate proposals
            proposals = await self._generate_proposals(current_node_id, problem)

            # Create child nodes
            scored_nodes: List[tuple[float, str]] = []
            for proposal in proposals:
                child_node = DebateNode(
                    parent_id=current_node_id,
                    proposal=proposal["content"],
                    proposing_agent_role=AgentRole(proposal["role"]),
                )

                # Evaluate proposals
                score, critique = await self._evaluate_proposal(child_node, problem)
                child_node.score = score
                child_node.critique = critique

                # Check if terminal
                if "final answer" in proposal["content"].lower() or score > 0.9:
                    child_node.is_terminal = True

                self.debate_tree[child_node.node_id] = child_node
                self.debate_tree[current_node_id].children.append(child_node.node_id)
                scored_nodes.append(score, child_node.node_id)

            # Select best node
            if scored_nodes:
                scored_nodes.sort(reverse=True)
                current_node_id = scored_nodes[0][1]
                depth += 1
            else:
                break

        # Extract winning path
        winning_path = self._extract_winning_path(current_node_id)

        # Generate transcript
        transcript = self._format_transcript(winning_path, problem)

        return DebateTrace(
            original_problem=problem,
            final_answer=(
                winning_path[-1].proposal if winning_path else "No solution found"
            ),
            full_debate_transcript=transcript,
            total_nodes=len(self.debate_tree),
            tree_depth=depth,
        )

    def _extract_winning_path(self, terminal_node_id: str) -> List[DebateNode]:
        """Extract the winning path from root to terminal node."""
        path: List[DebateNode] = []
        current_id = terminal_node_id

        while current_id is not None:
            node = self.debate_tree[current_id]
            path.append(node)
            current_id = node.parent_id

        return list(reversed(path))

    def _format_transcript(
        self, path: List[DebateNode], problem: str
    ) -> List[Dict[str, str]]:
        """Format the debate path as a ShareGPT-style transcript."""
        transcript: List[Dict[str, str]] = [
            {"from": "human", "value": f"Problem: {problem}"}
        ]

        for node in path[1:]:  # Skip root:
            role_name = node.proposing_agent_role.value.title()
            transcript.append({"from": "gpt", "value": f"[{role_name}] {node.proposal}"}
            )

            if node.critique:
                transcript.append({
                        "from": "gpt",
                        "value": f"[Critic] {node.critique} (Score: {node.score:.2f})",
                    }
                )

        return transcript

    async def generate_dataset(
        self, problems: List[str], output_file: str, max_depth: int = 5
    ) -> None:
        """Generate a complete dataset from a list of problems."""
        traces: List[DebateTrace] = []

        for i, problem in enumerate(problems):
            print(f"Processing problem {i + 1}/{len(problems)}: {problem[:50]}...")

            # Reset tree for each problem
            self.debate_tree.clear()
            self.root_id = None

            try:
                trace = await self.generate_debate_trace(problem, max_depth)
                traces.append(trace)
            except Exception as e:
                print(f"Error processing problem {i + 1}: {e}")
                continue

        # Save to file
        with open(output_file, "w") as f:
            for trace in traces:
                if hasattr(trace, "full_debate_transcript"):
                    json.dump(
                        {
                            "conversation": trace.full_debate_transcript,
                            "metadata": {
                                "original_problem": trace.original_problem,
                                "total_nodes": trace.total_nodes,
                                "tree_depth": trace.tree_depth,
                            },
                        },
                        f,
                    )
                    f.write("\n")

        print(f"Generated {len(traces)} debate traces saved to {output_file}")

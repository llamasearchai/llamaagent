from __future__ import annotations

import json
from typing import Dict, List, Optional

from ..agents.base import AgentConfig, AgentRole
from ..agents.react import ReactAgent
from ..llm import MockProvider
from .base import DebateNode, DebateTrace


class GDTOrchestrator:
    """Orchestrates generative debate tree creation."""

    RESEARCHER_PROMPT = """You are a researcher in a debate. Given the current argument, find a verifiable piece of external information that either supports or refutes it. Provide a clear, factual statement."""

    ANALYZER_PROMPT = """You are an analyzer in a debate. Given the current argument, propose the next logical deduction or mathematical step required to advance the problem-solving process."""

    CRITIC_PROMPT = """You are a logical reasoner and critic. Analyze the following proposal in the context of the overall problem.

Assess:
1. Factual accuracy
2. Logical soundness
3. Relevance to the problem

Assign a score from 0.0 to 1.0 and provide a brief justification. Identify any fallacies or errors.

Format your response as:
SCORE: [0.0-1.0]
JUSTIFICATION: [brief explanation]"""

    def __init__(self, llm_provider=None):
        self.llm = llm_provider or MockProvider()
        self.debate_tree: Dict[str, DebateNode] = {}
        self.root_id: Optional[str] = None

        # Create specialized agents
        self.researcher = ReactAgent(
            config=AgentConfig(
                name="Researcher",
                role=AgentRole.RESEARCHER,
                description="Finds supporting evidence",
            ),
            llm_provider=self.llm,
        )

        self.analyzer = ReactAgent(
            config=AgentConfig(
                name="Analyzer",
                role=AgentRole.ANALYZER,
                description="Performs logical analysis",
            ),
            llm_provider=self.llm,
        )

        self.critic = ReactAgent(
            config=AgentConfig(
                name="Critic",
                role=AgentRole.CRITIC,
                description="Evaluates arguments",
            ),
            llm_provider=self.llm,
        )

    async def generate_debate_trace(self, problem: str, max_depth: int = 5) -> DebateTrace:
        """Generate a complete debate trace for a problem."""
        # Initialize root node
        root = DebateNode(
            proposal=f"Problem: {problem}",
            proposing_agent_role=AgentRole.COORDINATOR,
        )
        self.debate_tree[root.node_id] = root
        self.root_id = root.node_id

        # Expand tree
        current_node_id = root.node_id
        depth = 0

        while depth < max_depth and not self.debate_tree[current_node_id].is_terminal:
            # Generate proposals
            proposals = await self._generate_proposals(current_node_id, problem)

            # Evaluate proposals
            scored_nodes = []
            for proposal in proposals:
                child_node = DebateNode(
                    parent_id=current_node_id,
                    proposal=proposal["content"],
                    proposing_agent_role=AgentRole(proposal["role"]),
                )

                # Get critique and score
                score, critique = await self._evaluate_proposal(child_node, problem)
                child_node.score = score
                child_node.critique = critique

                # Check if terminal
                if "final answer" in proposal["content"].lower() or score > 0.9:
                    child_node.is_terminal = True

                self.debate_tree[child_node.node_id] = child_node
                self.debate_tree[current_node_id].children.append(child_node.node_id)
                scored_nodes.append((score, child_node.node_id))

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
            final_answer=winning_path[-1].proposal if winning_path else "No solution found",
            full_debate_transcript=transcript,
            winning_path=winning_path,
            total_nodes=len(self.debate_tree),
            tree_depth=depth,
        )

    async def _generate_proposals(self, node_id: str, problem: str) -> List[Dict[str, str]]:
        """Generate proposals from different agents."""
        current_node = self.debate_tree[node_id]
        context = f"Problem: {problem}\nCurrent argument: {current_node.proposal}"

        proposals = []

        # Researcher proposal
        researcher_response = await self.researcher.execute(f"{self.RESEARCHER_PROMPT}\n\nContext: {context}")
        proposals.append(
            {
                "content": researcher_response.content,
                "role": AgentRole.RESEARCHER,
            }
        )

        # Analyzer proposal
        analyzer_response = await self.analyzer.execute(f"{self.ANALYZER_PROMPT}\n\nContext: {context}")
        proposals.append(
            {
                "content": analyzer_response.content,
                "role": AgentRole.ANALYZER,
            }
        )

        return proposals

    async def _evaluate_proposal(self, node: DebateNode, problem: str) -> tuple[float, str]:
        """Evaluate a proposal using the critic."""
        evaluation_prompt = f"""
{self.CRITIC_PROMPT}

Problem: {problem}
Proposal: {node.proposal}
"""

        response = await self.critic.execute(evaluation_prompt)
        content = response.content

        # Parse score and justification
        score = 0.5  # default
        justification = content

        try:
            lines = content.split("\n")
            for line in lines:
                if line.startswith("SCORE:"):
                    score = float(line.split(":")[1].strip())
                elif line.startswith("JUSTIFICATION:"):
                    justification = line.split(":", 1)[1].strip()
        except (ValueError, IndexError):
            pass

        return score, justification

    def _extract_winning_path(self, terminal_node_id: str) -> List[DebateNode]:
        """Extract the winning path from root to terminal node."""
        path = []
        current_id = terminal_node_id

        while current_id is not None:
            node = self.debate_tree[current_id]
            path.append(node)
            current_id = node.parent_id

        return list(reversed(path))

    def _format_transcript(self, path: List[DebateNode], problem: str) -> List[Dict[str, str]]:
        """Format the debate path as a ShareGPT-style transcript."""
        transcript = [{"from": "human", "value": f"Problem: {problem}"}]

        for node in path[1:]:  # Skip root
            role_name = node.proposing_agent_role.value.title()
            transcript.append({"from": "gpt", "value": f"[{role_name}] {node.proposal}"})

            if node.critique:
                transcript.append({"from": "gpt", "value": f"[Critic] {node.critique} (Score: {node.score:.2f})"})

        return transcript

    async def generate_dataset(
        self,
        problems: List[str],
        output_file: str,
        max_depth: int = 5,
    ) -> None:
        """Generate a complete dataset from a list of problems."""
        traces = []

        for i, problem in enumerate(problems):
            print(f"Processing problem {i+1}/{len(problems)}: {problem[:50]}...")

            # Reset tree for each problem
            self.debate_tree.clear()
            self.root_id = None

            try:
                trace = await self.generate_debate_trace(problem, max_depth)
                traces.append(trace)
            except Exception as e:
                print(f"Error processing problem {i+1}: {e}")
                continue

        # Save to file
        with open(output_file, "w") as f:
            for trace in traces:
                json.dump(
                    {
                        "problem": trace.original_problem,
                        "conversation": trace.full_debate_transcript,
                        "metadata": {
                            "total_nodes": trace.total_nodes,
                            "tree_depth": trace.tree_depth,
                        },
                    },
                    f,
                )
                f.write("\n")

        print(f"Generated {len(traces)} debate traces saved to {output_file}")

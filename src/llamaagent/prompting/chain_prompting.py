"""
Chain-of-Thought and Advanced Chain Prompting Techniques

Implements various chain prompting strategies including:
- Standard Chain-of-Thought (CoT)
- Self-Consistency CoT
- Tree of Thoughts (ToT)
- Graph of Thoughts (GoT)
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from ..llm.messages import LLMMessage
from ..llm.providers.base_provider import BaseLLMProvider


class ChainType(Enum):
    """Types of chain prompting strategies"""

    STANDARD_COT = "standard_cot"
    SELF_CONSISTENCY = "self_consistency"
    TREE_OF_THOUGHTS = "tree_of_thoughts"
    GRAPH_OF_THOUGHTS = "graph_of_thoughts"
    LEAST_TO_MOST = "least_to_most"
    RECURSIVE = "recursive"


@dataclass
class ThoughtNode:
    """Node in a thought chain/tree/graph"""

    id: str
    content: str
    reasoning: str
    confidence: float
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChainResult:
    """Result of chain prompting"""

    final_answer: str
    reasoning_path: List[str]
    confidence: float
    all_thoughts: List[ThoughtNode]
    chain_type: ChainType
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChainOfThoughtPrompt:
    """Advanced Chain-of-Thought prompting implementation"""

    def __init__(self, llm_provider: Optional[BaseLLMProvider] = None):
        self.llm_provider = llm_provider
        self.thought_nodes: Dict[str, ThoughtNode] = {}
        self.node_counter = 0

    async def generate(
        self,
        question: str,
        chain_type: ChainType = ChainType.STANDARD_COT,
        examples: Optional[List[Dict[str, str]]] = None,
        max_thoughts: int = 5,
        temperature: float = 0.7,
    ) -> ChainResult:
        """Generate chain-of-thought reasoning"""

        if chain_type == ChainType.STANDARD_COT:
            return await self._standard_cot(question, examples, temperature)
        elif chain_type == ChainType.SELF_CONSISTENCY:
            return await self._self_consistency_cot(
                question, examples, max_thoughts, temperature
            )
        elif chain_type == ChainType.TREE_OF_THOUGHTS:
            return await self._tree_of_thoughts(question, max_thoughts, temperature)
        elif chain_type == ChainType.GRAPH_OF_THOUGHTS:
            return await self._graph_of_thoughts(question, max_thoughts, temperature)
        elif chain_type == ChainType.LEAST_TO_MOST:
            return await self._least_to_most(question, temperature)
        else:
            raise ValueError(f"Unsupported chain type: {chain_type}")

    async def _standard_cot(
        self,
        question: str,
        examples: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
    ) -> ChainResult:
        """Standard chain-of-thought prompting"""

        prompt_parts = ["Let's solve this step by step."]

        # Add examples if provided
        if examples:
            prompt_parts.append("\nExamples:")
            for i, example in enumerate(examples, 1):
                prompt_parts.append(f"\nExample {i}:")
                prompt_parts.append(f"Q: {example['question']}")
                prompt_parts.append("A: Let me think step by step.")
                prompt_parts.append(f"{example['reasoning']}")
                prompt_parts.append(f"Therefore, {example['answer']}")

        prompt_parts.extend(
            [
                "\nNow, let's solve the actual problem:",
                f"Q: {question}",
                "A: Let me think step by step.",
            ]
        )

        prompt = "\n".join(prompt_parts)

        # Generate response
        response = await self._get_llm_response(prompt, temperature)

        # Parse response into thought steps
        thought_steps = self._parse_cot_response(response)

        # Create thought nodes
        nodes = []
        parent_id = None
        for i, step in enumerate(thought_steps):
            node = ThoughtNode(
                id=self._generate_node_id(),
                content=step["content"],
                reasoning=step["reasoning"],
                confidence=step.get("confidence", 0.8),
                parent_id=parent_id,
            )
            nodes.append(node)
            self.thought_nodes[node.id] = node
            parent_id = node.id

        # Extract final answer
        final_answer = self._extract_final_answer(response)

        return ChainResult(
            final_answer=final_answer,
            reasoning_path=[node.content for node in nodes],
            confidence=np.mean([node.confidence for node in nodes]) if nodes else 0.0,
            all_thoughts=nodes,
            chain_type=ChainType.STANDARD_COT,
        )

    async def _self_consistency_cot(
        self,
        question: str,
        examples: Optional[List[Dict[str, str]]] = None,
        num_samples: int = 5,
        temperature: float = 0.7,
    ) -> ChainResult:
        """Self-consistency chain-of-thought with multiple reasoning paths"""

        # Generate multiple reasoning paths
        tasks = []
        for _ in range(num_samples):
            task = self._standard_cot(question, examples, temperature)
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # Aggregate answers
        answer_votes = defaultdict(float)
        all_nodes = []
        all_paths = []

        for result in results:
            answer_votes[result.final_answer] += result.confidence
            all_nodes.extend(result.all_thoughts)
            all_paths.append(result.reasoning_path)

        # Select most confident answer
        final_answer = max(answer_votes.items(), key=lambda x: x[1])[0]

        # Find best reasoning path for the final answer
        best_path = []
        best_confidence = 0.0
        for i, result in enumerate(results):
            if (
                result.final_answer == final_answer
                and result.confidence > best_confidence
            ):
                best_path = result.reasoning_path
                best_confidence = result.confidence

        return ChainResult(
            final_answer=final_answer,
            reasoning_path=best_path,
            confidence=answer_votes[final_answer] / len(results),
            all_thoughts=all_nodes,
            chain_type=ChainType.SELF_CONSISTENCY,
            metadata={
                "num_samples": num_samples,
                "answer_distribution": dict(answer_votes),
                "all_paths": all_paths,
            },
        )

    async def _tree_of_thoughts(
        self,
        question: str,
        max_depth: int = 3,
        temperature: float = 0.7,
        branching_factor: int = 3,
    ) -> ChainResult:
        """Tree of Thoughts reasoning with exploration"""

        # Initialize root node
        root_prompt = f"Problem: {question}\n\nLet's explore different approaches:"
        await self._get_llm_response(root_prompt, temperature)

        root_node = ThoughtNode(
            id=self._generate_node_id(),
            content=question,
            reasoning="Initial problem statement",
            confidence=1.0,
        )
        self.thought_nodes[root_node.id] = root_node

        # Build tree
        [root_node]
        leaf_nodes = await self._build_thought_tree(
            root_node, question, max_depth, branching_factor, temperature
        )

        # Evaluate leaf nodes
        evaluated_leaves = await self._evaluate_leaf_nodes(leaf_nodes, question)

        # Select best path
        best_leaf = max(evaluated_leaves, key=lambda x: x.confidence)
        best_path = self._reconstruct_path(best_leaf)

        return ChainResult(
            final_answer=best_leaf.content,
            reasoning_path=[node.reasoning for node in best_path],
            confidence=best_leaf.confidence,
            all_thoughts=list(self.thought_nodes.values()),
            chain_type=ChainType.TREE_OF_THOUGHTS,
            metadata={
                "tree_depth": max_depth,
                "num_leaves": len(leaf_nodes),
                "explored_nodes": len(self.thought_nodes),
            },
        )

    async def _build_thought_tree(
        self,
        parent: ThoughtNode,
        question: str,
        depth: int,
        branching_factor: int,
        temperature: float,
    ) -> List[ThoughtNode]:
        """Recursively build thought tree"""

        if depth == 0:
            return [parent]

        # Generate child thoughts
        prompt = f"""
Given the problem: {question}
Current thought: {parent.content}
Current reasoning: {parent.reasoning}

Generate {branching_factor} different next steps or approaches:
"""

        response = await self._get_llm_response(prompt, temperature)
        child_thoughts = self._parse_branching_thoughts(response, branching_factor)

        # Create child nodes
        child_nodes = []
        for thought in child_thoughts:
            child = ThoughtNode(
                id=self._generate_node_id(),
                content=thought["content"],
                reasoning=thought["reasoning"],
                confidence=thought.get("confidence", 0.7),
                parent_id=parent.id,
            )
            child_nodes.append(child)
            self.thought_nodes[child.id] = child
            parent.children_ids.append(child.id)

        # Recursively build subtrees
        all_leaves = []
        for child in child_nodes:
            leaves = await self._build_thought_tree(
                child, question, depth - 1, branching_factor, temperature
            )
            all_leaves.extend(leaves)

        return all_leaves

    async def _evaluate_leaf_nodes(
        self, leaf_nodes: List[ThoughtNode], question: str
    ) -> List[ThoughtNode]:
        """Evaluate and score leaf nodes"""

        evaluation_tasks = []
        for leaf in leaf_nodes:
            prompt = f"""
Question: {question}
Proposed solution: {leaf.content}
Reasoning path: {leaf.reasoning}

Evaluate this solution:
1. Is it correct? (yes/no)
2. How confident are you? (0-1)
3. What's the final answer?
"""
            task = self._get_llm_response(prompt, 0.3)
            evaluation_tasks.append(task)

        evaluations = await asyncio.gather(*evaluation_tasks)

        # Update leaf nodes with evaluation results
        for leaf, eval_response in zip(leaf_nodes, evaluations, strict=False):
            eval_data = self._parse_evaluation(eval_response)
            leaf.confidence = eval_data.get("confidence", 0.5)
            leaf.metadata["evaluation"] = eval_data
            if "final_answer" in eval_data:
                leaf.content = eval_data["final_answer"]

        return leaf_nodes

    async def _graph_of_thoughts(
        self, question: str, max_nodes: int = 10, temperature: float = 0.7
    ) -> ChainResult:
        """Graph of Thoughts with interconnected reasoning"""

        # Initialize with root node
        root_node = ThoughtNode(
            id=self._generate_node_id(),
            content=question,
            reasoning="Initial problem",
            confidence=1.0,
        )
        self.thought_nodes[root_node.id] = root_node

        # Build graph iteratively
        active_nodes = [root_node]

        while len(self.thought_nodes) < max_nodes and active_nodes:
            # Select node to expand
            node_to_expand = max(active_nodes, key=lambda x: x.confidence)
            active_nodes.remove(node_to_expand)

            # Generate connected thoughts
            prompt = f"""
Problem: {question}
Current node: {node_to_expand.content}

Generate 2-3 related thoughts that:
1. Build on this idea
2. Connect to other existing ideas
3. Explore alternative perspectives

Existing thoughts:
{self._summarize_existing_thoughts()}
"""

            response = await self._get_llm_response(prompt, temperature)
            new_thoughts = self._parse_graph_thoughts(response)

            # Create new nodes and connections
            for thought in new_thoughts:
                new_node = ThoughtNode(
                    id=self._generate_node_id(),
                    content=thought["content"],
                    reasoning=thought["reasoning"],
                    confidence=thought.get("confidence", 0.7),
                    parent_id=node_to_expand.id,
                )

                # Add connections to related nodes
                for related_id in thought.get("connections", []):
                    if related_id in self.thought_nodes:
                        new_node.metadata.setdefault("connections", []).append(
                            related_id
                        )

                self.thought_nodes[new_node.id] = new_node
                active_nodes.append(new_node)

        # Find best solution path through graph
        solution_path = await self._find_best_graph_path(question)

        return ChainResult(
            final_answer=solution_path[-1].content,
            reasoning_path=[node.reasoning for node in solution_path],
            confidence=np.mean([node.confidence for node in solution_path]),
            all_thoughts=list(self.thought_nodes.values()),
            chain_type=ChainType.GRAPH_OF_THOUGHTS,
            metadata={
                "graph_size": len(self.thought_nodes),
                "connections": sum(
                    len(n.metadata.get("connections", []))
                    for n in self.thought_nodes.values()
                ),
            },
        )

    async def _least_to_most(
        self, question: str, temperature: float = 0.7
    ) -> ChainResult:
        """Least-to-most prompting - decompose then solve"""

        # Decompose the problem
        decompose_prompt = f"""
Problem: {question}

Break this problem down into simpler sub-problems.
List each sub-problem from simplest to most complex:
"""

        decomposition = await self._get_llm_response(decompose_prompt, temperature)
        sub_problems = self._parse_sub_problems(decomposition)

        # Solve sub-problems incrementally
        nodes = []
        solutions = {}

        for i, sub_problem in enumerate(sub_problems):
            # Build context from previous solutions
            context = "\n".join(
                [
                    f"Sub-problem {j + 1}: {sp}\nSolution: {solutions[sp]}"
                    for j, sp in enumerate(sub_problems[:i])
                    if sp in solutions
                ]
            )

            solve_prompt = f"""
Previous solutions:
{context}

Now solve: {sub_problem}
"""

            solution = await self._get_llm_response(solve_prompt, temperature)
            solutions[sub_problem] = solution

            node = ThoughtNode(
                id=self._generate_node_id(),
                content=solution,
                reasoning=f"Solution to: {sub_problem}",
                confidence=0.8 + 0.2 * (i / len(sub_problems)),  # Increasing confidence
                parent_id=nodes[-1].id if nodes else None,
            )
            nodes.append(node)
            self.thought_nodes[node.id] = node

        # Combine solutions for final answer
        final_prompt = f"""
Original problem: {question}

Sub-problem solutions:
{chr(10).join([f"- {sp}: {sol}" for sp, sol in solutions.items()])}

Combine these solutions to answer the original problem:
"""

        final_answer = await self._get_llm_response(final_prompt, temperature)

        return ChainResult(
            final_answer=final_answer,
            reasoning_path=[f"{sp}: {solutions[sp]}" for sp in sub_problems],
            confidence=0.9,
            all_thoughts=nodes,
            chain_type=ChainType.LEAST_TO_MOST,
            metadata={"sub_problems": sub_problems, "solutions": solutions},
        )

    # Helper methods

    def _generate_node_id(self) -> str:
        """Generate unique node ID"""
        self.node_counter += 1
        return f"thought_{self.node_counter}"

    async def _get_llm_response(self, prompt: str, temperature: float) -> str:
        """Get response from LLM"""
        if self.llm_provider:
            message = LLMMessage(role="user", content=prompt)
            response = await self.llm_provider.complete(
                [message], temperature=temperature
            )
            return response.content
        else:
            # Mock response for testing
            return f"Mock response to: {prompt[:50]}..."

    def _parse_cot_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse chain-of-thought response into steps"""
        steps = []
        lines = response.strip().split("\n")
        current_step = {"content": "", "reasoning": ""}

        for line in lines:
            line = line.strip()
            if line.startswith(("Step", "First", "Next", "Then", "Finally")):
                if current_step["content"]:
                    steps.append(current_step)
                current_step = {"content": line, "reasoning": ""}
            elif line:
                current_step["reasoning"] += line + " "

        if current_step["content"]:
            steps.append(current_step)

        return steps

    def _extract_final_answer(self, response: str) -> str:
        """Extract final answer from response"""
        lines = response.strip().split("\n")
        for line in reversed(lines):
            if any(
                marker in line.lower()
                for marker in ["therefore", "answer:", "conclusion:", "final answer"]
            ):
                return line.split(":", 1)[-1].strip()
        return lines[-1].strip() if lines else ""

    def _parse_branching_thoughts(
        self, response: str, expected_count: int
    ) -> List[Dict[str, Any]]:
        """Parse response into multiple thought branches"""
        thoughts = []
        sections = response.split("\n\n")

        for section in sections[:expected_count]:
            thought = {
                "content": section.split("\n")[0] if section else "",
                "reasoning": (
                    "\n".join(section.split("\n")[1:]) if "\n" in section else section
                ),
                "confidence": 0.7,
            }
            thoughts.append(thought)

        return thoughts

    def _parse_evaluation(self, response: str) -> Dict[str, Any]:
        """Parse evaluation response"""
        eval_data = {
            "correct": "yes" in response.lower(),
            "confidence": 0.5,
            "final_answer": "",
        }

        # Extract confidence
        import re

        confidence_match = re.search(r"confidence[:\s]+([0-9.]+)", response.lower())
        if confidence_match:
            eval_data["confidence"] = float(confidence_match.group(1))

        # Extract final answer
        lines = response.strip().split("\n")
        for line in lines:
            if "answer" in line.lower():
                eval_data["final_answer"] = line.split(":", 1)[-1].strip()
                break

        return eval_data

    def _reconstruct_path(self, leaf: ThoughtNode) -> List[ThoughtNode]:
        """Reconstruct path from root to leaf"""
        path = []
        current = leaf

        while current:
            path.append(current)
            current = (
                self.thought_nodes.get(current.parent_id) if current.parent_id else None
            )

        return list(reversed(path))

    def _summarize_existing_thoughts(self) -> str:
        """Summarize existing thoughts for context"""
        summaries = []
        for node_id, node in list(self.thought_nodes.items())[
            :5
        ]:  # Limit to recent thoughts
            summaries.append(f"- {node.content[:50]}...")
        return "\n".join(summaries)

    def _parse_graph_thoughts(self, response: str) -> List[Dict[str, Any]]:
        """Parse response for graph-based thoughts"""
        thoughts = []
        sections = response.split("\n\n")

        for section in sections:
            lines = section.strip().split("\n")
            if lines:
                thought = {
                    "content": lines[0],
                    "reasoning": " ".join(lines[1:]) if len(lines) > 1 else "",
                    "confidence": 0.7,
                    "connections": [],  # Could parse connections from response
                }
                thoughts.append(thought)

        return thoughts

    def _parse_sub_problems(self, response: str) -> List[str]:
        """Parse decomposed sub-problems"""
        sub_problems = []
        lines = response.strip().split("\n")

        for line in lines:
            line = line.strip()
            if line and any(char in line for char in ["1", "2", "3", "-", "•"]):
                # Remove numbering/bullets
                clean_line = re.sub(r"^[\d\-•.]+\s*", "", line)
                if clean_line:
                    sub_problems.append(clean_line)

        return sub_problems

    async def _find_best_graph_path(self, question: str) -> List[ThoughtNode]:
        """Find best solution path through thought graph"""
        # Simple implementation - can be enhanced with graph algorithms

        # Find nodes that seem like solutions
        solution_nodes = []
        for node in self.thought_nodes.values():
            if any(
                word in node.content.lower()
                for word in ["answer", "solution", "therefore", "conclusion"]
            ):
                solution_nodes.append(node)

        if not solution_nodes:
            solution_nodes = list(self.thought_nodes.values())

        # Select highest confidence solution
        best_solution = max(solution_nodes, key=lambda x: x.confidence)

        # Reconstruct path
        return self._reconstruct_path(best_solution)


class ChainPromptOptimizer:
    """Optimizer for chain prompting strategies"""

    def __init__(self, llm_provider: Optional[BaseLLMProvider] = None):
        self.llm_provider = llm_provider
        self.performance_history: List[Dict[str, Any]] = []

    async def optimize(
        self,
        questions: List[str],
        ground_truth: List[str],
        chain_types: Optional[List[ChainType]] = None,
    ) -> Dict[str, Any]:
        """Find optimal chain prompting strategy for given task"""

        if chain_types is None:
            chain_types = list(ChainType)

        results = {}

        for chain_type in chain_types:
            chain_prompt = ChainOfThoughtPrompt(self.llm_provider)

            correct = 0
            total_confidence = 0.0

            for question, truth in zip(questions, ground_truth, strict=False):
                try:
                    result = await chain_prompt.generate(question, chain_type)

                    # Simple accuracy check
                    is_correct = truth.lower() in result.final_answer.lower()
                    if is_correct:
                        correct += 1

                    total_confidence += result.confidence

                except Exception as e:
                    print(f"Error with {chain_type}: {e}")

            accuracy = correct / len(questions) if questions else 0
            avg_confidence = total_confidence / len(questions) if questions else 0

            results[chain_type.value] = {
                "accuracy": accuracy,
                "avg_confidence": avg_confidence,
                "score": accuracy * avg_confidence,  # Combined metric
            }

        # Find best strategy
        best_strategy = max(results.items(), key=lambda x: x[1]["score"])

        return {
            "best_strategy": best_strategy[0],
            "best_score": best_strategy[1]["score"],
            "all_results": results,
        }

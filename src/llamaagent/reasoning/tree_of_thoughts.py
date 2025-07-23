"""
Tree of Thoughts Implementation for LlamaAgent

Based on "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
by Yao et al., 2023. This implementation provides multi-path reasoning capabilities
with sophisticated search strategies and evaluation mechanisms.

Key Features:
- Multiple search strategies (BFS, DFS, Best-First, Monte Carlo Tree Search)
- Dynamic thought evaluation and pruning
- Confidence-based decision making
- Integration with existing SPRE pipeline
- Comprehensive error handling and validation

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import json
import logging
import math
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ..llm import LLMMessage

logger = logging.getLogger(__name__)


class SearchStrategy(Enum):
    """Available search strategies for Tree of Thoughts"""

    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"
    BEST_FIRST = "best_first"
    MONTE_CARLO = "monte_carlo"
    ADAPTIVE = "adaptive"


@dataclass
class ThoughtNode:
    """Represents a single thought in the reasoning tree"""

    content: str
    depth: int = 0
    score: float = 0.0
    children: List['ThoughtNode'] = field(default_factory=lambda: [])
    parent: Optional['ThoughtNode'] = None
    metadata: Dict[str, Any] = field(default_factory=lambda: {})
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def add_child(self, child: 'ThoughtNode') -> None:
        """Add a child thought to this node"""
        # No need to check isinstance - type annotation handles this

        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)

    def get_path(self) -> List[str]:
        """Get the path from root to this node"""
        path = [self.content]
        current = self.parent
        while current:
            path.append(current.content)
            current = current.parent
        return list(reversed(path))

    def __repr__(self) -> str:
        return f"ThoughtNode(content='{self.content[:50]}...', depth={self.depth}, score={self.score:.2f})"


class ThoughtTree:
    """Tree structure for organizing thoughts"""

    def __init__(self, root_content: str = "Initial Problem"):
        self.root = ThoughtNode(root_content, depth=0)
        self.nodes: Dict[str, ThoughtNode] = {self.root.id: self.root}
        self.max_depth = 10
        self.max_nodes = 100

    def add_node(self, parent_id: str, content: str) -> Optional[ThoughtNode]:
        """Add a new node to the tree"""
        if len(self.nodes) >= self.max_nodes:
            logger.warning("Tree node limit reached")
            return None

        parent = self.nodes.get(parent_id)
        if not parent:
            logger.error(f"Parent node {parent_id} not found")
            return None

        if parent.depth >= self.max_depth:
            logger.warning(f"Max depth {self.max_depth} reached")
            return None

        child = ThoughtNode(content, depth=parent.depth + 1)
        parent.add_child(child)
        self.nodes[child.id] = child
        return child

    def get_leaves(self) -> List[ThoughtNode]:
        """Get all leaf nodes"""
        return [node for node in self.nodes.values() if not node.children]

    def get_best_path(self) -> List[ThoughtNode]:
        """Get the best scoring path from root to leaf"""
        best_leaf = max(self.get_leaves(), key=lambda x: x.score, default=None)
        if not best_leaf:
            return [self.root]

        path: List[ThoughtNode] = []
        current: Optional[ThoughtNode] = best_leaf
        while current:
            path.append(current)
            current = current.parent
        return list(reversed(path))


class ThoughtEvaluator:
    """Evaluates the quality of thoughts"""

    def __init__(self, llm_provider: Any):
        if not llm_provider:
            raise ValueError("LLM provider cannot be null")

        if not hasattr(llm_provider, 'complete') or not callable(llm_provider.complete):
            raise ValueError("LLM provider must implement async complete() method")

        self.llm_provider = llm_provider

        self.evaluation_prompt = """
Evaluate the quality of this reasoning step on a scale of 0.0 to 1.0.

Problem: {problem}
Current reasoning path: {path}
Thought to evaluate: {thought}

Consider:
- Logical coherence (0.0-1.0)
- Progress toward solution (0.0-1.0)
- Creativity and insight (0.0-1.0)
- Feasibility of next steps (0.0-1.0)

Provide your evaluation as JSON:
{{
  "score": 0.85,
  "reasoning": "This thought shows good logical progression...",
  "confidence": 0.9
}}
"""

    async def evaluate_thought(self, thought: ThoughtNode, problem: str) -> float:
        """Evaluate a single thought"""
        try:
            path_description = " -> ".join(thought.get_path())

            prompt = self.evaluation_prompt.format(
                problem=problem, path=path_description, thought=thought.content
            )

            messages = [LLMMessage(role="user", content=prompt)]
            response = await self.llm_provider.complete(messages)

            if not response or not hasattr(response, 'content'):
                logger.warning("Invalid response from LLM provider")
                return 0.5

            content = response.content

            # Try to parse JSON response
            try:
                result = json.loads(content)
                score = result.get("score", 0.5)
                return max(0.0, min(1.0, float(score)))
            except (json.JSONDecodeError, ValueError, TypeError):
                # Fallback: extract score from text
                import re

                score_match = re.search(r'"score":\s*([0-9]*\.?[0-9]+)', content)
                if score_match:
                    score = float(score_match.group(1))
                    return max(0.0, min(1.0, score))
                else:
                    logger.warning("Could not parse evaluation score, using default")
                    return 0.5

        except Exception as e:
            logger.error(f"Error evaluating thought: {e}")
            return 0.5


class SearchStrategyBase(ABC):
    """Abstract base for search strategies"""

    @abstractmethod
    async def search(
        self,
        tree: ThoughtTree,
        evaluator: ThoughtEvaluator,
        problem: str,
        max_iterations: int,
    ) -> List[ThoughtNode]:
        """Execute search strategy"""
        pass


class BreadthFirstSearch(SearchStrategyBase):
    """Breadth-first search strategy"""

    def __init__(self, agent: Optional['TreeOfThoughtsAgent'] = None) -> None:
        """Initialize search strategy"""
        self.agent = agent

    async def search(
        self,
        tree: ThoughtTree,
        evaluator: ThoughtEvaluator,
        problem: str,
        max_iterations: int,
    ) -> List[ThoughtNode]:
        """Execute breadth-first search"""
        queue = [tree.root]
        iterations = 0

        while queue and iterations < max_iterations:
            current = queue.pop(0)
            iterations += 1

            # Generate children for current node
            if self.agent:
                children = await self.agent.generate_children(current, problem)
            else:
                children = []

            for child_content in children:
                child = tree.add_node(current.id, child_content)
                if child:
                    child.score = await evaluator.evaluate_thought(child, problem)
                    queue.append(child)

        return tree.get_best_path()


class BestFirstSearch(SearchStrategyBase):
    """Best-first search strategy"""

    def __init__(self, agent: Optional['TreeOfThoughtsAgent'] = None) -> None:
        """Initialize search strategy"""
        self.agent = agent

    async def search(
        self,
        tree: ThoughtTree,
        evaluator: ThoughtEvaluator,
        problem: str,
        max_iterations: int,
    ) -> List[ThoughtNode]:
        """Execute best-first search"""
        queue = [(0.0, tree.root)]  # (score, node)
        iterations = 0

        while queue and iterations < max_iterations:
            # Get node with highest score
            queue.sort(key=lambda x: x[0], reverse=True)
            _, current = queue.pop(0)
            iterations += 1

            # Generate children
            if self.agent:
                children = await self.agent.generate_children(current, problem)
            else:
                children = []

            for child_content in children:
                child = tree.add_node(current.id, child_content)
                if child:
                    child.score = await evaluator.evaluate_thought(child, problem)
                    queue.append((child.score, child))

        return tree.get_best_path()


class MonteCarloTreeSearch(SearchStrategyBase):
    """Monte Carlo Tree Search strategy"""

    def __init__(self, agent: Optional['TreeOfThoughtsAgent'] = None) -> None:
        """Initialize search strategy"""
        self.agent = agent

    async def search(
        self,
        tree: ThoughtTree,
        evaluator: ThoughtEvaluator,
        problem: str,
        max_iterations: int,
    ) -> List[ThoughtNode]:
        """Execute Monte Carlo Tree Search"""
        iterations = 0

        while iterations < max_iterations:
            # Selection
            node = self._select(tree.root)

            # Expansion
            if not node.children and self.agent:
                children = await self.agent.generate_children(node, problem)
                for child_content in children:
                    child = tree.add_node(node.id, child_content)
                    if child:
                        child.score = await evaluator.evaluate_thought(child, problem)

            # Simulation and backpropagation
            await self._simulate_and_backpropagate(node, evaluator, problem)

            iterations += 1

        return tree.get_best_path()

    def _select(self, node: ThoughtNode) -> ThoughtNode:
        """Select node using UCB1 formula"""
        if not node.children:
            return node

        # UCB1 formula: argmax(score + sqrt(2 * ln(parent_visits) / visits))
        best_child = None
        best_ucb = float('-inf')

        for child in node.children:
            visits = child.metadata.get('visits', 1)
            parent_visits = node.metadata.get('visits', 1)
            ucb = child.score + math.sqrt(2 * math.log(parent_visits) / visits)

            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child

        return best_child or node

    async def _simulate_and_backpropagate(
        self, node: ThoughtNode, evaluator: ThoughtEvaluator, problem: str
    ) -> None:
        """Simulate from node and backpropagate results"""
        # Simple simulation: evaluate the node
        score = await evaluator.evaluate_thought(node, problem)
        node.score = score

        # Backpropagate
        current = node
        while current:
            visits = current.metadata.get('visits', 0)
            current.metadata['visits'] = visits + 1
            current = current.parent


class TreeOfThoughtsAgent:
    """Main Tree of Thoughts reasoning agent"""

    def __init__(
        self,
        llm_provider: Any,
        strategy: SearchStrategy = SearchStrategy.BEST_FIRST,
        max_depth: int = 5,
        beam_width: int = 3,
        max_iterations: int = 50,
    ):
        """Initialize Tree of Thoughts agent with comprehensive validation"""

        # Validate llm_provider
        if not llm_provider:
            raise ValueError("LLM provider cannot be null")

        if not hasattr(llm_provider, 'complete') or not callable(llm_provider.complete):
            raise ValueError("LLM provider must implement async complete() method")

        # Strategy is already typed, no need to validate

        # Validate numeric parameters
        if max_depth <= 0:
            raise ValueError(f"max_depth must be positive, got {max_depth}")

        if beam_width <= 0:
            raise ValueError(f"beam_width must be positive, got {beam_width}")

        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {max_iterations}")

        # Set bounds
        max_depth = min(max_depth, 10)  # Reasonable upper bound
        beam_width = min(beam_width, 10)  # Reasonable upper bound
        max_iterations = min(max_iterations, 100)  # Reasonable upper bound

        self.llm_provider = llm_provider
        self.strategy = strategy
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.max_iterations = max_iterations

        # Initialize components with proper error handling
        try:
            self.evaluator = ThoughtEvaluator(llm_provider)
            logger.info("ThoughtEvaluator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ThoughtEvaluator: {e}")
            raise

        # Initialize search strategies with reference to self
        self.search_strategies = {
            SearchStrategy.BREADTH_FIRST: BreadthFirstSearch(self),
            SearchStrategy.BEST_FIRST: BestFirstSearch(self),
            SearchStrategy.MONTE_CARLO: MonteCarloTreeSearch(self),
        }

        # Validate strategy availability
        if strategy not in self.search_strategies:
            raise ValueError(f"Search strategy {strategy} not implemented")

        # Generation prompts
        self.thought_generation_prompt = """
Given the problem and current reasoning path, generate {num_thoughts} different next reasoning steps.
Each step should be a logical continuation that helps solve the problem.

Problem: {problem}
Current path: {current_path}

Generate {num_thoughts} distinct thoughts as JSON list:
[
  "First reasoning step...",
  "Second reasoning step...",
  "Third reasoning step..."
]
"""

        # Statistics
        self.stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "average_execution_time": 0.0,
            "average_tree_depth": 0.0,
            "average_nodes_created": 0.0,
        }

        logger.info(f"TreeOfThoughtsAgent initialized with strategy: {strategy.value}")

    async def solve(self, problem: str) -> Dict[str, Any]:
        """Solve a problem using Tree of Thoughts reasoning"""
        start_time = time.time()

        try:
            # Validate input
            if not problem:
                raise ValueError("Problem must be a non-empty string")

            # Initialize tree
            tree = ThoughtTree(problem)

            # Get search strategy
            search_strategy = self.search_strategies.get(self.strategy)
            if not search_strategy:
                raise ValueError(f"Search strategy {self.strategy} not available")

            # Execute search
            best_path = await search_strategy.search(
                tree, self.evaluator, problem, self.max_iterations
            )

            # Extract solution from best path
            if best_path:
                solution = best_path[-1].content
                confidence = best_path[-1].score
                path_contents = [node.content for node in best_path]
            else:
                solution = "No solution found"
                confidence = 0.0
                path_contents = []

            # Calculate statistics
            execution_time = time.time() - start_time
            tree_depth = (
                max(node.depth for node in tree.nodes.values()) if tree.nodes else 0
            )
            nodes_created = len(tree.nodes)

            # Update stats
            self.stats["total_executions"] += 1
            self.stats["successful_executions"] += 1
            self.stats["average_execution_time"] = (
                self.stats["average_execution_time"]
                * (self.stats["total_executions"] - 1)
                + execution_time
            ) / self.stats["total_executions"]
            self.stats["average_tree_depth"] = (
                self.stats["average_tree_depth"] * (self.stats["total_executions"] - 1)
                + tree_depth
            ) / self.stats["total_executions"]
            self.stats["average_nodes_created"] = (
                self.stats["average_nodes_created"]
                * (self.stats["total_executions"] - 1)
                + nodes_created
            ) / self.stats["total_executions"]

            return {
                "solution": solution,
                "confidence": confidence,
                "best_path": path_contents,
                "statistics": {
                    "execution_time": execution_time,
                    "tree_depth": tree_depth,
                    "nodes_created": nodes_created,
                    "iterations_used": self.max_iterations,
                    "strategy_used": self.strategy.value,
                },
                "success": True,
                "error": None,
            }

        except Exception as e:
            execution_time = time.time() - start_time
            self.stats["total_executions"] += 1

            logger.error(f"Tree of Thoughts solving failed: {e}")

            return {
                "solution": "Error occurred during solving",
                "confidence": 0.0,
                "best_path": [],
                "statistics": {
                    "execution_time": execution_time,
                    "tree_depth": 0,
                    "nodes_created": 0,
                    "iterations_used": 0,
                    "strategy_used": self.strategy.value,
                },
                "success": False,
                "error": str(e),
            }

    async def generate_children(self, node: ThoughtNode, problem: str) -> List[str]:
        """Generate child thoughts for a node"""
        try:
            current_path = " -> ".join(node.get_path())

            prompt = self.thought_generation_prompt.format(
                problem=problem, current_path=current_path, num_thoughts=self.beam_width
            )

            messages = [LLMMessage(role="user", content=prompt)]
            response = await self.llm_provider.complete(messages)

            if not response or not hasattr(response, 'content'):
                logger.warning("Invalid response from LLM provider")
                return []

            content = response.content

            # Try to parse JSON response
            try:
                thoughts_raw: Any = json.loads(content)
                if isinstance(thoughts_raw, list):
                    thoughts_list: List[Any] = thoughts_raw[: self.beam_width]  # type: ignore
                    return [str(thought) for thought in thoughts_list]
                else:
                    logger.warning("Response is not a list")
                    return []
            except json.JSONDecodeError:
                # Fallback: split by newlines and clean up
                lines = content.split('\n')
                thoughts: List[str] = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('[') and not line.startswith(']'):
                        # Remove quotes and clean up
                        line = line.strip('"\'')
                        if line:
                            thoughts.append(line)

                return thoughts[: self.beam_width]

        except Exception as e:
            logger.error(f"Error generating children: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return self.stats.copy()

    async def reset(self) -> None:
        """Reset agent state"""
        self.stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "average_execution_time": 0.0,
            "average_tree_depth": 0.0,
            "average_nodes_created": 0.0,
        }
        logger.info("TreeOfThoughtsAgent statistics reset")


# Export main classes
__all__ = [
    "TreeOfThoughtsAgent",
    "ThoughtNode",
    "ThoughtTree",
    "SearchStrategy",
    "ThoughtEvaluator",
    "BreadthFirstSearch",
    "BestFirstSearch",
    "MonteCarloTreeSearch",
]

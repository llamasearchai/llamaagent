"""
Cognitive Agent Implementation for LlamaAgent

This module provides a unified cognitive agent that integrates multiple reasoning
strategies including Tree of Thoughts, Graph of Thoughts, Constitutional AI, and
Meta-Reasoning capabilities.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import logging
import time
from typing import Any, Dict, Optional

from ..agents.base import AgentConfig, AgentResponse, BaseAgent
from ..agents.react import ReactAgent
from ..llm import BaseLLMProvider

logger = logging.getLogger(__name__)


class CognitiveAgent(BaseAgent):
    """
    Unified Cognitive Agent with multiple reasoning strategies.

    This agent integrates Tree of Thoughts, Graph of Thoughts, Constitutional AI,
    and Meta-Reasoning capabilities into a single unified interface.
    """

    def __init__(
        self,
        config: AgentConfig,
        llm_provider: BaseLLMProvider,
        tools: Optional[Any] = None,
        memory: Optional[Any] = None,
        enable_constitutional_ai: bool = True,
        enable_meta_reasoning: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the cognitive agent with comprehensive validation"""

        # Store the parameters
        # Config and llm_provider are always provided due to type hints

        # Initialize base agent
        super().__init__(config, **kwargs)

        self.llm_provider = llm_provider
        self.tools = tools
        self.memory = memory

        # Initialize base ReactAgent
        self.base_agent = ReactAgent(config, **kwargs)
        self.enable_constitutional_ai = bool(enable_constitutional_ai)
        self.enable_meta_reasoning = bool(enable_meta_reasoning)

        # Initialize reasoning components with proper error handling
        self.tree_agent = None
        self.graph_agent = None
        self.constitutional_agent = None
        self.meta_agent = None

        try:
            self._initialize_reasoning_components()
            logger.info("Cognitive agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize cognitive agent components: {e}")
            raise

    def _initialize_reasoning_components(self) -> None:
        """Initialize all reasoning components"""

        # Initialize Tree of Thoughts agent
        try:
            from .tree_of_thoughts import TreeOfThoughtsAgent

            self.tree_agent = TreeOfThoughtsAgent(
                llm_provider=self.llm_provider, max_depth=5, beam_width=3
            )
            logger.info("Tree of Thoughts agent initialized")
        except ImportError as e:
            logger.warning(f"Tree of Thoughts agent not available: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize Tree of Thoughts agent: {e}")

        # Initialize Graph of Thoughts agent
        try:
            from .graph_of_thoughts import GraphOfThoughtsAgent

            self.graph_agent = GraphOfThoughtsAgent(
                llm_provider=self.llm_provider, max_concepts=8, max_depth=3
            )
            logger.info("Graph of Thoughts agent initialized")
        except ImportError as e:
            logger.warning(f"Graph of Thoughts agent not available: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize Graph of Thoughts agent: {e}")

        # Initialize Constitutional AI agent
        if self.enable_constitutional_ai:
            try:
                from .constitutional_ai import ConstitutionalAgent

                self.constitutional_agent = ConstitutionalAgent(
                    llm_provider=self.llm_provider, enable_revision=True
                )
                logger.info("Constitutional AI agent initialized")
            except ImportError as e:
                logger.warning(f"Constitutional AI agent not available: {e}")
            except Exception as e:
                logger.error(f"Failed to initialize Constitutional AI agent: {e}")

        # Initialize Meta-Reasoning agent
        if self.enable_meta_reasoning:
            try:
                from .meta_reasoning import MetaCognitiveAgent

                self.meta_agent = MetaCognitiveAgent(llm_provider=self.llm_provider)
                logger.info("Meta-cognitive agent initialized")
            except ImportError as e:
                logger.warning(f"Meta-cognitive agent not available: {e}")
            except Exception as e:
                logger.error(f"Failed to initialize Meta-cognitive agent: {e}")

    async def execute(
        self, task: str, context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Execute a task using the most appropriate reasoning strategy"""
        start_time = time.time()

        try:
            # Validate input
            if not task:
                raise ValueError("Task must be a non-empty string")

            context = context or {}

            # Determine the best reasoning strategy based on task and context
            strategy = self._select_strategy(task, context)

            # Execute using selected strategy
            if strategy == "tree_of_thoughts" and self.tree_agent:
                result = await self._execute_tree_of_thoughts(task, context)
            elif strategy == "graph_of_thoughts" and self.graph_agent:
                result = await self._execute_graph_of_thoughts(task, context)
            elif strategy == "constitutional" and self.constitutional_agent:
                result = await self._execute_constitutional(task, context)
            elif strategy == "meta_reasoning" and self.meta_agent:
                result = await self._execute_meta_reasoning(task, context)
            else:
                # Fallback to basic execution
                result = await self._execute_basic(task, context)

            # Update statistics
            execution_time = time.time() - start_time
            self.stats.update(execution_time, result.success, result.tokens_used)

            # Add strategy information to metadata
            if result.metadata is None:
                result.metadata = {}
            result.metadata["strategy_used"] = strategy
            result.metadata["cognitive_agent"] = True

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self.stats.update(execution_time, False, 0)

            logger.error(f"Cognitive agent execution failed: {e}")

            return AgentResponse(
                content=f"Task execution failed: {str(e)}",
                success=False,
                execution_time=execution_time,
                error=str(e),
                metadata={"strategy_used": "error", "cognitive_agent": True},
            )

    def _select_strategy(self, task: str, context: Dict[str, Any]) -> str:
        """Select the most appropriate reasoning strategy for the task"""

        # Check for explicit strategy in context
        if "strategy" in context:
            strategy = context["strategy"]
            if strategy in [
                "tree_of_thoughts",
                "graph_of_thoughts",
                "constitutional",
                "meta_reasoning",
            ]:
                return strategy

        # Analyze task characteristics to select strategy
        task_lower = task.lower()

        # Constitutional AI for safety-critical tasks
        if any(
            keyword in task_lower
            for keyword in ["ethical", "safety", "harm", "bias", "privacy"]
        ):
            if self.constitutional_agent:
                return "constitutional"

        # Graph of Thoughts for complex, multi-domain problems
        if any(
            keyword in task_lower
            for keyword in [
                "relationship",
                "network",
                "system",
                "integration",
                "cross-domain",
            ]
        ):
            if self.graph_agent:
                return "graph_of_thoughts"

        # Tree of Thoughts for step-by-step reasoning
        if any(
            keyword in task_lower
            for keyword in ["analyze", "solve", "plan", "strategy", "step-by-step"]
        ):
            if self.tree_agent:
                return "tree_of_thoughts"

        # Meta-reasoning for complex decision making
        if any(
            keyword in task_lower
            for keyword in ["choose", "decide", "select", "compare", "evaluate"]
        ):
            if self.meta_agent:
                return "meta_reasoning"

        # Default to tree of thoughts if available
        if self.tree_agent:
            return "tree_of_thoughts"

        # Fallback to basic execution
        return "basic"

    async def _execute_tree_of_thoughts(
        self, task: str, context: Dict[str, Any]
    ) -> AgentResponse:
        """Execute task using Tree of Thoughts reasoning"""
        try:
            if not self.tree_agent:
                raise ValueError("Tree of Thoughts agent not initialized")
            result = await self.tree_agent.solve(task)

            return AgentResponse(
                content=result.get("solution", "No solution found"),
                success=result.get("success", False),
                execution_time=result.get("statistics", {}).get("execution_time", 0.0),
                tokens_used=result.get("statistics", {}).get("tokens_used", 0),
                metadata={
                    "strategy": "tree_of_thoughts",
                    "confidence": result.get("confidence", 0.0),
                    "best_path": result.get("best_path", []),
                    "statistics": result.get("statistics", {}),
                },
            )
        except Exception as e:
            logger.error(f"Tree of Thoughts execution failed: {e}")
            raise

    async def _execute_graph_of_thoughts(
        self, task: str, context: Dict[str, Any]
    ) -> AgentResponse:
        """Execute task using Graph of Thoughts reasoning"""
        try:
            if not self.graph_agent:
                raise ValueError("Graph of Thoughts agent not initialized")
            domain = context.get("domain", "general")
            result = await self.graph_agent.solve(task, domain)

            return AgentResponse(
                content=result.get("solution", "No solution found"),
                success=result.get("success", False),
                execution_time=result.get("statistics", {}).get("execution_time", 0.0),
                tokens_used=result.get("statistics", {}).get("tokens_used", 0),
                metadata={
                    "strategy": "graph_of_thoughts",
                    "confidence": result.get("confidence", 0.0),
                    "concepts": result.get("concepts", []),
                    "relationships": result.get("relationships", []),
                    "reasoning_paths": result.get("reasoning_paths", []),
                    "statistics": result.get("statistics", {}),
                },
            )
        except Exception as e:
            logger.error(f"Graph of Thoughts execution failed: {e}")
            raise

    async def _execute_constitutional(
        self, task: str, context: Dict[str, Any]
    ) -> AgentResponse:
        """Execute task using Constitutional AI reasoning"""
        try:
            if not self.constitutional_agent:
                raise ValueError("Constitutional AI agent not initialized")
            result = await self.constitutional_agent.process_response(task)

            return AgentResponse(
                content=result.get("response", "No response generated"),
                success=result.get("success", False),
                execution_time=result.get("statistics", {}).get("execution_time", 0.0),
                tokens_used=result.get("statistics", {}).get("tokens_used", 0),
                metadata={
                    "strategy": "constitutional",
                    "compliance_score": result.get("compliance_score", 0.0),
                    "overall_compliance": result.get("overall_compliance", False),
                    "violations": result.get("violations", []),
                    "revision_attempts": result.get("revision_attempts", 0),
                    "statistics": result.get("statistics", {}),
                },
            )
        except Exception as e:
            logger.error(f"Constitutional AI execution failed: {e}")
            raise

    async def _execute_meta_reasoning(
        self, task: str, context: Dict[str, Any]
    ) -> AgentResponse:
        """Execute task using Meta-Reasoning"""
        try:
            if not self.meta_agent:
                raise ValueError("Meta-reasoning agent not initialized")
            result = await self.meta_agent.select_and_execute_strategy(task, context)

            return AgentResponse(
                content=result.get("solution", "No solution found"),
                success=result.get("success", False),
                execution_time=result.get("execution_time", 0.0),
                tokens_used=result.get("tokens_used", 0),
                metadata={
                    "strategy": "meta_reasoning",
                    "selected_strategy": result.get("selected_strategy", "unknown"),
                    "confidence": result.get("confidence", 0.0),
                    "strategy_performance": result.get("strategy_performance", {}),
                    "reasoning_trace": result.get("reasoning_trace", []),
                },
            )
        except Exception as e:
            logger.error(f"Meta-reasoning execution failed: {e}")
            raise

    async def _execute_basic(self, task: str, context: Dict[str, Any]) -> AgentResponse:
        """Execute task using basic reasoning (fallback)"""
        try:
            # Simple prompt-based execution
            prompt = f"Task: {task}\n\nContext: {context}\n\nPlease provide a comprehensive response."

            from ..llm import LLMMessage

            messages = [LLMMessage(role="user", content=prompt)]
            response = await self.llm_provider.complete(messages)

            content = (
                response.content if hasattr(response, 'content') else str(response)
            )
            tokens_used = getattr(response, 'tokens_used', len(content) // 4)

            return AgentResponse(
                content=content,
                success=True,
                execution_time=0.0,
                tokens_used=tokens_used,
                metadata={
                    "strategy": "basic",
                    "confidence": 0.5,
                    "fallback_used": True,
                },
            )
        except Exception as e:
            logger.error(f"Basic execution failed: {e}")
            raise

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        base_metrics = self.get_stats()

        # Add cognitive agent specific metrics
        cognitive_metrics = {
            "cognitive_agent_metrics": {
                "components_status": {
                    "tree_of_thoughts": self.tree_agent is not None,
                    "graph_of_thoughts": self.graph_agent is not None,
                    "constitutional_ai": self.constitutional_agent is not None,
                    "meta_reasoning": self.meta_agent is not None,
                },
                "cognitive_agent_stats": {
                    "reasoning_depth": self._calculate_reasoning_depth(),
                    "adaptation_score": self._calculate_adaptation_score(),
                    "confidence_calibration": self._calculate_confidence_calibration(),
                },
            }
        }

        # Add component-specific metrics
        if self.tree_agent:
            try:
                tree_stats = self.tree_agent.get_statistics()
                cognitive_metrics["tree_of_thoughts_metrics"] = tree_stats
            except Exception as e:
                logger.warning(f"Failed to get Tree of Thoughts metrics: {e}")

        if self.graph_agent:
            try:
                graph_stats = self.graph_agent.get_statistics()
                cognitive_metrics["graph_of_thoughts_metrics"] = graph_stats
            except Exception as e:
                logger.warning(f"Failed to get Graph of Thoughts metrics: {e}")

        if self.constitutional_agent:
            try:
                constitutional_stats = self.constitutional_agent.get_statistics()
                cognitive_metrics["constitutional_ai_metrics"] = constitutional_stats
            except Exception as e:
                logger.warning(f"Failed to get Constitutional AI metrics: {e}")

        return {**base_metrics, **cognitive_metrics}

    def _calculate_reasoning_depth(self) -> int:
        """Calculate the reasoning depth based on available components"""
        depth = 0
        if self.tree_agent:
            depth += 1
        if self.graph_agent:
            depth += 1
        if self.constitutional_agent:
            depth += 1
        if self.meta_agent:
            depth += 1
        return depth

    def _calculate_adaptation_score(self) -> float:
        """Calculate adaptation score based on component availability"""
        available_components = sum(
            [
                self.tree_agent is not None,
                self.graph_agent is not None,
                self.constitutional_agent is not None,
                self.meta_agent is not None,
            ]
        )
        return min(1.0, available_components / 4.0)

    def _calculate_confidence_calibration(self) -> float:
        """Calculate confidence calibration score"""
        # This would typically be based on historical performance
        # For now, return a default value
        return 0.8

    async def cleanup(self) -> None:
        """Cleanup all cognitive agent resources"""
        try:
            # Cleanup base agent
            await super().cleanup()

            # Cleanup reasoning components
            # Tree and Graph agents have reset methods
            if self.tree_agent and hasattr(self.tree_agent, 'reset'):
                await self.tree_agent.reset()

            if self.graph_agent and hasattr(self.graph_agent, 'reset'):
                await self.graph_agent.reset()

            # Constitutional and Meta agents may not have reset methods
            # Just set them to None to free resources
            if self.constitutional_agent:
                self.constitutional_agent = None

            if self.meta_agent:
                self.meta_agent = None

            logger.info("Cognitive agent cleanup completed")

        except Exception as e:
            logger.error(f"Error during cognitive agent cleanup: {e}")


# Export main class
__all__ = ["CognitiveAgent"]

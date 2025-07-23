"""
Meta-Reasoning Implementation for LlamaAgent

This module provides adaptive strategy selection, confidence assessment,
and self-aware reasoning capabilities that optimize the choice between
different cognitive architectures based on problem characteristics.

Key Features:
- Dynamic reasoning strategy selection
- Confidence calibration and uncertainty quantification
- Adaptive learning from past performance
- Problem complexity analysis
- Strategy performance tracking

Author: LlamaAgent Development Team
"""

import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..llm import LLMMessage

logger = logging.getLogger(__name__)


class ReasoningStrategy(Enum):
    """Available reasoning strategies"""

    SIMPLE = "simple"
    TREE_OF_THOUGHTS = "tree_of_thoughts"
    GRAPH_OF_THOUGHTS = "graph_of_thoughts"
    CONSTITUTIONAL = "constitutional"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


class ProblemComplexity(Enum):
    """Problem complexity levels"""

    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    HIGHLY_COMPLEX = "highly_complex"


@dataclass
class StrategyPerformance:
    """Performance metrics for a reasoning strategy"""

    strategy: ReasoningStrategy

    # Performance metrics
    success_rate: float = 0.0
    average_confidence: float = 0.0
    average_execution_time: float = 0.0
    token_usage: int = 0

    # Usage statistics
    total_uses: int = 0
    successful_uses: int = 0
    failed_uses: int = 0

    # Complexity-specific performance
    performance_by_complexity: Dict[ProblemComplexity, float] = field(
        default_factory=dict
    )

    # Recent performance (for adaptation)
    recent_scores: List[float] = field(default_factory=list)
    max_recent_scores: int = 20

    def update_performance(
        self,
        success: bool,
        confidence: float,
        execution_time: float,
        tokens: int,
        complexity: ProblemComplexity,
    ):
        """Update performance metrics"""
        self.total_uses += 1

        if success:
            self.successful_uses += 1
        else:
            self.failed_uses += 1

        # Update averages
        self.success_rate = self.successful_uses / self.total_uses

        # Update average confidence with decay
        alpha = 0.1  # Learning rate
        self.average_confidence = (
            1 - alpha
        ) * self.average_confidence + alpha * confidence

        # Update average execution time with decay
        self.average_execution_time = (
            1 - alpha
        ) * self.average_execution_time + alpha * execution_time

        self.token_usage += tokens

        # Update complexity-specific performance
        if complexity not in self.performance_by_complexity:
            self.performance_by_complexity[complexity] = confidence
        else:
            self.performance_by_complexity[complexity] = (
                1 - alpha
            ) * self.performance_by_complexity[complexity] + alpha * confidence

        # Update recent scores
        self.recent_scores.append(confidence if success else 0.0)
        if len(self.recent_scores) > self.max_recent_scores:
            self.recent_scores.pop(0)

    def get_recent_performance_trend(self) -> float:
        """Get trend of recent performance (-1 to 1, negative means declining)"""
        if len(self.recent_scores) < 5:
            return 0.0

        # Simple linear regression on recent scores
        n = len(self.recent_scores)
        x = list(range(n))
        y = self.recent_scores

        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y, strict=False))
        sum_x2 = sum(xi * xi for xi in x)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

        # Normalize to [-1, 1] range
        return max(-1.0, min(1.0, slope * 10))

    def get_complexity_score(self, complexity: ProblemComplexity) -> float:
        """Get performance score for specific complexity level"""
        return self.performance_by_complexity.get(complexity, self.success_rate * 0.5)


@dataclass
class ProblemAnalysis:
    """Analysis of problem characteristics"""

    problem_text: str
    complexity: ProblemComplexity = ProblemComplexity.MODERATE

    # Problem characteristics
    requires_multi_step: bool = False
    requires_creativity: bool = False
    requires_factual_knowledge: bool = False
    requires_mathematical_reasoning: bool = False
    requires_ethical_consideration: bool = False
    has_ambiguity: bool = False
    has_constraints: bool = False

    # Analysis metrics
    text_length: int = 0
    sentence_count: int = 0
    question_count: int = 0
    keyword_density: float = 0.0

    # Confidence in analysis
    analysis_confidence: float = 0.0

    def __post_init__(self):
        """Calculate derived metrics"""
        self.text_length = len(self.problem_text)
        self.sentence_count = len(
            [s for s in self.problem_text.split('.') if s.strip()]
        )
        self.question_count = self.problem_text.count('?')


class ComplexityAnalyzer:
    """Analyzes problem complexity and characteristics"""

    def __init__(self, llm_provider: Any):
        self.llm_provider = llm_provider

        self.analysis_prompt = """
Analyze the following problem to determine its complexity and characteristics.

Problem: {problem_text}

Provide analysis as JSON:
{{
  "complexity": "trivial/simple/moderate/complex/highly_complex",
  "requires_multi_step": true/false,
  "requires_creativity": true/false,
  "requires_factual_knowledge": true/false,
  "requires_mathematical_reasoning": true/false,
  "requires_ethical_consideration": true/false,
  "has_ambiguity": true/false,
  "has_constraints": true/false,
  "reasoning": "explanation of complexity assessment",
  "confidence": 0.0-1.0
}}

Consider:
- Trivial: Simple factual questions, basic arithmetic
- Simple: Single-step problems, straightforward logic
- Moderate: Multi-step reasoning, some complexity
- Complex: Advanced reasoning, multiple considerations
- Highly Complex: Research-level problems, deep analysis needed
"""

    async def analyze_problem(self, problem_text: str) -> ProblemAnalysis:
        """Analyze problem complexity and characteristics"""
        try:
            prompt = self.analysis_prompt.format(problem_text=problem_text)
            messages = [LLMMessage(role="user", content=prompt)]
            response = await self.llm_provider.complete(messages)

            analysis_data = self._parse_analysis(response.content)

            if analysis_data:
                analysis = ProblemAnalysis(
                    problem_text=problem_text,
                    complexity=ProblemComplexity(
                        analysis_data.get("complexity", "moderate")
                    ),
                    requires_multi_step=analysis_data.get("requires_multi_step", False),
                    requires_creativity=analysis_data.get("requires_creativity", False),
                    requires_factual_knowledge=analysis_data.get(
                        "requires_factual_knowledge", False
                    ),
                    requires_mathematical_reasoning=analysis_data.get(
                        "requires_mathematical_reasoning", False
                    ),
                    requires_ethical_consideration=analysis_data.get(
                        "requires_ethical_consideration", False
                    ),
                    has_ambiguity=analysis_data.get("has_ambiguity", False),
                    has_constraints=analysis_data.get("has_constraints", False),
                    analysis_confidence=float(analysis_data.get("confidence", 0.7)),
                )

                return analysis

        except Exception as e:
            logger.error(f"Problem analysis failed: {e}")

        # Fallback analysis
        return self._fallback_analysis(problem_text)

    def _parse_analysis(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse analysis response from LLM"""
        try:
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1

            if start_idx != -1 and end_idx != 0:
                json_str = content[start_idx:end_idx]
                return json.loads(json_str)
        except Exception as e:
            logger.warning(f"Failed to parse analysis: {e}")

        return None

    def _fallback_analysis(self, problem_text: str) -> ProblemAnalysis:
        """Fallback analysis using heuristics"""
        analysis = ProblemAnalysis(problem_text=problem_text)

        text_lower = problem_text.lower()

        # Basic complexity heuristics
        if len(problem_text) < 50:
            analysis.complexity = ProblemComplexity.SIMPLE
        elif len(problem_text) > 200:
            analysis.complexity = ProblemComplexity.COMPLEX

        # Characteristic detection
        analysis.requires_multi_step = any(
            word in text_lower
            for word in ["then", "next", "after", "step", "first", "second", "finally"]
        )

        analysis.requires_mathematical_reasoning = any(
            word in text_lower
            for word in [
                "calculate",
                "compute",
                "math",
                "equation",
                "formula",
                "number",
            ]
        )

        analysis.requires_ethical_consideration = any(
            word in text_lower
            for word in [
                "should",
                "ought",
                "ethical",
                "moral",
                "right",
                "wrong",
                "fair",
            ]
        )

        analysis.has_ambiguity = "?" in problem_text or "unclear" in text_lower

        analysis.analysis_confidence = 0.5  # Low confidence for fallback

        return analysis


class StrategySelector:
    """Selects optimal reasoning strategy based on problem analysis and performance history"""

    def __init__(self, llm_provider: Any):
        self.llm_provider = llm_provider
        self.complexity_analyzer = ComplexityAnalyzer(llm_provider)

        # Strategy performance tracking
        self.strategy_performances: Dict[ReasoningStrategy, StrategyPerformance] = {}

        # Initialize performance trackers
        for strategy in ReasoningStrategy:
            if strategy != ReasoningStrategy.ADAPTIVE:  # Avoid recursion
                self.strategy_performances[strategy] = StrategyPerformance(
                    strategy=strategy
                )

        # Strategy selection rules
        self.strategy_rules = {
            ProblemComplexity.TRIVIAL: [ReasoningStrategy.SIMPLE],
            ProblemComplexity.SIMPLE: [
                ReasoningStrategy.SIMPLE,
                ReasoningStrategy.CONSTITUTIONAL,
            ],
            ProblemComplexity.MODERATE: [
                ReasoningStrategy.TREE_OF_THOUGHTS,
                ReasoningStrategy.CONSTITUTIONAL,
            ],
            ProblemComplexity.COMPLEX: [
                ReasoningStrategy.TREE_OF_THOUGHTS,
                ReasoningStrategy.GRAPH_OF_THOUGHTS,
            ],
            ProblemComplexity.HIGHLY_COMPLEX: [
                ReasoningStrategy.GRAPH_OF_THOUGHTS,
                ReasoningStrategy.HYBRID,
            ],
        }

        # Exploration parameters
        self.exploration_rate = 0.1  # Epsilon for epsilon-greedy
        self.min_uses_for_confidence = 5

    async def select_strategy(
        self, problem_text: str, context: Optional[Dict[str, Any]] = None
    ) -> Tuple[ReasoningStrategy, ProblemAnalysis]:
        """Select optimal reasoning strategy for a problem"""
        # Analyze problem
        analysis = await self.complexity_analyzer.analyze_problem(problem_text)

        # Get candidate strategies for this complexity level
        candidates = self.strategy_rules.get(
            analysis.complexity, [ReasoningStrategy.SIMPLE]
        )

        # Add special case strategies based on characteristics
        if analysis.requires_ethical_consideration:
            candidates.append(ReasoningStrategy.CONSTITUTIONAL)

        if analysis.requires_creativity and analysis.complexity.value in [
            "complex",
            "highly_complex",
        ]:
            candidates.append(ReasoningStrategy.GRAPH_OF_THOUGHTS)

        if analysis.requires_multi_step:
            candidates.append(ReasoningStrategy.TREE_OF_THOUGHTS)

        # Remove duplicates and sort by preference
        candidates = list(set(candidates))

        # Select strategy using performance-based approach
        selected_strategy = self._select_best_strategy(candidates, analysis)

        logger.info(
            f"Selected strategy '{selected_strategy.value}' for {analysis.complexity.value} problem"
        )

        return selected_strategy, analysis

    def _select_best_strategy(
        self, candidates: List[ReasoningStrategy], analysis: ProblemAnalysis
    ) -> ReasoningStrategy:
        """Select best strategy from candidates using performance history"""
        if not candidates:
            return ReasoningStrategy.SIMPLE

        if len(candidates) == 1:
            return candidates[0]

        # Calculate scores for each candidate
        strategy_scores: List[Tuple[float, ReasoningStrategy]] = []

        for strategy in candidates:
            if strategy in self.strategy_performances:
                perf = self.strategy_performances[strategy]

                # Base score from complexity-specific performance
                base_score = perf.get_complexity_score(analysis.complexity)

                # Adjust for recent trends
                trend = perf.get_recent_performance_trend()
                trend_bonus = trend * 0.1

                # Exploration bonus for under-explored strategies
                if perf.total_uses < self.min_uses_for_confidence:
                    exploration_bonus = 0.2
                else:
                    exploration_bonus = 0.0

                # Efficiency penalty for slow strategies (if time is important)
                efficiency_penalty = min(0.1, perf.average_execution_time / 100.0)

                total_score = (
                    base_score + trend_bonus + exploration_bonus - efficiency_penalty
                )

                strategy_scores.append((total_score, strategy))
            else:
                # New strategy gets moderate score + exploration bonus
                strategy_scores.append((0.6, strategy))

        # Apply epsilon-greedy exploration
        import random

        if random.random() < self.exploration_rate:
            # Explore: choose randomly
            return random.choice(candidates)
        else:
            # Exploit: choose best scoring strategy
            strategy_scores.sort(key=lambda x: x[0], reverse=True)
            return strategy_scores[0][1]

    def update_strategy_performance(
        self,
        strategy: ReasoningStrategy,
        success: bool,
        confidence: float,
        execution_time: float,
        tokens_used: int,
        problem_analysis: ProblemAnalysis,
    ):
        """Update performance metrics for a strategy"""
        if strategy in self.strategy_performances:
            self.strategy_performances[strategy].update_performance(
                success=success,
                confidence=confidence,
                execution_time=execution_time,
                tokens=tokens_used,
                complexity=problem_analysis.complexity,
            )

    def get_strategy_statistics(self) -> Dict[str, Any]:
        """Get performance statistics for all strategies"""
        stats: Dict[str, Any] = {}

        for strategy, performance in self.strategy_performances.items():
            stats[strategy.value] = {
                "success_rate": performance.success_rate,
                "average_confidence": performance.average_confidence,
                "total_uses": performance.total_uses,
                "average_execution_time": performance.average_execution_time,
                "recent_trend": performance.get_recent_performance_trend(),
                "performance_by_complexity": {
                    complexity.value: score
                    for complexity, score in performance.performance_by_complexity.items()
                },
            }

        return stats


class ConfidenceSystem:
    """Advanced confidence assessment and uncertainty quantification"""

    def __init__(self, llm_provider: Any):
        self.llm_provider = llm_provider

        # Confidence calibration parameters
        self.calibration_data: List[
            Tuple[float, bool]
        ] = []  # (predicted_confidence, actual_success)
        self.calibration_window = 100

        self.confidence_prompt = """
Assess your confidence in the following response to the given problem.

Problem: {problem}
Response: {response}

Provide confidence assessment as JSON:
{{
  "confidence": 0.0-1.0,
  "uncertainty_factors": ["factor1", "factor2"],
  "confidence_reasoning": "explanation of confidence level",
  "alternative_approaches": ["approach1", "approach2"],
  "key_assumptions": ["assumption1", "assumption2"]
}}

Consider:
- Factual accuracy and completeness
- Logical consistency
- Potential edge cases or exceptions
- Availability of alternative solutions
- Complexity of the problem domain
"""

    async def assess_confidence(
        self,
        problem: str,
        response: str,
        strategy_used: ReasoningStrategy = ReasoningStrategy.SIMPLE,
    ) -> Dict[str, Any]:
        """Assess confidence in a response"""
        try:
            prompt = self.confidence_prompt.format(problem=problem, response=response)

            messages = [LLMMessage(role="user", content=prompt)]
            llm_response = await self.llm_provider.complete(messages)

            assessment = self._parse_confidence_assessment(llm_response.content)

            if assessment:
                # Apply calibration if we have enough data
                calibrated_confidence = self._apply_calibration(
                    assessment["confidence"], strategy_used
                )
                assessment["calibrated_confidence"] = calibrated_confidence
                assessment["raw_confidence"] = assessment["confidence"]
                assessment["confidence"] = calibrated_confidence

                return assessment

        except Exception as e:
            logger.error(f"Confidence assessment failed: {e}")

        # Fallback confidence assessment
        return {
            "confidence": 0.5,
            "calibrated_confidence": 0.5,
            "uncertainty_factors": ["assessment_failed"],
            "confidence_reasoning": "Unable to assess confidence due to error",
            "alternative_approaches": [],
            "key_assumptions": [],
        }

    def _parse_confidence_assessment(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse confidence assessment from LLM response"""
        try:
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1

            if start_idx != -1 and end_idx != 0:
                json_str = content[start_idx:end_idx]
                assessment = json.loads(json_str)

                # Validate confidence range
                assessment["confidence"] = max(
                    0.0, min(1.0, float(assessment.get("confidence", 0.5)))
                )

                return assessment
        except Exception as e:
            logger.warning(f"Failed to parse confidence assessment: {e}")

        return None

    def _apply_calibration(
        self, raw_confidence: float, strategy: ReasoningStrategy
    ) -> float:
        """Apply confidence calibration based on historical data"""
        if len(self.calibration_data) < 10:
            return raw_confidence  # Not enough data for calibration

        # Simple Platt scaling approach
        try:
            # Find similar confidence predictions
            similar_predictions = [
                (conf, success)
                for conf, success in self.calibration_data
                if abs(conf - raw_confidence) < 0.2
            ]

            if len(similar_predictions) >= 5:
                actual_success_rate = sum(
                    success for _, success in similar_predictions
                ) / len(similar_predictions)

                # Apply calibration with some smoothing
                calibrated = 0.7 * actual_success_rate + 0.3 * raw_confidence
                return max(0.0, min(1.0, calibrated))

        except Exception:
            pass

        return raw_confidence

    def update_calibration(self, predicted_confidence: float, actual_success: bool):
        """Update confidence calibration data"""
        self.calibration_data.append((predicted_confidence, actual_success))

        # Keep only recent data for calibration
        if len(self.calibration_data) > self.calibration_window:
            self.calibration_data.pop(0)

    def get_calibration_statistics(self) -> Dict[str, float]:
        """Get confidence calibration statistics"""
        if len(self.calibration_data) < 5:
            return {"calibration_error": 0.0, "data_points": len(self.calibration_data)}

        # Calculate Expected Calibration Error (ECE)
        bins = 10
        bin_boundaries = [i / bins for i in range(bins + 1)]
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        total_samples = len(self.calibration_data)

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=False):
            in_bin = [
                (conf, success)
                for conf, success in self.calibration_data
                if bin_lower <= conf < bin_upper
            ]

            if len(in_bin) > 0:
                avg_confidence = sum(conf for conf, _ in in_bin) / len(in_bin)
                accuracy = sum(success for _, success in in_bin) / len(in_bin)

                ece += (len(in_bin) / total_samples) * abs(avg_confidence - accuracy)

        return {
            "calibration_error": ece,
            "data_points": total_samples,
            "avg_confidence": statistics.mean(
                conf for conf, _ in self.calibration_data
            ),
            "success_rate": statistics.mean(
                success for _, success in self.calibration_data
            ),
        }


class MetaCognitiveAgent:
    """Meta-cognitive agent that orchestrates reasoning strategy selection and performance monitoring"""

    def __init__(self, llm_provider: Any):
        self.llm_provider = llm_provider

        # Core components
        self.strategy_selector = StrategySelector(llm_provider)
        self.confidence_system = ConfidenceSystem(llm_provider)

        # Meta-learning parameters
        self.adaptation_rate = 0.05
        self.performance_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000

        # Statistics
        self.stats = {
            "total_decisions": 0,
            "successful_decisions": 0,
            "strategy_switches": 0,
            "average_confidence": 0.0,
            "calibration_accuracy": 0.0,
        }

    async def select_and_execute_strategy(
        self,
        problem: str,
        available_strategies: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Select optimal strategy and coordinate execution"""
        start_time = time.time()

        # Phase 1: Strategy Selection
        (
            selected_strategy,
            problem_analysis,
        ) = await self.strategy_selector.select_strategy(problem, context)

        # Phase 2: Execute selected strategy
        if selected_strategy.value in available_strategies:
            strategy_executor = available_strategies[selected_strategy.value]

            try:
                execution_result = await strategy_executor(problem, context or {})
                success = True
                response_content = execution_result.get(
                    "solution", execution_result.get("answer", "")
                )
            except Exception as e:
                logger.error(f"Strategy execution failed: {e}")
                success = False
                response_content = f"Strategy execution failed: {e}"
                execution_result = {"error": str(e)}
        else:
            # Fallback to simple execution
            success = False
            response_content = f"Strategy {selected_strategy.value} not available"
            execution_result = {"error": "Strategy not implemented"}

        # Phase 3: Confidence Assessment
        confidence_assessment = await self.confidence_system.assess_confidence(
            problem, response_content, selected_strategy
        )

        execution_time = time.time() - start_time

        # Phase 4: Performance Update
        tokens_used = int(
            execution_result.get("tokens_used", len(response_content) // 4)
        )
        final_confidence = confidence_assessment["confidence"]

        self.strategy_selector.update_strategy_performance(
            selected_strategy,
            success,
            final_confidence,
            execution_time,
            tokens_used,
            problem_analysis,
        )

        self.confidence_system.update_calibration(final_confidence, success)

        # Update meta-statistics
        self._update_meta_statistics(success, final_confidence)

        # Store performance history
        self._record_performance(
            problem_analysis,
            selected_strategy,
            success,
            final_confidence,
            execution_time,
            execution_result,
        )

        return {
            "selected_strategy": selected_strategy.value,
            "problem_analysis": {
                "complexity": problem_analysis.complexity.value,
                "characteristics": {
                    "requires_multi_step": problem_analysis.requires_multi_step,
                    "requires_creativity": problem_analysis.requires_creativity,
                    "requires_factual_knowledge": problem_analysis.requires_factual_knowledge,
                    "requires_mathematical_reasoning": problem_analysis.requires_mathematical_reasoning,
                    "requires_ethical_consideration": problem_analysis.requires_ethical_consideration,
                    "has_ambiguity": problem_analysis.has_ambiguity,
                    "has_constraints": problem_analysis.has_constraints,
                },
            },
            "execution_result": execution_result,
            "confidence_assessment": confidence_assessment,
            "execution_time": execution_time,
            "success": success,
            "meta_statistics": self.get_meta_statistics(),
        }

    def _update_meta_statistics(self, success: bool, confidence: float):
        """Update meta-cognitive statistics"""
        self.stats["total_decisions"] += 1

        if success:
            self.stats["successful_decisions"] += 1

        # Update average confidence with decay
        alpha = 0.1
        self.stats["average_confidence"] = (1 - alpha) * self.stats[
            "average_confidence"
        ] + alpha * confidence

        # Update success rate
        if self.stats["total_decisions"] > 0:
            success_rate = (
                self.stats["successful_decisions"] / self.stats["total_decisions"]
            )
            # Calibration accuracy is how close average confidence is to success rate
            self.stats["calibration_accuracy"] = 1.0 - abs(
                self.stats["average_confidence"] - success_rate
            )

    def _record_performance(
        self,
        problem_analysis: ProblemAnalysis,
        strategy: ReasoningStrategy,
        success: bool,
        confidence: float,
        execution_time: float,
        execution_result: Dict[str, Any],
    ):
        """Record performance for meta-learning"""
        record = {
            "timestamp": time.time(),
            "problem_complexity": problem_analysis.complexity.value,
            "problem_characteristics": {
                "requires_multi_step": problem_analysis.requires_multi_step,
                "requires_creativity": problem_analysis.requires_creativity,
                "requires_factual_knowledge": problem_analysis.requires_factual_knowledge,
                "requires_mathematical_reasoning": problem_analysis.requires_mathematical_reasoning,
                "requires_ethical_consideration": problem_analysis.requires_ethical_consideration,
            },
            "selected_strategy": strategy.value,
            "success": success,
            "confidence": confidence,
            "execution_time": execution_time,
            "tokens_used": execution_result.get("tokens_used", 0),
        }

        self.performance_history.append(record)

        # Keep history within limits
        if len(self.performance_history) > self.max_history_size:
            self.performance_history.pop(0)

    def get_meta_statistics(self) -> Dict[str, Any]:
        """Get comprehensive meta-cognitive statistics"""
        strategy_stats = self.strategy_selector.get_strategy_statistics()
        confidence_stats = self.confidence_system.get_calibration_statistics()

        return {
            "meta_agent_stats": self.stats,
            "strategy_performance": strategy_stats,
            "confidence_calibration": confidence_stats,
            "performance_history_size": len(self.performance_history),
        }

    def analyze_performance_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in performance history for insights"""
        if len(self.performance_history) < 10:
            return {"insufficient_data": True}

        # Analyze performance by complexity
        complexity_performance: Dict[str, List[bool]] = {}
        for record in self.performance_history:
            complexity: str = record["problem_complexity"]
            if complexity not in complexity_performance:
                complexity_performance[complexity] = []
            complexity_performance[complexity].append(record["success"])

        complexity_analysis: Dict[str, Dict[str, Any]] = {
            complexity: {
                "success_rate": statistics.mean(successes),
                "sample_count": len(successes),
            }
            for complexity, successes in complexity_performance.items()
        }

        # Analyze strategy effectiveness trends
        recent_performance = self.performance_history[-50:]  # Last 50 decisions
        strategy_trends: Dict[str, List[float]] = {}

        for record in recent_performance:
            strategy: str = record["selected_strategy"]
            if strategy not in strategy_trends:
                strategy_trends[strategy] = []
            strategy_trends[strategy].append(record["confidence"])

        trend_analysis: Dict[str, Dict[str, Any]] = {
            strategy: {
                "average_confidence": statistics.mean(confidences),
                "confidence_trend": (
                    "improving"
                    if len(confidences) > 5
                    and statistics.mean(confidences[-5:])
                    > statistics.mean(confidences[:5])
                    else "stable"
                ),
            }
            for strategy, confidences in strategy_trends.items()
            if len(confidences) >= 3
        }

        return {
            "complexity_analysis": complexity_analysis,
            "strategy_trends": trend_analysis,
            "overall_improvement": self._calculate_improvement_trend(),
        }

    def _calculate_improvement_trend(self) -> Dict[str, float]:
        """Calculate overall performance improvement trend"""
        if len(self.performance_history) < 20:
            return {"trend": 0.0, "confidence": 0.0}

        # Split into early and recent periods
        mid_point = len(self.performance_history) // 2
        early_period = self.performance_history[:mid_point]
        recent_period = self.performance_history[mid_point:]

        early_success = statistics.mean(r["success"] for r in early_period)
        recent_success = statistics.mean(r["success"] for r in recent_period)

        early_confidence = statistics.mean(r["confidence"] for r in early_period)
        recent_confidence = statistics.mean(r["confidence"] for r in recent_period)

        return {
            "success_rate_trend": recent_success - early_success,
            "confidence_trend": recent_confidence - early_confidence,
            "overall_trend": (recent_success + recent_confidence)
            - (early_success + early_confidence),
        }

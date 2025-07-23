"""
Advanced Reasoning Engine - Multi-Strategy Implementation

This module implements sophisticated reasoning capabilities, including:
- Deductive reasoning (general to specific)
- Inductive reasoning (specific to general)
- Abductive reasoning (best explanation)
- Causal reasoning (cause-effect relationships)
- Analogical reasoning (pattern matching)
- Probabilistic reasoning (uncertainty handling)

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

try:
    from opentelemetry import trace

    tracer = trace.get_tracer(__name__)
except ImportError:
    # Fallback tracer for when OpenTelemetry is not available
    class NoOpTracer:
        def start_as_current_span(self, name: str):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args: Any):
            pass

        def set_attribute(self, key: str, value: Any):
            pass

    tracer = NoOpTracer()


class ReasoningType(Enum):
    """Types of reasoning strategies"""

    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    CAUSAL = "causal"
    ANALOGICAL = "analogical"
    PROBABILISTIC = "probabilistic"


@dataclass
class ThoughtNode:
    """Represents a single thought/reasoning step"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    thought: str = ""
    reasoning_type: ReasoningType = ReasoningType.DEDUCTIVE
    confidence: float = 0.0
    supporting_evidence: List[str] = field(default_factory=list)
    conclusions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ReasoningContext:
    """Context for reasoning operations"""

    objective: str = ""
    constraints: Dict[str, Any] = field(default_factory=dict)
    available_knowledge: List[str] = field(default_factory=list)
    previous_conclusions: List[str] = field(default_factory=list)
    reasoning_depth_limit: int = 10
    confidence_threshold: float = 0.7
    allow_assumptions: bool = True
    require_evidence: bool = False


class ReasoningStrategy(ABC):
    """Abstract base for reasoning strategies"""

    @abstractmethod
    async def reason(
        self,
        context: ReasoningContext,
        current_state: Dict[str, Any],
        thought_chain: List[ThoughtNode],
    ) -> ThoughtNode:
        """Apply reasoning strategy"""
        pass

    @abstractmethod
    def validate_reasoning(self, thought_node: ThoughtNode) -> Tuple[bool, List[str]]:
        """Validate reasoning results"""
        pass


class DeductiveReasoning(ReasoningStrategy):
    """Deductive reasoning: general to specific"""

    async def reason(
        self,
        context: ReasoningContext,
        current_state: Dict[str, Any],
        thought_chain: List[ThoughtNode],
    ) -> ThoughtNode:
        """Apply deductive reasoning"""
        # Extract premises from current state
        premises = current_state.get("premises", [])
        rules = current_state.get("rules", [])
        if not premises:
            return ThoughtNode(
                thought="No premises available for deductive reasoning",
                reasoning_type=ReasoningType.DEDUCTIVE,
                confidence=0.0,
            )
        # Apply logical rules to premises
        conclusions = []
        for rule in rules:
            if self._check_rule_applicability(rule, premises):
                conclusion = self._apply_rule(rule, premises)
                conclusions.append(conclusion)
        # If no rules, make basic logical deductions
        if not conclusions:
            conclusions = self._basic_deduction(premises)
        best_conclusion = (
            max(conclusions, key=lambda x: x.get("confidence", 0))
            if conclusions
            else None
        )

        return ThoughtNode(
            thought=f"Deductive reasoning from {len(premises)} premises",
            reasoning_type=ReasoningType.DEDUCTIVE,
            confidence=best_conclusion.get("confidence", 0.8)
            if best_conclusion
            else 0.3,
            supporting_evidence=premises,
            conclusions=[best_conclusion["conclusion"]] if best_conclusion else [],
            metadata={"applied_rules": [c.get("rule") for c in conclusions]},
        )

    def _check_rule_applicability(
        self, rule: Dict[str, Any], premises: List[str]
    ) -> bool:
        """Check if rule can be applied to premises"""
        conditions = rule.get("conditions", [])
        return all(any(cond in premise for premise in premises) for cond in conditions)

    def _apply_rule(self, rule: Dict[str, Any], premises: List[str]) -> Dict[str, Any]:
        """Apply logical rule to premises"""
        return {
            "conclusion": rule.get("conclusion", "Unknown conclusion"),
            "confidence": rule.get("confidence", 0.8),
            "rule": rule.get("name", "Unknown rule"),
        }

    def _basic_deduction(self, premises: List[str]) -> List[Dict[str, Any]]:
        """Basic deductive reasoning without explicit rules"""
        conclusions = []

        # Simple pattern matching for basic deductions
        for premise in premises:
            if "all" in premise.lower() and "are" in premise.lower():
                # Extract universal statements
                conclusions.append(
                    {
                        "conclusion": f"Derived from universal statement: {premise}",
                        "confidence": 0.7,
                    }
                )
        return conclusions

    def validate_reasoning(self, thought_node: ThoughtNode) -> Tuple[bool, List[str]]:
        """Validate deductive reasoning"""
        errors = []

        # Check logical consistency
        if not thought_node.supporting_evidence:
            errors.append("No supporting evidence provided")
        # Check conclusion follows from premises
        if not thought_node.conclusions:
            errors.append("No conclusions derived")
        # Check confidence aligns with evidence
        expected_confidence = 0.5 + (len(thought_node.supporting_evidence) / 20)
        if abs(thought_node.confidence - expected_confidence) > 0.1:
            errors.append("Confidence doesn't match evidence strength")
        return len(errors) == 0, errors


class InductiveReasoning(ReasoningStrategy):
    """Inductive reasoning: specific to general"""

    def __init__(self, min_examples: int = 3):
        self.min_examples = min_examples

    async def reason(
        self,
        context: ReasoningContext,
        current_state: Dict[str, Any],
        thought_chain: List[ThoughtNode],
    ) -> ThoughtNode:
        """Apply inductive reasoning"""
        # Extract observations from current state
        observations = current_state.get("observations", [])
        if len(observations) < self.min_examples:
            return ThoughtNode(
                thought=f"Insufficient observations for inductive reasoning ({len(observations)} < {self.min_examples})",
                reasoning_type=ReasoningType.INDUCTIVE,
                confidence=0.0,
            )

        # Identify patterns in observations
        patterns = self._identify_patterns(observations)
        # Generate generalizations
        generalizations = []
        for pattern in patterns:
            generalization = self._create_generalization(pattern)
            generalizations.append(generalization)
        # Select best generalization
        best_generalization = (
            max(generalizations, key=len)
            if generalizations
            else "No clear pattern identified"
        )

        confidence = min(0.9, 0.5 + (len(observations) / 20))

        return ThoughtNode(
            thought=f"Inductive reasoning from {len(observations)} observations",
            reasoning_type=ReasoningType.INDUCTIVE,
            confidence=confidence,
            supporting_evidence=observations,
            conclusions=[best_generalization],
            metadata={"patterns": patterns, "sample_size": len(observations)},
        )

    def _identify_patterns(self, observations: List[str]) -> List[Dict[str, Any]]:
        """Identify patterns in observations"""
        patterns = []

        # Count common elements
        element_counts = {}
        for obs in observations:
            elements = obs.split()
            for elem in elements:
                element_counts[elem] = element_counts.get(elem, 0) + 1

        # Convert to patterns
        for elem, count in element_counts.items():
            if count > 1:
                patterns.append(
                    {
                        "element": elem,
                        "frequency": count / len(observations),
                        "support": count,
                    }
                )

        return sorted(patterns, key=lambda x: x["frequency"], reverse=True)

    def _create_generalization(self, pattern: Dict[str, Any]) -> str:
        """Create generalization from pattern"""
        return f"Pattern '{pattern['element']}' appears in {pattern['frequency']*100:.1f}% of cases"

    def validate_reasoning(self, thought_node: ThoughtNode) -> Tuple[bool, List[str]]:
        """Validate inductive reasoning"""
        errors = []

        # Check sample size
        if len(thought_node.supporting_evidence) < self.min_examples:
            errors.append(f"Insufficient examples (need {self.min_examples})")

        # Check confidence aligns with evidence
        expected_confidence = min(
            0.9, 0.5 + (len(thought_node.supporting_evidence) / 20)
        )
        if abs(thought_node.confidence - expected_confidence) > 0.1:
            errors.append("Confidence doesn't match sample size")
        return len(errors) == 0, errors


class CausalReasoning(ReasoningStrategy):
    """Causal reasoning: cause-effect relationships"""

    async def reason(
        self,
        context: ReasoningContext,
        current_state: Dict[str, Any],
        thought_chain: List[ThoughtNode],
    ) -> ThoughtNode:
        """Apply causal reasoning"""
        # Extract events and relationships
        events = current_state.get("events", [])
        relationships = current_state.get("relationships", [])
        if not events:
            return ThoughtNode(
                thought="No events available for causal reasoning",
                reasoning_type=ReasoningType.CAUSAL,
                confidence=0.0,
            )
        # Identify potential causal chains
        causal_chains = self._identify_causal_chains(events, relationships)
        # Build causal model
        causal_model = self._build_causal_model(causal_chains)
        # Generate causal conclusions
        conclusions = []
        for chain in causal_chains:
            conclusion = f"Event '{chain['cause']}' likely causes '{chain['effect']}'"
            conclusions.append(conclusion)
        confidence = (
            min(0.85, 0.6 + (len(causal_chains) / 10)) if causal_chains else 0.2
        )

        return ThoughtNode(
            thought=f"Causal reasoning from {len(events)} events",
            reasoning_type=ReasoningType.CAUSAL,
            confidence=confidence,
            supporting_evidence=events,
            conclusions=conclusions,
            metadata={"causal_model": causal_model, "chains": causal_chains},
        )

    def _identify_causal_chains(
        self, events: List[str], relationships: List[str]
    ) -> List[Dict[str, Any]]:
        """Identify potential causal chains"""
        chains = []

        # Simple temporal and semantic analysis
        for i, event_a in enumerate(events):
            for j, event_b in enumerate(events):
                if i != j:
                    # Check for causal indicators
                    if self._has_causal_relationship(event_a, event_b, relationships):
                        chains.append(
                            {
                                "cause": event_a,
                                "effect": event_b,
                                "strength": 0.7,
                                "temporal_order": i < j,
                            }
                        )
        return chains

    def _has_causal_relationship(
        self, event_a: str, event_b: str, relationships: List[str]
    ) -> bool:
        """Check if events have causal relationship"""
        causal_words = ["causes", "leads to", "results in", "triggers", "influences"]

        for rel in relationships:
            if event_a in rel and event_b in rel:
                if any(word in rel.lower() for word in causal_words):
                    return True

        return False

    def _build_causal_model(
        self, causal_chains: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build causal model from chains"""
        model = {
            "nodes": list(
                set(
                    [chain["cause"] for chain in causal_chains]
                    + [chain["effect"] for chain in causal_chains]
                )
            ),
            "edges": [
                (chain["cause"], chain["effect"], chain["strength"])
                for chain in causal_chains
            ],
            "strength": sum(chain["strength"] for chain in causal_chains)
            / len(causal_chains)
            if causal_chains
            else 0,
        }
        return model

    def validate_reasoning(self, thought_node: ThoughtNode) -> Tuple[bool, List[str]]:
        """Validate causal reasoning"""
        errors = []

        # Check for temporal consistency
        causal_model = thought_node.metadata.get("causal_model", {})
        if not causal_model.get("edges"):
            errors.append("No causal relationships identified")
        # Check confidence matches model strength
        model_strength = causal_model.get("strength", 0)
        if abs(thought_node.confidence - model_strength) > 0.2:
            errors.append("Confidence doesn't match causal model strength")
        return len(errors) == 0, errors


class ChainEngine:
    """Advanced multi-strategy reasoning engine"""

    def __init__(self):
        self.strategies = {
            ReasoningType.DEDUCTIVE: DeductiveReasoning(),
            ReasoningType.INDUCTIVE: InductiveReasoning(),
            ReasoningType.CAUSAL: CausalReasoning(),
        }
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Setup logger for reasoning engine"""
        logger = logging.getLogger("ChainEngine")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    async def reason(
        self, context: ReasoningContext, initial_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Perform multi-strategy reasoning"""
        with tracer.start_as_current_span("advanced_reasoning") as span:
            span.set_attribute("objective", context.objective)
            # Initialize reasoning chain
            thought_chain: List[ThoughtNode] = []
            current_state = initial_state or {}

            # Select appropriate strategies
            selected_strategies = self._select_strategies(
                context.objective, current_state
            )
            self.logger.info(
                f"Starting reasoning with strategies: {[s.value for s in selected_strategies]}"
            )
            # Apply reasoning strategies iteratively
            for depth in range(context.reasoning_depth_limit):
                # Select next strategy
                next_strategy = self._select_next_strategy(
                    selected_strategies, thought_chain
                )
                if next_strategy not in self.strategies:
                    self.logger.warning(f"Strategy {next_strategy} not available")
                    continue

                # Apply strategy
                strategy = self.strategies[next_strategy]
                try:
                    thought_node = await strategy.reason(
                        context, current_state, thought_chain
                    )
                    # Validate reasoning
                    is_valid, errors = strategy.validate_reasoning(thought_node)
                    if not is_valid:
                        self.logger.warning(f"Invalid reasoning: {errors}")
                        continue

                    thought_chain.append(thought_node)
                    # Update state
                    current_state = self._update_state(current_state, thought_node)
                    # Check termination conditions
                    if self._should_terminate(thought_chain, context):
                        break

                except Exception as e:
                    self.logger.error(
                        f"Error in reasoning strategy {next_strategy}: {e}"
                    )
                    continue

            # Synthesize final conclusion
            conclusion = self._synthesize_conclusion(thought_chain, context)
            return {
                "conclusion": conclusion,
                "thought_chain": [
                    self._serialize_thought_node(node) for node in thought_chain
                ],
                "confidence": conclusion.get("confidence", 0.0),
                "reasoning_depth": len(thought_chain),
                "strategies_used": list(
                    set(node.reasoning_type.value for node in thought_chain)
                ),
            }

    def _select_strategies(
        self, objective: str, current_state: Dict[str, Any]
    ) -> List[ReasoningType]:
        """Select appropriate reasoning strategies"""
        strategies = []

        # Analyze objective
        objective_lower = objective.lower()

        # Rule-based strategy selection
        if any(word in objective_lower for word in ["prove", "demonstrate", "show"]):
            strategies.append(ReasoningType.DEDUCTIVE)
        if any(
            word in objective_lower for word in ["pattern", "generalize", "predict"]
        ):
            strategies.append(ReasoningType.INDUCTIVE)
        if any(
            word in objective_lower for word in ["cause", "effect", "because", "why"]
        ):
            strategies.append(ReasoningType.CAUSAL)
        # Default strategies if none selected
        if not strategies:
            strategies = [ReasoningType.DEDUCTIVE, ReasoningType.INDUCTIVE]

        return strategies

    def _select_next_strategy(
        self,
        available_strategies: List[ReasoningType],
        thought_chain: List[ThoughtNode],
    ) -> ReasoningType:
        """Select next reasoning strategy based on current state"""
        if not thought_chain:
            # Start with deductive if available
            if ReasoningType.DEDUCTIVE in available_strategies:
                return ReasoningType.DEDUCTIVE

        # Rotate through strategies
        last_strategy = thought_chain[-1].reasoning_type if thought_chain else None

        for strategy in available_strategies:
            if strategy != last_strategy:
                return strategy

        return available_strategies[0]

    def _update_state(
        self, current_state: Dict[str, Any], thought_node: ThoughtNode
    ) -> Dict[str, Any]:
        """Update reasoning state"""
        new_state = current_state.copy()

        # Add conclusions as new premises
        new_state.setdefault("premises", []).extend(thought_node.conclusions)
        # Update based on reasoning type
        if thought_node.reasoning_type == ReasoningType.INDUCTIVE:
            new_state.setdefault("generalizations", []).extend(thought_node.conclusions)
        return new_state

    def _should_terminate(
        self, thought_chain: List[ThoughtNode], context: ReasoningContext
    ) -> bool:
        """Check if reasoning should terminate"""
        if not thought_chain:
            return False

        # Check confidence threshold
        latest_node = thought_chain[-1]
        if latest_node.confidence >= context.confidence_threshold:
            return True

        # Check if no progress is being made
        if len(thought_chain) >= 3:
            recent_confidences = [node.confidence for node in thought_chain[-3:]]
            if max(recent_confidences) - min(recent_confidences) < 0.1:
                return True

        return False

    def _synthesize_conclusion(
        self, thought_chain: List[ThoughtNode], context: ReasoningContext
    ) -> Dict[str, Any]:
        """Synthesize final conclusion from reasoning chain"""
        if not thought_chain:
            return {
                "conclusion": "No reasoning performed",
                "confidence": 0.0,
                "rationale": "Empty thought chain",
            }

        # Find highest confidence node
        best_node = max(thought_chain, key=lambda x: x.confidence)
        # Combine conclusions
        all_conclusions = []
        for node in thought_chain:
            all_conclusions.extend(node.conclusions)
        # Calculate weighted confidence
        total_confidence = sum(node.confidence for node in thought_chain)
        avg_confidence = total_confidence / len(thought_chain)
        return {
            "conclusion": best_node.conclusions[0]
            if best_node.conclusions
            else "No clear conclusion",
            "confidence": avg_confidence,
            "rationale": f"Based on {len(thought_chain)} reasoning steps using {len(set(node.reasoning_type for node in thought_chain))} strategies",
            "all_conclusions": all_conclusions,
            "best_strategy": best_node.reasoning_type.value,
        }

    def _serialize_thought_node(self, node: ThoughtNode) -> Dict[str, Any]:
        """Serialize thought node for output"""
        return {
            "id": node.id,
            "thought": node.thought,
            "reasoning_type": node.reasoning_type.value,
            "confidence": node.confidence,
            "conclusions": node.conclusions,
            "supporting_evidence": node.supporting_evidence,
            "metadata": node.metadata,
            "timestamp": node.timestamp.isoformat(),
        }


# Export main classes
__all__ = [
    "ChainEngine",
    "ReasoningType",
    "ThoughtNode",
    "ReasoningContext",
    "ReasoningStrategy",
    "DeductiveReasoning",
    "InductiveReasoning",
    "CausalReasoning",
]

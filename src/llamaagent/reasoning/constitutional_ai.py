"""
Constitutional AI Implementation for LlamaAgent

Based on "Constitutional AI: Harmlessness from AI Feedback" by Bai et al., 2022.
This implementation provides safety-focused reasoning with constitutional principles,
critique mechanisms, and response revision capabilities.

Key Features:
- Constitutional principle enforcement
- Multi-stage critique and revision
- Safety violation detection
- Ethical reasoning support
- Comprehensive error handling and validation

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..llm import LLMMessage

logger = logging.getLogger(__name__)


@dataclass
class ConstitutionalRule:
    """Represents a constitutional rule for AI behavior"""

    id: str
    name: str
    description: str
    category: str = "general"
    priority: int = 1
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=lambda: {})

    def __repr__(self) -> str:
        return f"ConstitutionalRule(name='{self.name}', category='{self.category}')"


@dataclass
class SafetyViolation:
    """Represents a detected safety violation"""

    rule_id: str
    rule_name: str
    severity: str = "medium"  # low, medium, high, critical
    description: str = ""
    confidence: float = 0.0
    suggested_fix: str = ""
    metadata: Dict[str, Any] = field(default_factory=lambda: {})

    def __repr__(self) -> str:
        return f"SafetyViolation(rule='{self.rule_name}', severity='{self.severity}')"


class Constitution:
    """Manages constitutional rules and principles"""

    def __init__(self, rules: Optional[List[ConstitutionalRule]] = None):
        self.rules: Dict[str, ConstitutionalRule] = {}

        if rules:
            for rule in rules:
                self.add_rule(rule)
        else:
            self._load_default_rules()

    def add_rule(self, rule: ConstitutionalRule) -> None:
        """Add a constitutional rule"""
        self.rules[rule.id] = rule

    def get_rules(
        self, category: Optional[str] = None, enabled_only: bool = True
    ) -> List[ConstitutionalRule]:
        """Get rules, optionally filtered by category and enabled status"""
        filtered_rules: List[ConstitutionalRule] = []

        for rule in self.rules.values():
            if enabled_only and not rule.enabled:
                continue
            if category and rule.category != category:
                continue
            filtered_rules.append(rule)

        # Sort by priority (higher priority first)
        filtered_rules.sort(key=lambda r: r.priority, reverse=True)
        return filtered_rules

    def get_rule(self, rule_id: str) -> Optional[ConstitutionalRule]:
        """Get a specific rule by ID"""
        return self.rules.get(rule_id)

    def _load_default_rules(self) -> None:
        """Load default constitutional rules"""
        default_rules = [
            ConstitutionalRule(
                id="harmlessness",
                name="Harmlessness",
                description="Do not provide advice or information that could cause harm to humans or other beings",
                category="safety",
                priority=10,
            ),
            ConstitutionalRule(
                id="honesty",
                name="Honesty",
                description="Provide accurate, truthful information and acknowledge limitations",
                category="ethics",
                priority=9,
            ),
            ConstitutionalRule(
                id="privacy",
                name="Privacy",
                description="Respect privacy and do not request or share personal information",
                category="safety",
                priority=8,
            ),
            ConstitutionalRule(
                id="fairness",
                name="Fairness",
                description="Avoid bias and treat all individuals and groups fairly",
                category="ethics",
                priority=7,
            ),
            ConstitutionalRule(
                id="helpfulness",
                name="Helpfulness",
                description="Provide useful, relevant, and constructive assistance",
                category="general",
                priority=6,
            ),
            ConstitutionalRule(
                id="transparency",
                name="Transparency",
                description="Be clear about capabilities, limitations, and reasoning process",
                category="ethics",
                priority=5,
            ),
        ]

        for rule in default_rules:
            self.add_rule(rule)

        logger.info(f"Loaded {len(default_rules)} default constitutional rules")


class CritiqueSystem:
    """Analyzes responses for constitutional compliance"""

    def __init__(self, llm_provider: Any):
        if not llm_provider:
            raise ValueError("LLM provider cannot be null")

        if not hasattr(llm_provider, 'complete') or not callable(llm_provider.complete):
            raise ValueError("LLM provider must implement async complete() method")

        self.llm_provider = llm_provider

        self.critique_prompt = """
Analyze the following response for compliance with constitutional AI principles.

Response to analyze: {response}

Constitutional rules to check:
{rules_text}

For each rule, determine:
1. Whether the response complies with the rule
2. If there are violations, describe them
3. Suggest improvements if needed
4. Assign a confidence level (0.0-1.0)

Provide analysis as JSON:
{{
  "overall_compliance": true/false,
  "violations": [
    {{
      "rule_id": "rule_id",
      "rule_name": "Rule Name",
      "severity": "low/medium/high/critical",
      "description": "Description of violation",
      "confidence": 0.85,
      "suggested_fix": "How to fix this violation"
    }}
  ],
  "compliance_score": 0.85,
  "reasoning": "Overall assessment reasoning"
}}
"""

    async def critique_response(
        self, response: str, constitution: Constitution
    ) -> Dict[str, Any]:
        """Critique a response for constitutional compliance"""
        try:
            if not response:
                raise ValueError("Response must be a non-empty string")

            # Get enabled rules
            rules = constitution.get_rules(enabled_only=True)
            if not rules:
                logger.warning("No enabled constitutional rules found")
                return {
                    "overall_compliance": True,
                    "violations": [],
                    "compliance_score": 1.0,
                    "reasoning": "No rules to check against",
                }

            # Create rules text for prompt
            rules_text = "\n".join(
                [f"- {rule.name}: {rule.description}" for rule in rules]
            )

            prompt = self.critique_prompt.format(
                response=response, rules_text=rules_text
            )

            messages = [LLMMessage(role="user", content=prompt)]
            response_llm = await self.llm_provider.complete(messages)

            if not response_llm or not hasattr(response_llm, 'content'):
                logger.warning("Invalid response from LLM provider")
                return self._default_critique_result()

            content = response_llm.content

            # Parse critique response
            try:
                critique_data: Dict[str, Any] = json.loads(content)

                # Validate and normalize critique data
                overall_compliance = critique_data.get("overall_compliance", True)
                violations_data: List[Any] = critique_data.get("violations", [])
                compliance_score = float(critique_data.get("compliance_score", 1.0))
                reasoning = critique_data.get("reasoning", "No reasoning provided")

                # Process violations
                violations: List[SafetyViolation] = []
                for item in violations_data:
                    if isinstance(item, dict):
                        violation_data: Dict[str, Any] = item  # type: ignore
                        violation = SafetyViolation(
                            rule_id=str(violation_data.get("rule_id", "")),
                            rule_name=str(violation_data.get("rule_name", "")),
                            severity=str(violation_data.get("severity", "medium")),
                            description=str(violation_data.get("description", "")),
                            confidence=float(violation_data.get("confidence", 0.5)),
                            suggested_fix=str(violation_data.get("suggested_fix", "")),
                        )
                        violations.append(violation)

                return {
                    "overall_compliance": overall_compliance,
                    "violations": violations,
                    "compliance_score": max(0.0, min(1.0, compliance_score)),
                    "reasoning": reasoning,
                }

            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.warning(f"Failed to parse critique response: {e}")
                return self._default_critique_result()

        except Exception as e:
            logger.error(f"Error critiquing response: {e}")
            return self._default_critique_result()

    def _default_critique_result(self) -> Dict[str, Any]:
        """Return default critique result when analysis fails"""
        return {
            "overall_compliance": True,
            "violations": [],
            "compliance_score": 0.5,
            "reasoning": "Critique analysis failed, assuming compliance",
        }


class ResponseRevisor:
    """Revises responses to address constitutional violations"""

    def __init__(self, llm_provider: Any):
        if not llm_provider:
            raise ValueError("LLM provider cannot be null")

        if not hasattr(llm_provider, 'complete') or not callable(llm_provider.complete):
            raise ValueError("LLM provider must implement async complete() method")

        self.llm_provider = llm_provider

        self.revision_prompt = """
Revise the following response to address constitutional violations while maintaining helpfulness.

Original response: {original_response}

Detected violations:
{violations_text}

Constitutional principles:
{principles_text}

Please provide a revised response that:
1. Addresses all identified violations
2. Maintains the helpful intent of the original response
3. Follows constitutional principles
4. Is clear and constructive

Revised response:
"""

    async def revise_response(
        self,
        original_response: str,
        violations: List[SafetyViolation],
        constitution: Constitution,
    ) -> str:
        """Revise a response to address constitutional violations"""
        try:
            if not original_response:
                raise ValueError("Original response must be a non-empty string")

            if not violations:
                return original_response  # No violations to address

            # Create violations text
            violations_text = "\n".join(
                [
                    f"- {v.rule_name}: {v.description} (Severity: {v.severity})"
                    for v in violations
                ]
            )

            # Create principles text
            rules = constitution.get_rules(enabled_only=True)
            principles_text = "\n".join(
                [f"- {rule.name}: {rule.description}" for rule in rules]
            )

            prompt = self.revision_prompt.format(
                original_response=original_response,
                violations_text=violations_text,
                principles_text=principles_text,
            )

            messages = [LLMMessage(role="user", content=prompt)]
            response = await self.llm_provider.complete(messages)

            if response and hasattr(response, 'content'):
                return response.content.strip()
            else:
                logger.warning("Failed to get revision from LLM provider")
                return self._create_fallback_response(original_response, violations)

        except Exception as e:
            logger.error(f"Error revising response: {e}")
            return self._create_fallback_response(original_response, violations)

    def _create_fallback_response(
        self, original_response: str, violations: List[SafetyViolation]
    ) -> str:
        """Create a fallback response when revision fails"""
        if not violations:
            return original_response

        # Create a simple fallback that acknowledges the issue
        fallback = "I apologize, but I cannot provide the requested information as it may violate ethical principles. "
        fallback += "Please consider rephrasing your question or asking about a different topic. "
        fallback += "I'm here to help with questions that align with constitutional AI principles."

        return fallback


class ConstitutionalAgent:
    """Main Constitutional AI agent with safety-focused reasoning"""

    def __init__(
        self,
        llm_provider: Any,
        constitution: Optional[Constitution] = None,
        enable_revision: bool = True,
        max_revision_attempts: int = 3,
    ):
        """Initialize Constitutional AI agent with comprehensive validation"""

        # Validate llm_provider
        if not llm_provider:
            raise ValueError("LLM provider cannot be null")

        if not hasattr(llm_provider, 'complete') or not callable(llm_provider.complete):
            raise ValueError("LLM provider must implement async complete() method")

        # Validate numeric parameters
        if max_revision_attempts <= 0:
            raise ValueError(
                f"max_revision_attempts must be positive integer, got {max_revision_attempts}"
            )

        # Set bounds
        max_revision_attempts = min(max_revision_attempts, 5)  # Reasonable upper bound

        self.llm_provider = llm_provider
        self.enable_revision = bool(enable_revision)
        self.max_revision_attempts = max_revision_attempts

        # Initialize constitution
        if constitution:
            self.constitution = constitution
        else:
            self.constitution = Constitution()

        # Initialize components with proper error handling
        try:
            self.critique_system = CritiqueSystem(llm_provider)
            self.response_revisor = ResponseRevisor(llm_provider)
            logger.info("ConstitutionalAgent components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ConstitutionalAgent components: {e}")
            raise

        # Base reasoning prompt
        self.reasoning_prompt = """
Provide a helpful and accurate response to the following question or request.

Question/Request: {query}

Constitutional principles to follow:
{principles_text}

Please ensure your response:
1. Is helpful and informative
2. Follows constitutional AI principles
3. Is honest about limitations
4. Respects privacy and safety
5. Avoids harmful content

Response:
"""

        # Statistics
        self.stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "average_execution_time": 0.0,
            "violations_detected": 0,
            "revisions_performed": 0,
            "compliance_rate": 1.0,
        }

        logger.info(
            f"ConstitutionalAgent initialized with {len(self.constitution.rules)} rules"
        )

    async def process_response(
        self, query: str, enable_revision: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Process a query with constitutional AI safety checks"""
        start_time = time.time()

        try:
            # Validate input
            if not query:
                raise ValueError("Query must be a non-empty string")

            # Use instance setting if not overridden
            if enable_revision is None:
                enable_revision = self.enable_revision

            # Step 1: Generate initial response
            logger.info("Generating initial response...")
            initial_response = await self._generate_response(query)

            # Step 2: Critique for constitutional compliance
            logger.info("Critiquing response for constitutional compliance...")
            critique_result = await self.critique_system.critique_response(
                initial_response, self.constitution
            )

            violations = critique_result.get("violations", [])
            compliance_score = critique_result.get("compliance_score", 1.0)
            overall_compliance = critique_result.get("overall_compliance", True)

            # Step 3: Revise if needed and enabled
            final_response = initial_response
            revision_attempts = 0

            if (
                enable_revision
                and violations
                and revision_attempts < self.max_revision_attempts
            ):
                logger.info(
                    f"Revising response to address {len(violations)} violations..."
                )

                while violations and revision_attempts < self.max_revision_attempts:
                    revision_attempts += 1

                    # Revise response
                    revised_response = await self.response_revisor.revise_response(
                        final_response, violations, self.constitution
                    )

                    # Re-critique revised response
                    critique_result = await self.critique_system.critique_response(
                        revised_response, self.constitution
                    )
                    violations = critique_result.get("violations", [])
                    compliance_score = critique_result.get("compliance_score", 1.0)
                    overall_compliance = critique_result.get("overall_compliance", True)

                    final_response = revised_response

                    logger.info(
                        f"Revision attempt {revision_attempts}: {len(violations)} violations remaining"
                    )

            # Calculate statistics
            execution_time = time.time() - start_time

            # Update stats
            self.stats["total_executions"] += 1
            self.stats["successful_executions"] += 1
            self.stats["average_execution_time"] = (
                self.stats["average_execution_time"]
                * (self.stats["total_executions"] - 1)
                + execution_time
            ) / self.stats["total_executions"]

            if violations:
                self.stats["violations_detected"] += 1

            if revision_attempts > 0:
                self.stats["revisions_performed"] += 1

            # Update compliance rate
            total_executions = self.stats["total_executions"]
            compliant_executions = total_executions - self.stats["violations_detected"]
            self.stats["compliance_rate"] = (
                compliant_executions / total_executions if total_executions > 0 else 1.0
            )

            return {
                "response": final_response,
                "compliance_score": compliance_score,
                "overall_compliance": overall_compliance,
                "violations": [self._violation_to_dict(v) for v in violations],
                "revision_attempts": revision_attempts,
                "critique_reasoning": critique_result.get("reasoning", ""),
                "statistics": {
                    "execution_time": execution_time,
                    "initial_response_length": len(initial_response),
                    "final_response_length": len(final_response),
                    "rules_checked": len(
                        self.constitution.get_rules(enabled_only=True)
                    ),
                },
                "success": True,
                "error": None,
            }

        except Exception as e:
            execution_time = time.time() - start_time
            self.stats["total_executions"] += 1

            logger.error(f"Constitutional AI processing failed: {e}")

            return {
                "response": "I apologize, but I encountered an error while processing your request.",
                "compliance_score": 0.0,
                "overall_compliance": False,
                "violations": [],
                "revision_attempts": 0,
                "critique_reasoning": f"Processing error: {str(e)}",
                "statistics": {
                    "execution_time": execution_time,
                    "initial_response_length": 0,
                    "final_response_length": 0,
                    "rules_checked": 0,
                },
                "success": False,
                "error": str(e),
            }

    async def _generate_response(self, query: str) -> str:
        """Generate initial response to query"""
        try:
            # Get constitutional principles
            rules = self.constitution.get_rules(enabled_only=True)
            principles_text = "\n".join(
                [f"- {rule.name}: {rule.description}" for rule in rules]
            )

            prompt = self.reasoning_prompt.format(
                query=query, principles_text=principles_text
            )

            messages = [LLMMessage(role="user", content=prompt)]
            response = await self.llm_provider.complete(messages)

            if response and hasattr(response, 'content'):
                return response.content.strip()
            else:
                logger.warning("Failed to get response from LLM provider")
                return (
                    "I apologize, but I'm unable to generate a response at this time."
                )

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error: {str(e)}"

    def _violation_to_dict(self, violation: SafetyViolation) -> Dict[str, Any]:
        """Convert SafetyViolation to dictionary"""
        return {
            "rule_id": violation.rule_id,
            "rule_name": violation.rule_name,
            "severity": violation.severity,
            "description": violation.description,
            "confidence": violation.confidence,
            "suggested_fix": violation.suggested_fix,
        }

    def add_rule(self, rule: ConstitutionalRule) -> None:
        """Add a constitutional rule"""
        self.constitution.add_rule(rule)
        logger.info(f"Added constitutional rule: {rule.name}")

    def get_rules(self, category: Optional[str] = None) -> List[ConstitutionalRule]:
        """Get constitutional rules"""
        return self.constitution.get_rules(category=category, enabled_only=True)

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return self.stats.copy()

    async def reset(self) -> None:
        """Reset agent state"""
        self.stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "average_execution_time": 0.0,
            "violations_detected": 0,
            "revisions_performed": 0,
            "compliance_rate": 1.0,
        }
        logger.info("ConstitutionalAgent statistics reset")


# Export main classes
__all__ = [
    "ConstitutionalAgent",
    "Constitution",
    "ConstitutionalRule",
    "SafetyViolation",
    "CritiqueSystem",
    "ResponseRevisor",
]

"""
Advanced Knowledge Generator

Automatically generates synthetic guides, documentation, and metadata
using advanced LLM techniques and knowledge extraction methods.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

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


class KnowledgeType(Enum):
    """Types of knowledge that can be generated"""

    TUTORIAL = "tutorial"
    GUIDE = "guide"
    API_DOCUMENTATION = "api_documentation"
    FAQ = "faq"
    TROUBLESHOOTING = "troubleshooting"
    BEST_PRACTICES = "best_practices"
    REFERENCE = "reference"
    EXAMPLE = "example"


class GenerationStrategy(Enum):
    """Strategies for knowledge generation"""

    TEMPLATE_BASED = "template_based"
    LLM_BASED = "llm_based"
    HYBRID = "hybrid"
    RETRIEVAL_AUGMENTED = "retrieval_augmented"


@dataclass
class KnowledgeItem:
    """Represents a piece of generated knowledge"""

    id: str = ""
    title: str = ""
    content: str = ""
    knowledge_type: KnowledgeType = KnowledgeType.GUIDE
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = ""
    references: List[str] = field(default_factory=list)


@dataclass
class GenerationConfig:
    """Configuration for knowledge generation"""

    knowledge_type: KnowledgeType = KnowledgeType.GUIDE
    strategy: GenerationStrategy = GenerationStrategy.HYBRID
    target_audience: str = "general"
    complexity_level: str = "intermediate"
    include_examples: bool = True
    include_references: bool = True
    max_length: int = 2000
    min_quality_score: float = 0.6
    metadata_tags: List[str] = field(default_factory=list)
    custom_instructions: Optional[str] = None


@dataclass
class GenerationResult:
    """Result of knowledge generation process"""

    generated_items: List[KnowledgeItem]
    generation_stats: Dict[str, Any]
    quality_metrics: Dict[str, float]
    recommendations: List[str]


class KnowledgeGenerator:
    """Advanced knowledge generation system"""

    def __init__(self, llm_provider: Optional[Any] = None):
        self.llm_provider = llm_provider
        self.knowledge_base: Dict[str, KnowledgeItem] = {}
        self.knowledge_index: Dict[str, List[str]] = defaultdict(list)
        self.templates = self._load_templates()
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Setup logger for knowledge generator"""
        logger = logging.getLogger("KnowledgeGenerator")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _load_templates(self) -> Dict[KnowledgeType, str]:
        """Load knowledge generation templates"""
        return {
            KnowledgeType.TUTORIAL: """
                # {title}

                ## Overview
                This tutorial covers {topic} for {target_audience}.

                ## Prerequisites
                - Basic understanding of relevant concepts
                - Required tools and setup

                ## Step-by-Step Guide
                1. **Getting Started**
                   - Initial setup and configuration

                2. **Core Concepts**
                   - Key principles and terminology

                3. **Implementation**
                   - Practical examples and code samples

                4. **Best Practices**
                   - Tips and recommendations

                5. **Troubleshooting**
                   - Common issues and solutions

                ## Conclusion
                Summary of key learning points.

                ## References
                - Additional resources
                - Related documentation
            """,
            KnowledgeType.GUIDE: """
                # {title}

                ## Introduction
                This guide provides comprehensive information about {topic}.

                ## Key Concepts
                - Fundamental principles
                - Important terminology

                ## Implementation Details
                - Step-by-step instructions
                - Configuration options
                - Example usage

                ## Best Practices
                - Recommended approaches
                - Common pitfalls to avoid

                ## Advanced Topics
                - Expert-level features
                - Performance optimization

                ## Troubleshooting
                - Common issues and solutions
                - Debug techniques

                ## Conclusion
                Summary and next steps.
            """,
            KnowledgeType.API_DOCUMENTATION: """
                # {title} API Documentation

                ## Overview
                API documentation for {topic}.

                ## Authentication
                - Authentication methods
                - API key management

                ## Endpoints

                ### GET /api/endpoint
                - Description: Brief description
                - Parameters: Request parameters
                - Response: Example response
                - Error codes: Common errors

                ## Examples
                - Request examples
                - Response examples
                - Error handling

                ## Rate Limiting
                - Rate limit information
                - Best practices

                ## SDK Support
                - Available SDKs
                - Integration examples
            """,
            KnowledgeType.FAQ: """
                # {title} - Frequently Asked Questions

                ## General Questions

                **Q: What is {topic}?**
                A: Comprehensive answer explaining the concept.

                **Q: How do I get started?**
                A: Step-by-step getting started guide.

                ## Technical Questions

                **Q: How do I configure {topic}?**
                A: Configuration instructions and examples.

                **Q: What are the system requirements?**
                A: Detailed system requirements.

                ## Troubleshooting

                **Q: Why am I getting errors?**
                A: Common error causes and solutions.

                **Q: How do I debug issues?**
                A: Debugging techniques and tools.

                ## Advanced Topics

                **Q: How do I optimize performance?**
                A: Performance optimization strategies.
            """,
            KnowledgeType.TROUBLESHOOTING: """
                # {title} Troubleshooting Guide

                ## Common Issues

                ### Issue 1: [Problem Description]
                **Symptoms:**
                - List of symptoms

                **Causes:**
                - Potential causes

                **Solutions:**
                1. Step-by-step solution
                2. Alternative approaches

                ### Issue 2: [Problem Description]
                **Symptoms:**
                - List of symptoms

                **Causes:**
                - Potential causes

                **Solutions:**
                1. Step-by-step solution
                2. Alternative approaches

                ## Diagnostic Tools
                - Available diagnostic tools
                - How to use them

                ## Prevention
                - Best practices to prevent issues
                - Monitoring recommendations

                ## Getting Help
                - Where to find additional support
                - How to report issues
            """,
        }

    async def generate_knowledge(
        self,
        topic: str,
        config: GenerationConfig,
        context: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        """Generate knowledge item(s) on a given topic"""
        with tracer.start_as_current_span("generate_knowledge") as span:
            span.set_attribute("topic", topic)
            span.set_attribute("knowledge_type", config.knowledge_type.value)
            span.set_attribute("strategy", config.strategy.value)

            self.logger.info(
                f"Generating {config.knowledge_type.value} knowledge for topic: {topic}"
            )

            context = context or {}
            generated_items: List[KnowledgeItem] = []

            try:
                # Generate based on strategy
                if config.strategy == GenerationStrategy.TEMPLATE_BASED:
                    items = await self._generate_template_based(topic, config, context)
                elif config.strategy == GenerationStrategy.LLM_BASED:
                    items = await self._generate_llm_based(topic, config, context)
                elif config.strategy == GenerationStrategy.HYBRID:
                    items = await self._generate_hybrid(topic, config, context)
                elif config.strategy == GenerationStrategy.RETRIEVAL_AUGMENTED:
                    items = await self._generate_retrieval_augmented(
                        topic, config, context
                    )
                else:
                    raise ValueError(f"Unknown generation strategy: {config.strategy}")

                # Validate and filter items
                validated_items = []
                for item in items:
                    if self._validate_item(item, config):
                        validated_items.append(item)
                        self._add_to_knowledge_base(item)

                # Calculate quality metrics
                quality_metrics = self._calculate_quality_metrics(validated_items)

                # Generate recommendations
                recommendations = self._generate_recommendations(
                    validated_items, quality_metrics
                )

                # Create result
                result = GenerationResult(
                    generated_items=validated_items,
                    generation_stats={
                        "total_items": len(validated_items),
                        "average_quality": quality_metrics.get("average_quality", 0.0),
                        "generation_time": datetime.now(timezone.utc).isoformat(),
                    },
                    quality_metrics=quality_metrics,
                    recommendations=recommendations,
                )

                self.logger.info(
                    f"Generated {len(validated_items)} knowledge items for {topic}"
                )
                return result

            except Exception as e:
                self.logger.error(f"Failed to generate knowledge for {topic}: {e}")
                raise

    async def _generate_template_based(
        self, topic: str, config: GenerationConfig, context: Dict[str, Any]
    ) -> List[KnowledgeItem]:
        """Generate knowledge using templates"""
        template = self.templates.get(
            config.knowledge_type, self.templates[KnowledgeType.GUIDE]
        )

        # Format template with context
        formatted_content = template.format(
            title=f"{topic} {config.knowledge_type.value.title().replace('_', ' ')}",
            topic=topic,
            target_audience=config.target_audience,
            **context,
        )

        # Create knowledge item
        item = KnowledgeItem(
            id=self._generate_item_id(topic, config.knowledge_type),
            title=f"{topic} {config.knowledge_type.value.title().replace('_', ' ')}",
            content=formatted_content,
            knowledge_type=config.knowledge_type,
            tags=config.metadata_tags + [topic.lower()],
            quality_score=0.6,  # Base score for template-based
            source="template_based",
            metadata={
                "generation_strategy": config.strategy.value,
                "target_audience": config.target_audience,
                "complexity_level": config.complexity_level,
            },
        )

        return [item]

    async def _generate_llm_based(
        self, topic: str, config: GenerationConfig, context: Dict[str, Any]
    ) -> List[KnowledgeItem]:
        """Generate knowledge using LLM"""
        if not self.llm_provider:
            raise ValueError("LLM provider not configured")

        # Build prompt based on knowledge type
        prompt = self._build_generation_prompt(topic, config, context)

        try:
            # Generate content using LLM
            response = await self.llm_provider.complete(prompt)
            content = (
                response.content if hasattr(response, 'content') else str(response)
            )

            # Create knowledge item
            item = KnowledgeItem(
                id=self._generate_item_id(topic, config.knowledge_type),
                title=f"{topic} {config.knowledge_type.value.title().replace('_', ' ')}",
                content=content,
                knowledge_type=config.knowledge_type,
                tags=config.metadata_tags + [topic.lower()],
                quality_score=0.8,  # Higher score for LLM-based
                source="llm_based",
                metadata={
                    "generation_strategy": config.strategy.value,
                    "target_audience": config.target_audience,
                    "complexity_level": config.complexity_level,
                    "llm_generated": True,
                },
            )

            return [item]

        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            # Fallback to template-based
            return await self._generate_template_based(topic, config, context)

    async def _generate_hybrid(
        self, topic: str, config: GenerationConfig, context: Dict[str, Any]
    ) -> List[KnowledgeItem]:
        """Generate knowledge using hybrid approach"""
        # Start with template-based generation
        template_items = await self._generate_template_based(topic, config, context)

        if not self.llm_provider:
            return template_items

        # Enhance with LLM
        enhanced_items = []
        for item in template_items:
            try:
                enhanced_item = await self._enhance_with_llm(item, config)
                enhanced_items.append(enhanced_item)
            except Exception as e:
                self.logger.warning(f"Failed to enhance item with LLM: {e}")
                enhanced_items.append(item)  # Keep original if enhancement fails

        return enhanced_items

    async def _generate_retrieval_augmented(
        self, topic: str, config: GenerationConfig, context: Dict[str, Any]
    ) -> List[KnowledgeItem]:
        """Generate knowledge using retrieval-augmented approach"""
        # Search for relevant existing knowledge
        relevant_items = self.search_knowledge_base(topic, config.knowledge_type)

        # Use existing knowledge to inform generation
        augmented_context = context.copy()
        if relevant_items:
            augmented_context["existing_knowledge"] = [
                {"title": item.title, "content": item.content[:500]}
                for item in relevant_items[:3]
            ]

        # Generate with augmented context
        if self.llm_provider:
            return await self._generate_llm_based(topic, config, augmented_context)
        else:
            return await self._generate_template_based(topic, config, augmented_context)

    def _build_generation_prompt(
        self, topic: str, config: GenerationConfig, context: Dict[str, Any]
    ) -> str:
        """Build prompt for LLM generation"""
        prompts = {
            KnowledgeType.TUTORIAL: f"""
                Create a comprehensive tutorial on {topic} for {config.target_audience}.

                The tutorial should:
                - Be clear and easy to follow
                - Include practical examples
                - Cover common mistakes
                - Be appropriate for {config.complexity_level} level

                Topic: {topic}
                Context: {context}
            """,
            KnowledgeType.GUIDE: f"""
                Write a comprehensive guide on {topic} for {config.target_audience}.

                The guide should:
                - Explain key concepts clearly
                - Provide step-by-step implementation
                - Include best practices
                - Cover troubleshooting tips
                - Be suitable for {config.complexity_level} level

                Topic: {topic}
                Context: {context}
            """,
            KnowledgeType.API_DOCUMENTATION: f"""
                Generate API documentation for {topic}.

                Include:
                - Clear endpoint descriptions
                - Parameter specifications
                - Request/response examples
                - Error handling
                - Authentication methods

                API: {topic}
                Context: {context}
            """,
            KnowledgeType.FAQ: f"""
                Generate frequently asked questions about {topic}.

                Create 5-10 common questions and comprehensive answers that:
                - Address real user concerns
                - Provide actionable solutions
                - Include relevant context
                - Are appropriate for {config.target_audience}

                Topic: {topic}
                Context: {context}
            """,
            KnowledgeType.TROUBLESHOOTING: f"""
                Create a troubleshooting guide for {topic}.

                Include:
                - Common issues and symptoms
                - Step-by-step solutions
                - Diagnostic techniques
                - Prevention strategies

                Topic: {topic}
                Context: {context}
            """,
        }

        base_prompt = prompts.get(config.knowledge_type, prompts[KnowledgeType.GUIDE])

        # Add custom instructions
        if config.custom_instructions:
            base_prompt += f"\n\nAdditional instructions: {config.custom_instructions}"

        return base_prompt

    async def _enhance_with_llm(
        self, item: KnowledgeItem, config: GenerationConfig
    ) -> KnowledgeItem:
        """Enhance template-based content with LLM"""
        enhancement_prompt = f"""
        Enhance the following {config.knowledge_type.value} content to make it more comprehensive and engaging:

        Title: {item.title}
        Current Content:
        {item.content}

        Please:
        - Add more detailed explanations
        - Include practical examples
        - Improve clarity and flow
        - Maintain the existing structure
        - Keep it appropriate for {config.target_audience}
        """

        try:
            response = await self.llm_provider.complete(enhancement_prompt)
            enhanced_content = (
                response.content if hasattr(response, 'content') else str(response)
            )

            # Update item
            item.content = enhanced_content
            item.metadata["enhanced_with_llm"] = True
            item.quality_score = 0.85  # Higher score for hybrid approach
            item.source = "hybrid"

            return item

        except Exception as e:
            self.logger.error(f"Enhancement failed: {e}")
            return item

    def _generate_item_id(self, topic: str, knowledge_type: KnowledgeType) -> str:
        """Generate unique ID for knowledge item"""
        content = f"{topic}_{knowledge_type.value}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode().hexdigest())

    def _validate_item(self, item: KnowledgeItem, config: GenerationConfig) -> bool:
        """Validate knowledge item against configuration"""
        # Check minimum quality score
        if item.quality_score < config.min_quality_score:
            return False

        # Check content length
        if len(item.content) < 100:  # Minimum content length
            return False

        # Check maximum length
        if len(item.content) > config.max_length:
            item.content = item.content[: config.max_length] + "..."

        # Calculate and update quality score
        item.quality_score = self._calculate_item_quality(item)

        return True

    def _calculate_item_quality(self, item: KnowledgeItem) -> float:
        """Calculate quality score for knowledge item"""
        score = 0.0

        # Content length factor (0.3 weight)
        length_score = min(1.0, len(item.content) / 1000)
        score += 0.3 * length_score

        # Structure factor (0.2 weight)
        structure_indicators = ["#", "*", "-", "1.", "2."]
        structure_count = sum(
            1 for indicator in structure_indicators if indicator in item.content
        )
        structure_score = min(1.0, structure_count / 10)
        score += 0.2 * structure_score

        # Metadata completeness (0.15 weight)
        metadata_score = min(1.0, len(item.metadata) / 5)
        score += 0.15 * metadata_score

        # Tag relevance (0.1 weight)
        tag_score = min(1.0, len(item.tags) / 5)
        score += 0.1 * tag_score

        # Content diversity (0.25 weight)
        words = item.content.split()
        unique_words = len(set(words))
        total_words = len(words)
        diversity_score = unique_words / total_words if total_words > 0 else 0
        score += 0.25 * diversity_score

        return min(1.0, score)

    def _add_to_knowledge_base(self, item: KnowledgeItem):
        """Add item to knowledge base and update indexes"""
        self.knowledge_base[item.id] = item
        self.knowledge_index[item.knowledge_type.value].append(item.id)

        # Update tag index
        for tag in item.tags:
            self.knowledge_index[f"tag:{tag}"].append(item.id)

    def search_knowledge_base(
        self, query: str, knowledge_type: Optional[KnowledgeType] = None
    ) -> List[KnowledgeItem]:
        """Search knowledge base"""
        candidates = []

        # Filter by knowledge type
        if knowledge_type:
            candidates = [
                self.knowledge_base[item_id]
                for item_id in self.knowledge_index.get(knowledge_type.value, [])
            ]
        else:
            candidates = list(self.knowledge_base.values())

        # Simple text-based search
        query_lower = query.lower()
        results = []

        for item in candidates:
            if (
                query_lower in item.title.lower()
                or query_lower in item.content.lower()
                or any(query_lower in tag.lower() for tag in item.tags)
            ):
                results.append(item)

        # Sort by quality score
        results.sort(key=lambda x: x.quality_score, reverse=True)

        return results

    def _calculate_quality_metrics(
        self, items: List[KnowledgeItem]
    ) -> Dict[str, float]:
        """Calculate quality metrics for generated items"""
        if not items:
            return {}

        quality_scores = [item.quality_score for item in items]
        content_lengths = [len(item.content) for item in items]

        return {
            "average_quality": sum(quality_scores) / len(quality_scores),
            "max_quality": max(quality_scores),
            "min_quality": min(quality_scores),
            "average_length": sum(content_lengths) / len(content_lengths),
            "total_items": len(items),
            "high_quality_ratio": sum(1 for score in quality_scores if score > 0.8)
            / len(quality_scores),
        }

    def _generate_recommendations(
        self, items: List[KnowledgeItem], quality_metrics: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations for improving knowledge generation"""
        recommendations = []

        if quality_metrics.get("average_quality", 0) < 0.7:
            recommendations.append(
                "Consider using hybrid generation strategy for better quality"
            )

        if quality_metrics.get("high_quality_ratio", 0) < 0.6:
            recommendations.append(
                "Improve content quality by adding more detailed examples"
            )

        if quality_metrics.get("average_length", 0) < 500:
            recommendations.append("Consider generating more comprehensive content")

        # Check for content diversity
        if len(set(item.knowledge_type for item in items)) < 2:
            recommendations.append("Generate diverse types of knowledge content")

        return recommendations

    def export_knowledge_base(self, output_path: Path) -> Dict[str, Any]:
        """Export knowledge base to file"""
        export_data = {
            "items": [
                {
                    "id": item.id,
                    "title": item.title,
                    "content": item.content,
                    "knowledge_type": item.knowledge_type.value,
                    "tags": item.tags,
                    "metadata": item.metadata,
                    "quality_score": item.quality_score,
                    "generated_at": item.generated_at.isoformat(),
                    "source": item.source,
                }
                for item in self.knowledge_base.values()
            ],
            "export_metadata": {
                "total_items": len(self.knowledge_base),
                "export_time": datetime.now(timezone.utc).isoformat(),
                "knowledge_types": list(self.knowledge_index.keys()),
            },
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        return export_data


# Export main classes
__all__ = [
    "KnowledgeGenerator",
    "KnowledgeType",
    "GenerationStrategy",
    "KnowledgeItem",
    "GenerationConfig",
    "GenerationResult",
]

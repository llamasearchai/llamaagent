"""
Graph of Thoughts Implementation for LlamaAgent

Based on "Graph of Thoughts: Solving Elaborate Problems with Large Language Models"
by Besta et al., 2023. This implementation provides graph-based reasoning capabilities
with concept extraction, relationship mapping, and multi-path exploration.

Key Features:
- Dynamic concept extraction from problems
- Relationship mapping between concepts
- Multi-path reasoning with graph traversal
- Domain-specific reasoning patterns
- Comprehensive error handling and validation

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import json
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..llm import LLMMessage

logger = logging.getLogger(__name__)


@dataclass
class Concept:
    """Represents a concept in the reasoning graph"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    confidence: float = 0.0
    domain: str = ""
    metadata: Dict[str, Any] = field(default_factory=lambda: {})

    def __repr__(self) -> str:
        return f"Concept(name='{self.name}', confidence={self.confidence:.2f})"


@dataclass
class Relationship:
    """Represents a relationship between concepts"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    relationship_type: str = ""
    strength: float = 0.0
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=lambda: {})

    def __repr__(self) -> str:
        return f"Relationship({self.source_id} -> {self.target_id}: {self.relationship_type})"


class ConceptExtractor:
    """Extracts concepts from problems and domains"""

    def __init__(self, llm_provider: Any):
        if not llm_provider:
            raise ValueError("LLM provider cannot be null")

        if not hasattr(llm_provider, 'complete'):
            raise ValueError("LLM provider must implement async complete() method")

        self.llm_provider = llm_provider

        self.extraction_prompt = """
Extract key concepts from the given problem and domain context.

Problem: {problem}
Domain: {domain}

Identify the most important concepts that are relevant to solving this problem.
For each concept, provide:
- A clear, concise name
- A brief description of its role in the problem
- Your confidence in its relevance (0.0-1.0)

Provide the concepts as a JSON list:
[
  {{
    "name": "Concept Name",
    "description": "Brief description of the concept",
    "confidence": 0.85,
    "domain": "{domain}"
  }}
]

Extract between 3-8 concepts, focusing on the most relevant ones for solving the problem.
"""

    async def extract_concepts(
        self, problem: str, domain: str = "general"
    ) -> List[Concept]:
        """Extract concepts from problem and domain"""
        try:
            if not problem:
                raise ValueError("Problem must be a non-empty string")

            if not domain:
                domain = "general"

            prompt = self.extraction_prompt.format(problem=problem, domain=domain)
            messages = [LLMMessage(role="user", content=prompt)]

            response = await self.llm_provider.complete(messages)

            if not response or not hasattr(response, 'content'):
                logger.warning("Invalid response from LLM provider")
                return []

            content = response.content

            # Parse concepts from response
            try:
                concepts_raw: Any = json.loads(content)
                if not isinstance(concepts_raw, list):
                    logger.warning("Response is not a list of concepts")
                    return []

                concepts: List[Concept] = []
                concepts_list: List[Any] = concepts_raw  # type: ignore
                for item in concepts_list:
                    if isinstance(item, dict):
                        concept_data: Dict[str, Any] = item  # type: ignore
                        concept = Concept(
                            name=str(concept_data.get("name", "")),
                            description=str(concept_data.get("description", "")),
                            confidence=float(concept_data.get("confidence", 0.5)),
                            domain=str(concept_data.get("domain", domain)),
                        )
                        concepts.append(concept)

                # Filter out low-confidence concepts
                concepts = [c for c in concepts if c.confidence > 0.3]

                logger.info(f"Extracted {len(concepts)} concepts from problem")
                return concepts

            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.warning(f"Failed to parse concepts: {e}")
                return []

        except Exception as e:
            logger.error(f"Error extracting concepts: {e}")
            return []


class RelationshipMapper:
    """Maps relationships between concepts"""

    def __init__(self, llm_provider: Any):
        if not llm_provider:
            raise ValueError("LLM provider cannot be null")

        if not hasattr(llm_provider, 'complete'):
            raise ValueError("LLM provider must implement async complete() method")

        self.llm_provider = llm_provider

        self.mapping_prompt = """
Analyze the relationships between the following concepts in the context of the problem.

Problem: {problem}
Domain: {domain}

Concepts:
{concepts_text}

For each pair of concepts, identify if there's a meaningful relationship.
Relationship types include: causes, influences, depends_on, similar_to, opposite_of, part_of, etc.

Provide relationships as JSON:
[
  {{
    "source": "Concept Name 1",
    "target": "Concept Name 2",
    "type": "relationship_type",
    "strength": 0.75,
    "description": "Brief description of the relationship"
  }}
]

Focus on the most important relationships that help solve the problem.
"""

    async def map_relationships(
        self, concepts: List[Concept], problem: str, domain: str = "general"
    ) -> List[Relationship]:
        """Map relationships between concepts"""
        try:
            if not concepts:
                return []

            if not problem:
                raise ValueError("Problem must be a non-empty string")

            # Create concept text for prompt
            concepts_text = "\n".join(
                [
                    f"- {c.name}: {c.description} (confidence: {c.confidence:.2f})"
                    for c in concepts
                ]
            )

            prompt = self.mapping_prompt.format(
                problem=problem, domain=domain, concepts_text=concepts_text
            )

            messages = [LLMMessage(role="user", content=prompt)]
            response = await self.llm_provider.complete(messages)

            if not response or not hasattr(response, 'content'):
                logger.warning("Invalid response from LLM provider")
                return []

            content = response.content

            # Parse relationships from response
            try:
                relationships_raw: Any = json.loads(content)
                if not isinstance(relationships_raw, list):
                    logger.warning("Response is not a list of relationships")
                    return []

                # Create concept name to ID mapping
                concept_map: Dict[str, str] = {c.name: c.id for c in concepts}

                relationships: List[Relationship] = []
                relationships_list: List[Any] = relationships_raw  # type: ignore
                for item in relationships_list:
                    if isinstance(item, dict):
                        rel_data: Dict[str, Any] = item  # type: ignore
                        source_name: str = str(rel_data.get("source", ""))
                        target_name: str = str(rel_data.get("target", ""))

                        if source_name in concept_map and target_name in concept_map:
                            relationship = Relationship(
                                source_id=concept_map[source_name],
                                target_id=concept_map[target_name],
                                relationship_type=str(rel_data.get("type", "related")),
                                strength=float(rel_data.get("strength", 0.5)),
                                description=str(rel_data.get("description", "")),
                            )
                            relationships.append(relationship)

                logger.info(
                    f"Mapped {len(relationships)} relationships between concepts"
                )
                return relationships

            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.warning(f"Failed to parse relationships: {e}")
                return []

        except Exception as e:
            logger.error(f"Error mapping relationships: {e}")
            return []


class ReasoningGraph:
    """Graph structure for organizing concepts and relationships"""

    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self.relationships: Dict[str, Relationship] = {}
        self.adjacency_list: Dict[str, List[str]] = defaultdict(list)
        self.reverse_adjacency: Dict[str, List[str]] = defaultdict(list)

    def add_concept(self, concept: Concept) -> None:
        """Add a concept to the graph"""
        self.concepts[concept.id] = concept

    def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship to the graph"""
        # Validate that both concepts exist
        if relationship.source_id not in self.concepts:
            raise ValueError(f"Source concept {relationship.source_id} not found")

        if relationship.target_id not in self.concepts:
            raise ValueError(f"Target concept {relationship.target_id} not found")

        self.relationships[relationship.id] = relationship

        # Update adjacency lists
        self.adjacency_list[relationship.source_id].append(relationship.target_id)
        self.reverse_adjacency[relationship.target_id].append(relationship.source_id)

    def get_concept(self, concept_id: str) -> Optional[Concept]:
        """Get a concept by ID"""
        return self.concepts.get(concept_id)

    def get_relationships(self, concept_id: str) -> List[Relationship]:
        """Get all relationships involving a concept"""
        relationships: List[Relationship] = []
        for rel in self.relationships.values():
            if rel.source_id == concept_id or rel.target_id == concept_id:
                relationships.append(rel)
        return relationships

    def get_neighbors(self, concept_id: str) -> List[Concept]:
        """Get neighboring concepts"""
        neighbors: List[Concept] = []
        for target_id in self.adjacency_list[concept_id]:
            neighbor = self.concepts.get(target_id)
            if neighbor:
                neighbors.append(neighbor)
        return neighbors

    def find_paths(
        self, start_id: str, end_id: str, max_depth: int = 3
    ) -> List[List[Concept]]:
        """Find paths between two concepts"""
        if start_id not in self.concepts or end_id not in self.concepts:
            return []

        paths: List[List[Concept]] = []
        queue: deque[tuple[str, List[Concept]]] = deque(
            [(start_id, [self.concepts[start_id]])]
        )
        visited: set[tuple[str, int]] = set()

        while queue:
            current_id, current_path = queue.popleft()

            if current_id == end_id:
                paths.append(current_path)
                continue

            if len(current_path) >= max_depth:
                continue

            # Create state key for visited tracking
            state_key = (current_id, len(current_path))
            if state_key in visited:
                continue
            visited.add(state_key)

            # Explore neighbors
            for neighbor_id in self.adjacency_list[current_id]:
                neighbor = self.concepts.get(neighbor_id)
                if neighbor and neighbor_id not in [c.id for c in current_path]:
                    new_path = current_path + [neighbor]
                    queue.append((neighbor_id, new_path))

        return paths

    def get_central_concepts(self, top_k: int = 3) -> List[Concept]:
        """Get the most central concepts based on connectivity"""
        centrality: Dict[str, int] = {}

        for concept_id in self.concepts:
            # Calculate degree centrality (number of connections)
            in_degree = len(self.reverse_adjacency[concept_id])
            out_degree = len(self.adjacency_list[concept_id])
            centrality[concept_id] = in_degree + out_degree

        # Sort by centrality and return top concepts
        sorted_concepts: List[tuple[str, int]] = sorted(
            centrality.items(), key=lambda x: x[1], reverse=True
        )

        central_concepts: List[Concept] = []
        for concept_id, _ in sorted_concepts[:top_k]:
            concept = self.concepts.get(concept_id)
            if concept:
                central_concepts.append(concept)

        return central_concepts


class GraphOfThoughtsAgent:
    """Main Graph of Thoughts reasoning agent"""

    def __init__(
        self,
        llm_provider: Any,
        max_concepts: int = 8,
        max_depth: int = 3,
        max_iterations: int = 50,
    ):
        """Initialize Graph of Thoughts agent with comprehensive validation"""

        # Validate llm_provider
        if not llm_provider:
            raise ValueError("LLM provider cannot be null")

        if not hasattr(llm_provider, 'complete'):
            raise ValueError("LLM provider must implement async complete() method")

        # Validate numeric parameters
        if max_concepts <= 0:
            raise ValueError(
                f"max_concepts must be positive integer, got {max_concepts}"
            )

        if max_depth <= 0:
            raise ValueError(f"max_depth must be positive integer, got {max_depth}")

        if max_iterations <= 0:
            raise ValueError(
                f"max_iterations must be positive integer, got {max_iterations}"
            )

        # Set bounds
        max_concepts = min(max_concepts, 15)  # Reasonable upper bound
        max_depth = min(max_depth, 5)  # Reasonable upper bound
        max_iterations = min(max_iterations, 100)  # Reasonable upper bound

        self.llm_provider = llm_provider
        self.max_concepts = max_concepts
        self.max_depth = max_depth
        self.max_iterations = max_iterations

        # Initialize components with proper error handling
        try:
            self.concept_extractor = ConceptExtractor(llm_provider)
            self.relationship_mapper = RelationshipMapper(llm_provider)
            logger.info("GraphOfThoughtsAgent components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize GraphOfThoughtsAgent components: {e}")
            raise

        # Reasoning prompt
        self.reasoning_prompt = """
Based on the concept graph and relationships, provide a comprehensive solution to the problem.

Problem: {problem}
Domain: {domain}

Concepts and Relationships:
{graph_description}

Key Reasoning Paths:
{reasoning_paths}

Provide a detailed solution that leverages the relationships between concepts.
Structure your response as:
1. Problem Analysis
2. Key Insights from Concept Relationships
3. Solution Approach
4. Implementation Considerations
5. Expected Outcomes

Solution:
"""

        # Statistics
        self.stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "average_execution_time": 0.0,
            "average_concepts_extracted": 0.0,
            "average_relationships_mapped": 0.0,
        }

        logger.info(
            f"GraphOfThoughtsAgent initialized with max_concepts={max_concepts}, max_depth={max_depth}"
        )

    async def solve(self, problem: str, domain: str = "general") -> Dict[str, Any]:
        """Solve a problem using Graph of Thoughts reasoning"""
        start_time = time.time()

        try:
            # Validate input
            if not problem:
                raise ValueError("Problem must be a non-empty string")

            if not domain:
                domain = "general"

            # Step 1: Extract concepts
            logger.info("Extracting concepts from problem...")
            concepts = await self.concept_extractor.extract_concepts(problem, domain)

            if not concepts:
                logger.warning("No concepts extracted, using fallback approach")
                return await self._fallback_solve(problem, domain)

            # Limit concepts to max_concepts
            concepts = concepts[: self.max_concepts]

            # Step 2: Map relationships
            logger.info("Mapping relationships between concepts...")
            relationships = await self.relationship_mapper.map_relationships(
                concepts, problem, domain
            )

            # Step 3: Build reasoning graph
            graph = ReasoningGraph()

            # Add concepts to graph
            for concept in concepts:
                graph.add_concept(concept)

            # Add relationships to graph
            for relationship in relationships:
                try:
                    graph.add_relationship(relationship)
                except ValueError as e:
                    logger.warning(f"Skipping invalid relationship: {e}")

            # Step 4: Find reasoning paths
            logger.info("Finding reasoning paths...")
            reasoning_paths = self._find_reasoning_paths(graph, concepts)

            # Step 5: Generate solution
            logger.info("Generating solution...")
            solution = await self._generate_solution(
                problem, domain, graph, reasoning_paths
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
            self.stats["average_concepts_extracted"] = (
                self.stats["average_concepts_extracted"]
                * (self.stats["total_executions"] - 1)
                + len(concepts)
            ) / self.stats["total_executions"]
            self.stats["average_relationships_mapped"] = (
                self.stats["average_relationships_mapped"]
                * (self.stats["total_executions"] - 1)
                + len(relationships)
            ) / self.stats["total_executions"]

            return {
                "solution": solution,
                "confidence": self._calculate_confidence(concepts, relationships),
                "concepts": [
                    {
                        "name": c.name,
                        "description": c.description,
                        "confidence": c.confidence,
                    }
                    for c in concepts
                ],
                "relationships": [
                    {
                        "source": r.source_id,
                        "target": r.target_id,
                        "type": r.relationship_type,
                        "strength": r.strength,
                    }
                    for r in relationships
                ],
                "reasoning_paths": reasoning_paths,
                "statistics": {
                    "execution_time": execution_time,
                    "concepts_extracted": len(concepts),
                    "relationships_mapped": len(relationships),
                    "graph_nodes": len(graph.concepts),
                    "graph_edges": len(graph.relationships),
                    "domain": domain,
                },
                "success": True,
                "error": None,
            }

        except Exception as e:
            execution_time = time.time() - start_time
            self.stats["total_executions"] += 1

            logger.error(f"Graph of Thoughts solving failed: {e}")

            return {
                "solution": "Error occurred during solving",
                "confidence": 0.0,
                "concepts": [],
                "relationships": [],
                "reasoning_paths": [],
                "statistics": {
                    "execution_time": execution_time,
                    "concepts_extracted": 0,
                    "relationships_mapped": 0,
                    "graph_nodes": 0,
                    "graph_edges": 0,
                    "domain": domain,
                },
                "success": False,
                "error": str(e),
            }

    def _find_reasoning_paths(
        self, graph: ReasoningGraph, concepts: List[Concept]
    ) -> List[List[str]]:
        """Find important reasoning paths in the graph"""
        paths: List[List[str]] = []

        # Find paths between central concepts
        central_concepts = graph.get_central_concepts(min(3, len(concepts)))

        if len(central_concepts) >= 2:
            for i in range(len(central_concepts)):
                for j in range(i + 1, len(central_concepts)):
                    concept_paths = graph.find_paths(
                        central_concepts[i].id, central_concepts[j].id, self.max_depth
                    )

                    for path in concept_paths:
                        path_names = [c.name for c in path]
                        if path_names not in paths:
                            paths.append(path_names)

        # If no paths found, create simple concept sequences
        if not paths and concepts:
            for i in range(len(concepts) - 1):
                paths.append([concepts[i].name, concepts[i + 1].name])

        return paths[:5]  # Limit to top 5 paths

    async def _generate_solution(
        self,
        problem: str,
        domain: str,
        graph: ReasoningGraph,
        reasoning_paths: List[List[str]],
    ) -> str:
        """Generate solution using the reasoning graph"""
        try:
            # Create graph description
            graph_description = "Concepts:\n"
            for concept in graph.concepts.values():
                graph_description += f"- {concept.name}: {concept.description}\n"

            graph_description += "\nRelationships:\n"
            for rel in graph.relationships.values():
                source = graph.concepts.get(rel.source_id)
                target = graph.concepts.get(rel.target_id)
                if source and target:
                    graph_description += f"- {source.name} -> {target.name}: {rel.relationship_type} (strength: {rel.strength:.2f})\n"

            # Create reasoning paths description
            paths_description = ""
            for i, path in enumerate(reasoning_paths, 1):
                paths_description += f"Path {i}: {' -> '.join(path)}\n"

            prompt = self.reasoning_prompt.format(
                problem=problem,
                domain=domain,
                graph_description=graph_description,
                reasoning_paths=paths_description,
            )

            messages = [LLMMessage(role="user", content=prompt)]
            response = await self.llm_provider.complete(messages)

            if response and hasattr(response, 'content'):
                return response.content.strip()
            else:
                return "Unable to generate solution from reasoning graph"

        except Exception as e:
            logger.error(f"Error generating solution: {e}")
            return f"Solution generation failed: {str(e)}"

    def _calculate_confidence(
        self, concepts: List[Concept], relationships: List[Relationship]
    ) -> float:
        """Calculate confidence based on concept and relationship quality"""
        if not concepts:
            return 0.0

        # Average concept confidence
        avg_concept_confidence = sum(c.confidence for c in concepts) / len(concepts)

        # Relationship strength factor
        relationship_strength = 0.0
        if relationships:
            relationship_strength = sum(r.strength for r in relationships) / len(
                relationships
            )

        # Connectivity factor (more relationships = higher confidence)
        connectivity_factor = min(len(relationships) / max(len(concepts), 1), 1.0)

        # Weighted combination
        confidence = (
            0.5 * avg_concept_confidence
            + 0.3 * relationship_strength
            + 0.2 * connectivity_factor
        )

        return max(0.0, min(1.0, confidence))

    async def _fallback_solve(self, problem: str, domain: str) -> Dict[str, Any]:
        """Fallback solution when concept extraction fails"""
        try:
            fallback_prompt = f"""
Solve the following problem using systematic reasoning:

Problem: {problem}
Domain: {domain}

Provide a comprehensive solution with:
1. Problem Analysis
2. Key Considerations
3. Solution Approach
4. Implementation Steps
5. Expected Outcomes

Solution:
"""

            messages = [LLMMessage(role="user", content=fallback_prompt)]
            response = await self.llm_provider.complete(messages)

            solution = (
                response.content.strip()
                if response and hasattr(response, 'content')
                else "Fallback solution unavailable"
            )

            return {
                "solution": solution,
                "confidence": 0.3,  # Lower confidence for fallback
                "concepts": [],
                "relationships": [],
                "reasoning_paths": [],
                "statistics": {
                    "execution_time": 0.0,
                    "concepts_extracted": 0,
                    "relationships_mapped": 0,
                    "graph_nodes": 0,
                    "graph_edges": 0,
                    "domain": domain,
                    "fallback_used": True,
                },
                "success": True,
                "error": None,
            }

        except Exception as e:
            logger.error(f"Fallback solution failed: {e}")
            return {
                "solution": "Both primary and fallback solutions failed",
                "confidence": 0.0,
                "concepts": [],
                "relationships": [],
                "reasoning_paths": [],
                "statistics": {
                    "execution_time": 0.0,
                    "concepts_extracted": 0,
                    "relationships_mapped": 0,
                    "graph_nodes": 0,
                    "graph_edges": 0,
                    "domain": domain,
                    "fallback_used": True,
                },
                "success": False,
                "error": str(e),
            }

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return self.stats.copy()

    async def reset(self) -> None:
        """Reset agent state"""
        self.stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "average_execution_time": 0.0,
            "average_concepts_extracted": 0.0,
            "average_relationships_mapped": 0.0,
        }
        logger.info("GraphOfThoughtsAgent statistics reset")


# Export main classes
__all__ = [
    "GraphOfThoughtsAgent",
    "Concept",
    "Relationship",
    "ReasoningGraph",
    "ConceptExtractor",
    "RelationshipMapper",
]

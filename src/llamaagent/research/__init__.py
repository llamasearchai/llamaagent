"""
Research Module for LlamaAgent

This module provides cutting-edge research capabilities including:
- Citation management and analysis
- Evidence evaluation and fact-checking
- Knowledge graph construction
- Scientific reasoning and analysis
- Literature review automation
- Academic report generation
"""

from .citations import (
    Citation,
    CitationAnalyzer,
    CitationFormatter,
    CitationGraph,
    CitationManager,
    DOIResolver,
)
from .evidence import (
    ClaimVerifier,
    ConsensusBuilder,
    ContradictionDetector,
    Evidence,
    EvidenceAnalyzer,
    EvidenceRanker,
    SourceCredibilityAnalyzer,
)
from .knowledge_graph import (
    ConceptExtractor,
    GraphQuerying,
    GraphVisualizer,
    KnowledgeGraph,
    RelationshipMapper,
    SemanticSimilarityAnalyzer,
)
from .literature_review import (
    DataSynthesizer,
    ExperimentDesigner,
    HypothesisGenerator,
    LiteratureReviewer,
    ResearchGapAnalyzer,
    SystematicReviewProtocol,
)
from .report_generator import (
    AbstractGenerator,
    AcademicFormatter,
    ExecutiveSummarizer,
    ResearchReporter,
    VisualizationGenerator,
)
from .scientific_reasoning import (
    BayesianReasoner,
    CausalReasoner,
    EffectSizeCalculator,
    HypothesisTester,
    MetaAnalyzer,
    StatisticalAnalyzer,
)

__all__ = [
    # Citations
    "Citation",
    "CitationManager",
    "CitationFormatter",
    "DOIResolver",
    "CitationGraph",
    "CitationAnalyzer",
    # Evidence
    "Evidence",
    "EvidenceAnalyzer",
    "ClaimVerifier",
    "EvidenceRanker",
    "ContradictionDetector",
    "ConsensusBuilder",
    "SourceCredibilityAnalyzer",
    # Knowledge Graph
    "KnowledgeGraph",
    "ConceptExtractor",
    "RelationshipMapper",
    "GraphQuerying",
    "GraphVisualizer",
    "SemanticSimilarityAnalyzer",
    # Scientific Reasoning
    "CausalReasoner",
    "StatisticalAnalyzer",
    "BayesianReasoner",
    "MetaAnalyzer",
    "HypothesisTester",
    "EffectSizeCalculator",
    # Literature Review
    "LiteratureReviewer",
    "HypothesisGenerator",
    "ExperimentDesigner",
    "DataSynthesizer",
    "SystematicReviewProtocol",
    "ResearchGapAnalyzer",
    # Report Generation
    "ResearchReporter",
    "AcademicFormatter",
    "VisualizationGenerator",
    "AbstractGenerator",
    "ExecutiveSummarizer",
]

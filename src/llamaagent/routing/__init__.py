"""
AI Router System for intelligent task routing between AI providers.

This package provides a comprehensive routing system for intelligently
distributing coding tasks between different AI providers based on various
strategies including task type, language, complexity, performance, and cost.
"""

from .ai_router import AIRouter, RoutingDecision
from .metrics import PerformanceTracker, RoutingMetrics
from .provider_registry import ProviderCapabilities, ProviderRegistry
from .strategies import (
    ComplexityBasedRouting,
    ConsensusRouting,
    CostOptimizedRouting,
    HybridRouting,
    LanguageBasedRouting,
    PerformanceBasedRouting,
    RoutingStrategy,
    TaskBasedRouting,
)
from .task_analyzer import TaskAnalyzer, TaskCharacteristics

__all__ = [
    "AIRouter",
    "RoutingDecision",
    "ProviderRegistry",
    "ProviderCapabilities",
    "RoutingStrategy",
    "TaskBasedRouting",
    "LanguageBasedRouting",
    "ComplexityBasedRouting",
    "PerformanceBasedRouting",
    "CostOptimizedRouting",
    "HybridRouting",
    "ConsensusRouting",
    "TaskAnalyzer",
    "TaskCharacteristics",
    "RoutingMetrics",
    "PerformanceTracker",
]

__version__ = "0.1.0"

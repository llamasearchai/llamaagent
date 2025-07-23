"""
Shared types and data classes for the routing module.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class RoutingMode(Enum):
    """Routing modes for different use cases."""

    SINGLE = "single"  # Route to single best provider
    PARALLEL = "parallel"  # Route to multiple providers in parallel
    CONSENSUS = "consensus"  # Get consensus from multiple providers
    FALLBACK = "fallback"  # Try providers in order until success
    LOAD_BALANCED = "load_balanced"  # Distribute load across providers


@dataclass
class RoutingDecision:
    """Represents a routing decision with metadata."""

    provider_id: str
    confidence: float
    reasoning: str
    estimated_cost: float
    estimated_duration: float
    alternative_providers: List[Tuple[str, float]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingConfig:
    """Configuration for the AI router."""

    mode: RoutingMode = RoutingMode.SINGLE
    enable_ab_testing: bool = False
    ab_test_percentage: float = 0.1
    enable_caching: bool = True
    cache_ttl: int = 3600
    enable_fallback: bool = True
    max_retries: int = 3
    timeout: float = 30.0
    cost_threshold: Optional[float] = None
    quality_threshold: Optional[float] = None
    load_balance_weights: Dict[str, float] = field(default_factory=dict)

"""
Provider registry for managing AI provider capabilities and configuration.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Types of AI providers."""

    CLOUD = "cloud"
    LOCAL = "local"
    HYBRID = "hybrid"


class ProviderStatus(Enum):
    """Provider availability status."""

    ACTIVE = "active"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


@dataclass
class ProviderCapabilities:
    """Capabilities and characteristics of an AI provider."""

    provider_id: str
    provider_type: ProviderType
    display_name: str
    description: str

    # Language support
    supported_languages: Set[str] = field(default_factory=set)
    specialized_languages: Set[str] = field(
        default_factory=set
    )  # Languages with exceptional support

    # Task type strengths (task_type -> strength score 0-1)
    strengths: Dict[str, float] = field(default_factory=dict)
    # Technical capabilities
    max_tokens: int = 4096
    supports_streaming: bool = True
    supports_function_calling: bool = False
    supports_vision: bool = False
    supports_code_execution: bool = False

    # Performance characteristics
    avg_latency_ms: float = 1000.0  # Average response time
    throughput_rps: float = 10.0  # Requests per second
    concurrency_limit: int = 10  # Max concurrent requests

    # Cost information
    cost_per_token: float = 0.001  # Cost per token in USD
    cost_per_request: float = 0.0  # Fixed cost per request
    free_tier_tokens: int = 0  # Free tokens per month

    # Quality metrics
    accuracy_score: float = 0.85  # General accuracy score
    consistency_score: float = 0.8  # Response consistency
    creativity_score: float = 0.7  # Creative problem solving

    # Constraints
    rate_limit_rpm: int = 60  # Requests per minute
    rate_limit_tpd: int = 100000  # Tokens per day
    context_window: int = 8192  # Maximum context size

    # API information
    api_endpoint: str = ""
    api_version: str = "v1"
    requires_api_key: bool = True

    # Metadata
    status: ProviderStatus = ProviderStatus.ACTIVE
    last_updated: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProviderRegistry:
    """Registry for managing AI provider information and capabilities."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize provider registry.

        Args:
            config_path: Path to provider configuration file
        """
        self.providers: Dict[str, ProviderCapabilities] = {}
        self.config_path = config_path

        # Initialize with default providers
        self._initialize_default_providers()

        # Load custom configuration if provided
        if config_path and config_path.exists():
            self.load_config(config_path)

    def _initialize_default_providers(self) -> None:
        """Initialize registry with default provider configurations."""
        # Claude Code
        self.register_provider(
            ProviderCapabilities(
                provider_id="claude-code",
                provider_type=ProviderType.CLOUD,
                display_name="Claude Code",
                description="Anthropic's advanced coding assistant with excellent reasoning",
                supported_languages={
                    "python",
                    "javascript",
                    "typescript",
                    "java",
                    "cpp",
                    "csharp",
                    "rust",
                    "go",
                    "ruby",
                    "php",
                    "swift",
                    "kotlin",
                    "r",
                    "sql",
                    "html",
                    "css",
                    "bash",
                    "powershell",
                    "yaml",
                    "json",
                    "xml",
                },
                specialized_languages={"python", "javascript", "typescript", "rust"},
                strengths={
                    "debugging": 0.95,
                    "code_review": 0.95,
                    "documentation": 0.9,
                    "optimization": 0.9,
                    "architecture": 0.85,
                    "refactoring": 0.85,
                    "testing": 0.85,
                },
                max_tokens=100000,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=True,
                avg_latency_ms=2000,
                throughput_rps=20,
                concurrency_limit=20,
                cost_per_token=0.00003,
                accuracy_score=0.92,
                consistency_score=0.9,
                creativity_score=0.85,
                rate_limit_rpm=40,
                rate_limit_tpd=1000000,
                context_window=100000,
                api_endpoint="https://api.anthropic.com/v1/messages",
                tags=["premium", "reasoning", "long-context"],
            )
        )

        # OpenAI Codex
        self.register_provider(
            ProviderCapabilities(
                provider_id="openai-codex",
                provider_type=ProviderType.CLOUD,
                display_name="OpenAI Codex",
                description="OpenAI's specialized code generation model",
                supported_languages={
                    "python",
                    "javascript",
                    "typescript",
                    "java",
                    "cpp",
                    "csharp",
                    "go",
                    "ruby",
                    "php",
                    "swift",
                    "kotlin",
                    "r",
                    "sql",
                    "bash",
                },
                specialized_languages={"python", "javascript", "java"},
                strengths={
                    "code_generation": 0.95,
                    "refactoring": 0.9,
                    "testing": 0.9,
                    "debugging": 0.85,
                    "optimization": 0.8,
                },
                max_tokens=8192,
                supports_streaming=True,
                supports_function_calling=True,
                avg_latency_ms=1500,
                throughput_rps=30,
                concurrency_limit=30,
                cost_per_token=0.00002,
                accuracy_score=0.88,
                consistency_score=0.85,
                creativity_score=0.8,
                rate_limit_rpm=60,
                rate_limit_tpd=2000000,
                context_window=8192,
                api_endpoint="https://api.openai.com/v1/completions",
                tags=["code-generation", "fast", "popular"],
            )
        )

        # GitHub Copilot
        self.register_provider(
            ProviderCapabilities(
                provider_id="github-copilot",
                provider_type=ProviderType.CLOUD,
                display_name="GitHub Copilot",
                description="GitHub's AI pair programmer",
                supported_languages={
                    "python",
                    "javascript",
                    "typescript",
                    "java",
                    "cpp",
                    "csharp",
                    "go",
                    "ruby",
                    "php",
                    "swift",
                    "kotlin",
                    "rust",
                    "scala",
                },
                specialized_languages={"javascript", "typescript", "python"},
                strengths={
                    "code_generation": 0.9,
                    "code_completion": 0.95,
                    "testing": 0.8,
                    "refactoring": 0.75,
                },
                max_tokens=4096,
                supports_streaming=True,
                avg_latency_ms=500,
                throughput_rps=50,
                concurrency_limit=50,
                cost_per_token=0.00001,
                accuracy_score=0.85,
                consistency_score=0.88,
                creativity_score=0.75,
                rate_limit_rpm=100,
                rate_limit_tpd=5000000,
                context_window=4096,
                api_endpoint="https://api.github.com/copilot/completions",
                tags=["fast", "ide-integrated", "autocomplete"],
            )
        )

        # Local Model (e.g., CodeLlama)
        self.register_provider(
            ProviderCapabilities(
                provider_id="local-codellama",
                provider_type=ProviderType.LOCAL,
                display_name="Local CodeLlama",
                description="Local code-specific language model",
                supported_languages={
                    "python",
                    "javascript",
                    "typescript",
                    "java",
                    "cpp",
                    "csharp",
                    "go",
                    "ruby",
                    "php",
                    "rust",
                    "sql",
                },
                specialized_languages={"python", "cpp"},
                strengths={
                    "code_generation": 0.75,
                    "debugging": 0.7,
                    "refactoring": 0.65,
                    "documentation": 0.7,
                },
                max_tokens=16384,
                supports_streaming=True,
                avg_latency_ms=200,
                throughput_rps=100,
                concurrency_limit=5,  # Limited by local resources
                cost_per_token=0.0,  # Free but uses local compute
                accuracy_score=0.75,
                consistency_score=0.8,
                creativity_score=0.65,
                rate_limit_rpm=1000,  # Only limited by hardware
                context_window=16384,
                api_endpoint="http://localhost:11434/api/generate",
                requires_api_key=False,
                tags=["local", "private", "free"],
            )
        )

        # GPT-4 (General purpose but good at code)
        self.register_provider(
            ProviderCapabilities(
                provider_id="gpt-4",
                provider_type=ProviderType.CLOUD,
                display_name="GPT-4",
                description="OpenAI's most capable model, excellent for complex tasks",
                supported_languages={
                    "python",
                    "javascript",
                    "typescript",
                    "java",
                    "cpp",
                    "csharp",
                    "rust",
                    "go",
                    "ruby",
                    "php",
                    "swift",
                    "kotlin",
                    "r",
                    "sql",
                    "html",
                    "css",
                    "bash",
                    "powershell",
                    "yaml",
                    "json",
                },
                specialized_languages={"python", "javascript", "typescript"},
                strengths={
                    "architecture": 0.9,
                    "code_review": 0.85,
                    "debugging": 0.85,
                    "documentation": 0.85,
                    "code_generation": 0.8,
                },
                max_tokens=8192,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=True,
                avg_latency_ms=3000,
                throughput_rps=10,
                concurrency_limit=20,
                cost_per_token=0.00003,
                accuracy_score=0.9,
                consistency_score=0.88,
                creativity_score=0.9,
                rate_limit_rpm=40,
                rate_limit_tpd=500000,
                context_window=8192,
                api_endpoint="https://api.openai.com/v1/chat/completions",
                tags=["premium", "versatile", "reasoning"],
            )
        )

    def register_provider(self, capabilities: ProviderCapabilities) -> None:
        """Register a new provider or update existing one."""
        self.providers[capabilities.provider_id] = capabilities
        logger.info(f"Registered provider: {capabilities.provider_id}")

    def unregister_provider(self, provider_id: str) -> bool:
        """Unregister a provider."""
        if provider_id in self.providers:
            del self.providers[provider_id]
            logger.info(f"Unregistered provider: {provider_id}")
            return True
        return False

    def get_provider(self, provider_id: str) -> Optional[ProviderCapabilities]:
        """Get provider capabilities by ID."""
        return self.providers.get(provider_id)

    def get_capabilities(self, provider_id: str) -> Optional[ProviderCapabilities]:
        """Alias for get_provider."""
        return self.get_provider(provider_id)

    def get_all_providers(self) -> List[str]:
        """Get list of all registered provider IDs."""
        return list(self.providers.keys())

    def get_active_providers(self) -> List[str]:
        """Get list of active provider IDs."""
        return [
            pid
            for pid, caps in self.providers.items()
            if caps.status == ProviderStatus.ACTIVE
        ]

    def get_providers_by_language(self, language: str) -> List[str]:
        """Get providers that support a specific language."""
        language_lower = language.lower()
        return [
            pid
            for pid, caps in self.providers.items()
            if language_lower in caps.supported_languages
        ]

    def get_providers_by_task_type(self, task_type: str) -> List[Tuple[str, float]]:
        """Get providers ranked by their strength for a task type."""
        providers_with_scores = []

        for pid, caps in self.providers.items():
            score = caps.strengths.get(task_type, 0.5)  # Default score of 0.5
            providers_with_scores.append((pid, score))
        # Sort by score descending
        providers_with_scores.sort(key=lambda x: x[1], reverse=True)
        return providers_with_scores

    def get_providers_by_cost(
        self, max_cost: Optional[float] = None
    ) -> List[Tuple[str, float]]:
        """Get providers sorted by cost per token."""
        providers_with_cost = []

        for pid, caps in self.providers.items():
            if max_cost is None or caps.cost_per_token <= max_cost:
                providers_with_cost.append((pid, caps.cost_per_token))
        # Sort by cost ascending
        providers_with_cost.sort(key=lambda x: x[1])
        return providers_with_cost

    def update_provider_status(self, provider_id: str, status: ProviderStatus) -> bool:
        """Update provider status."""
        if provider_id in self.providers:
            self.providers[provider_id].status = status
            self.providers[provider_id].last_updated = datetime.now()
            logger.info(f"Updated {provider_id} status to {status.value}")
            return True
        return False

    def update_provider_metrics(
        self,
        provider_id: str,
        latency_ms: Optional[float] = None,
        accuracy: Optional[float] = None,
        consistency: Optional[float] = None,
    ) -> bool:
        """Update provider performance metrics."""
        if provider_id not in self.providers:
            return False

        provider = self.providers[provider_id]

        if latency_ms is not None:
            # Exponential moving average
            provider.avg_latency_ms = 0.7 * provider.avg_latency_ms + 0.3 * latency_ms

        if accuracy is not None:
            provider.accuracy_score = 0.8 * provider.accuracy_score + 0.2 * accuracy

        if consistency is not None:
            provider.consistency_score = (
                0.8 * provider.consistency_score + 0.2 * consistency
            )

        provider.last_updated = datetime.now()
        return True

    def save_config(self, path: Optional[Path] = None) -> None:
        """Save provider configuration to file."""
        save_path = path or self.config_path
        if not save_path:
            raise ValueError("No save path specified")
        config = {}
        for pid, caps in self.providers.items():
            config[pid] = {
                "provider_type": caps.provider_type.value,
                "display_name": caps.display_name,
                "description": caps.description,
                "supported_languages": list(caps.supported_languages),
                "specialized_languages": list(caps.specialized_languages),
                "strengths": caps.strengths,
                "max_tokens": caps.max_tokens,
                "supports_streaming": caps.supports_streaming,
                "supports_function_calling": caps.supports_function_calling,
                "supports_vision": caps.supports_vision,
                "supports_code_execution": caps.supports_code_execution,
                "avg_latency_ms": caps.avg_latency_ms,
                "throughput_rps": caps.throughput_rps,
                "concurrency_limit": caps.concurrency_limit,
                "cost_per_token": caps.cost_per_token,
                "cost_per_request": caps.cost_per_request,
                "free_tier_tokens": caps.free_tier_tokens,
                "accuracy_score": caps.accuracy_score,
                "consistency_score": caps.consistency_score,
                "creativity_score": caps.creativity_score,
                "rate_limit_rpm": caps.rate_limit_rpm,
                "rate_limit_tpd": caps.rate_limit_tpd,
                "context_window": caps.context_window,
                "api_endpoint": caps.api_endpoint,
                "api_version": caps.api_version,
                "requires_api_key": caps.requires_api_key,
                "status": caps.status.value,
                "tags": caps.tags,
                "metadata": caps.metadata,
            }

        with open(save_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved provider configuration to {save_path}")

    def load_config(self, path: Path) -> None:
        """Load provider configuration from file."""
        with open(path, "r") as f:
            config = json.load(f)
        for pid, data in config.items():
            caps = ProviderCapabilities(
                provider_id=pid,
                provider_type=ProviderType(data["provider_type"]),
                display_name=data["display_name"],
                description=data["description"],
                supported_languages=set(data["supported_languages"]),
                specialized_languages=set(data.get("specialized_languages", [])),
                strengths=data.get("strengths", {}),
                max_tokens=data.get("max_tokens", 4096),
                supports_streaming=data.get("supports_streaming", True),
                supports_function_calling=data.get("supports_function_calling", False),
                supports_vision=data.get("supports_vision", False),
                supports_code_execution=data.get("supports_code_execution", False),
                avg_latency_ms=data.get("avg_latency_ms", 1000),
                throughput_rps=data.get("throughput_rps", 10),
                concurrency_limit=data.get("concurrency_limit", 10),
                cost_per_token=data.get("cost_per_token", 0.001),
                cost_per_request=data.get("cost_per_request", 0.0),
                free_tier_tokens=data.get("free_tier_tokens", 0),
                accuracy_score=data.get("accuracy_score", 0.85),
                consistency_score=data.get("consistency_score", 0.8),
                creativity_score=data.get("creativity_score", 0.7),
                rate_limit_rpm=data.get("rate_limit_rpm", 60),
                rate_limit_tpd=data.get("rate_limit_tpd", 100000),
                context_window=data.get("context_window", 8192),
                api_endpoint=data.get("api_endpoint", ""),
                api_version=data.get("api_version", "v1"),
                requires_api_key=data.get("requires_api_key", True),
                status=ProviderStatus(data.get("status", "active")),
                tags=data.get("tags", []),
                metadata=data.get("metadata", {}),
            )
            self.register_provider(caps)
        logger.info(f"Loaded provider configuration from {path}")

    def get_provider_summary(self, provider_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of provider capabilities."""
        provider = self.get_provider(provider_id)
        if not provider:
            return None

        return {
            "id": provider.provider_id,
            "name": provider.display_name,
            "type": provider.provider_type.value,
            "status": provider.status.value,
            "languages": len(provider.supported_languages),
            "specialized_in": list(provider.specialized_languages),
            "best_for": [k for k, v in provider.strengths.items() if v >= 0.8],
            "cost_per_1k_tokens": provider.cost_per_token * 1000,
            "avg_latency_ms": provider.avg_latency_ms,
            "context_window": provider.context_window,
            "features": {
                "streaming": provider.supports_streaming,
                "functions": provider.supports_function_calling,
                "vision": provider.supports_vision,
                "code_execution": provider.supports_code_execution,
            },
        }

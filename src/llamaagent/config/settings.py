"""
Advanced configuration management for LlamaAgent with validation and environment support.

Author: Nik Jois <nikjois@llamasearch.ai>

This module provides comprehensive configuration management, including:
- Environment-based configuration
- YAML/JSON configuration files
- Configuration validation and defaults
- Secret management
- Runtime configuration updates
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from pydantic import BaseSettings, Field
    from pydantic_settings import SettingsConfigDict
    PYDANTIC_AVAILABLE = True
except ImportError:
    # Fallback for older versions or missing pydantic
    BaseSettings = object
    PYDANTIC_AVAILABLE = False
    
    def Field(default, **kwargs):
        return default
    
    SettingsConfigDict = dict

logger = logging.getLogger(__name__)


class LogLevel(str, Enum):
    """Supported log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    MOCK = "mock"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    MLX = "mlx"


class AgentRole(str, Enum):
    """Agent role types."""
    GENERALIST = "generalist"
    PLANNER = "planner"
    EXECUTOR = "executor"
    ANALYST = "analyst"


@dataclass
class LLMConfig:
    """LLM provider configuration."""
    provider: LLMProvider = LLMProvider.MOCK
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    stream: bool = False
    context_window: int = 4096
    supports_tools: bool = True
    supports_vision: bool = False


@dataclass
class AgentConfig:
    """Agent configuration."""
    role: AgentRole = AgentRole.GENERALIST
    max_iterations: int = 10
    verbose: bool = False
    debug: bool = False
    spree_enabled: bool = True
    tool_timeout: int = 30
    memory_enabled: bool = True
    reasoning_enabled: bool = True
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.tool_timeout < 1:
            raise ValueError("tool_timeout must be positive")


@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: Optional[str] = None
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    auto_migrate: bool = True
    vector_dimensions: int = 384


@dataclass
class SecurityConfig:
    """Security configuration."""
    secret_key: str = "development-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 30
    api_key_expire_days: int = 365
    rate_limit_enabled: bool = True
    max_requests_per_minute: int = 60
    max_requests_per_hour: int = 1000
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    trusted_hosts: List[str] = field(default_factory=list)


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    enable_metrics: bool = True
    metrics_port: int = 9090
    enable_tracing: bool = False
    enable_health_checks: bool = True
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "json"
    sentry_dsn: Optional[str] = None
    enable_profiling: bool = False


@dataclass
class CacheConfig:
    """Cache configuration."""
    enabled: bool = True
    backend: str = "memory"  # memory, redis, file
    ttl: int = 3600  # seconds
    max_size: int = 1000
    redis_url: Optional[str] = None
    file_path: Optional[str] = None


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    debug: bool = False
    access_log: bool = True
    
    # Request limits
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    request_timeout: float = 300.0
    keepalive_timeout: float = 5.0


@dataclass
class StorageConfig:
    """Storage configuration."""
    data_directory: str = "./data"
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_extensions: List[str] = field(default_factory=lambda: [".txt", ".json", ".csv"])
    cleanup_interval: int = 3600  # seconds


class LlamaAgentSettings:
    """Main configuration settings with environment variable support."""

    def __init__(self, **kwargs):
        # Core settings
        self.environment: str = "development"
        self.debug: bool = False

        # Component configurations
        self.llm = LLMConfig()
        self.agent = AgentConfig()
        self.database = DatabaseConfig()
        self.security = SecurityConfig()
        self.monitoring = MonitoringConfig()
        self.cache = CacheConfig()
        self.api = APIConfig()
        self.storage = StorageConfig()

        # Apply any provided kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Initialize from environment if using pydantic
        if PYDANTIC_AVAILABLE:
            try:
                # Load from environment variables
                self._load_from_env()
            except Exception as e:
                logger.warning(f"Failed to load from environment: {e}")
        else:
            # Fallback initialization
            self._load_from_env_fallback()

    def _load_from_env_fallback(self):
        """Fallback environment loading when pydantic is not available."""
        # LLM configuration
        if provider := os.getenv("LLAMAAGENT_LLM__PROVIDER"):
            self.llm.provider = LLMProvider(provider)
        
        if model := os.getenv("LLAMAAGENT_LLM__MODEL"):
            self.llm.model = model
        
        if api_key := os.getenv("LLAMAAGENT_LLM__API_KEY"):
            self.llm.api_key = api_key
        
        if base_url := os.getenv("LLAMAAGENT_LLM__BASE_URL"):
            self.llm.base_url = base_url
        
        if temperature := os.getenv("LLAMAAGENT_LLM__TEMPERATURE"):
            try:
                self.llm.temperature = float(temperature)
            except ValueError:
                logger.warning(f"Invalid temperature: {temperature}")
        
        # Agent configuration
        if role := os.getenv("LLAMAAGENT_AGENT__ROLE"):
            self.agent.role = AgentRole(role)
        
        if max_iterations := os.getenv("LLAMAAGENT_AGENT__MAX_ITERATIONS"):
            try:
                self.agent.max_iterations = int(max_iterations)
            except ValueError:
                logger.warning(f"Invalid max_iterations: {max_iterations}")
        
        if verbose := os.getenv("LLAMAAGENT_AGENT__VERBOSE"):
            self.agent.verbose = verbose.lower() in ("true", "1", "yes")
        
        if debug := os.getenv("LLAMAAGENT_DEBUG"):
            self.debug = debug.lower() in ("true", "1", "yes")
            self.agent.debug = self.debug
        
        # Database configuration
        if db_url := os.getenv("DATABASE_URL"):
            self.database.url = db_url
        
        # Security configuration
        if secret := os.getenv("SECRET_KEY"):
            self.security.secret_key = secret
        
        # API configuration
        if host := os.getenv("LLAMAAGENT_API__HOST"):
            self.api.host = host
        
        if port := os.getenv("LLAMAAGENT_API__PORT"):
            try:
                self.api.port = int(port)
            except ValueError:
                logger.warning(f"Invalid port: {port}")

    def _load_from_env(self):
        """Load configuration from environment variables using pydantic."""
        # This would be implemented if pydantic is available
        pass

    @classmethod
    def from_file(cls, config_path: str) -> 'LlamaAgentSettings':
        """Load configuration from file."""
        try:
            config_path = Path(config_path)
            
            if not config_path.exists():
                logger.warning(f"Config file not found: {config_path}")
                return cls()
            
            with open(config_path, 'r') as f:
                if config_path.suffix in [".yaml", ".yml"]:
                    if not YAML_AVAILABLE:
                        raise RuntimeError("YAML support not available")
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            return cls.from_dict(config_data)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return cls()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LlamaAgentSettings':
        """Create settings from dictionary."""
        # Convert nested dicts to dataclass instances
        if "llm" in config_dict:
            config_dict["llm"] = LLMConfig(**config_dict["llm"])
        
        if "agent" in config_dict:
            config_dict["agent"] = AgentConfig(**config_dict["agent"])
        
        if "database" in config_dict:
            config_dict["database"] = DatabaseConfig(**config_dict["database"])
        
        if "security" in config_dict:
            config_dict["security"] = SecurityConfig(**config_dict["security"])
        
        if "monitoring" in config_dict:
            config_dict["monitoring"] = MonitoringConfig(**config_dict["monitoring"])
        
        if "cache" in config_dict:
            config_dict["cache"] = CacheConfig(**config_dict["cache"])
        
        if "api" in config_dict:
            config_dict["api"] = APIConfig(**config_dict["api"])
        
        if "storage" in config_dict:
            config_dict["storage"] = StorageConfig(**config_dict["storage"])
        
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            "environment": self.environment,
            "debug": self.debug,
            "llm": self.llm.__dict__,
            "agent": self.agent.__dict__,
            "database": self.database.__dict__,
            "security": self.security.__dict__,
            "monitoring": self.monitoring.__dict__,
            "cache": self.cache.__dict__,
            "api": self.api.__dict__,
            "storage": self.storage.__dict__
        }

    def to_file(self, config_path: str) -> None:
        """Save configuration to file."""
        config_path = Path(config_path)
        config_dict = self.to_dict()
        
        # Create directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            if config_path.suffix in [".yaml", ".yml"]:
                if not YAML_AVAILABLE:
                    raise RuntimeError("YAML support not available")
                yaml.dump(config_dict, f, default_flow_style=False)
            else:
                json.dump(config_dict, f, indent=2)

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Validate LLM configuration
        if (
            self.llm.provider in [LLMProvider.OPENAI, LLMProvider.ANTHROPIC]
            and not self.llm.api_key
        ):
            errors.append(f"API key required for {self.llm.provider.value} provider")
        
        if self.llm.temperature < 0 or self.llm.temperature > 2:
            errors.append("Temperature must be between 0 and 2")
        
        if self.llm.max_tokens < 1:
            errors.append("Max tokens must be positive")
        
        # Validate agent configuration
        if self.agent.max_iterations < 1:
            errors.append("Max iterations must be positive")
        
        # Validate API configuration
        if self.api.port < 1 or self.api.port > 65535:
            errors.append("API port must be between 1 and 65535")
        
        # Validate security configuration
        if len(self.security.secret_key) < 16:
            errors.append("Secret key must be at least 16 characters long")
        
        return errors

    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"

    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"


# Global settings instance
settings = LlamaAgentSettings()


def get_settings() -> LlamaAgentSettings:
    """Get the global settings instance."""
    return settings


def load_settings(config_path: Optional[str] = None) -> LlamaAgentSettings:
    """Load settings from file or environment."""
    global settings
    
    if config_path:
        settings = LlamaAgentSettings.from_file(config_path)
    else:
        # Try to load from default locations
        default_paths = [
            "config.yaml",
            "config.yml",
            "config.json",
            "llamaagent.yaml",
            "llamaagent.yml",
            "llamaagent.json"
        ]
        
        for path in default_paths:
            if Path(path).exists():
                settings = LlamaAgentSettings.from_file(path)
                break
        else:
            # No config file found, use environment variables
            settings = LlamaAgentSettings()
    
    return settings


def update_settings(**kwargs) -> None:
    """Update global settings."""
    global settings
    if settings is None:
        settings = get_settings()

    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)

    # Validate configuration
    errors = settings.validate()
    if errors:
        logger.warning(f"Configuration validation errors: {errors}")


def reload_settings():
    """Reload settings from environment and files."""
    global settings
    settings = None
    return get_settings()


# Convenience accessors
def get_llm_config() -> LLMConfig:
    """Get LLM configuration."""
    return get_settings().llm


def get_agent_config() -> AgentConfig:
    """Get agent configuration."""
    return get_settings().agent


def get_database_config() -> DatabaseConfig:
    """Get database configuration."""
    return get_settings().database


def get_api_config() -> APIConfig:
    """Get API configuration."""
    return get_settings().api


def get_security_config() -> SecurityConfig:
    """Get security configuration."""
    return get_settings().security


def get_monitoring_config() -> MonitoringConfig:
    """Get monitoring configuration."""
    return get_settings().monitoring


def get_cache_config() -> CacheConfig:
    """Get cache configuration."""
    return get_settings().cache


def get_storage_config() -> StorageConfig:
    """Get storage configuration."""
    return get_settings().storage
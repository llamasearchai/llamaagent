"""Configuration module for LlamaAgent.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import json
import os
from typing import Any, Dict, Optional


class ConfigManager:
    """Basic configuration manager."""

    def __init__(self):
        self.config: Dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.config[key] = value


def get_config(key: str, default: Any = None) -> Any:
    """Convenience accessor used in tests.

    Loads from environment variables first, then in-memory defaults.
    """
    env_key = key.upper().replace(".", "_")
    if env_key in os.environ:
        return os.environ[env_key]
    return ConfigManager().get(key, default)


__all__ = ['ConfigManager', 'get_config']

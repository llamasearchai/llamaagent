"""Configuration module for LlamaAgent."""

from typing import Any, Dict, Optional


class ConfigManager:
    """Basic configuration manager."""
    
    def __init__(self):
        self.config = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.config[key] = value

__all__ = ['ConfigManager']

#!/usr/bin/env python3
"""
Configuration Management System
Author: Nik Jois <nikjois@llamasearch.ai>

This module provides:
- Environment-based configuration
- User preferences management
- API key and credential storage
- Model and provider settings
- Custom configuration validation
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml

    yaml_available = True
except ImportError:
    yaml_available = False

try:
    from cryptography.fernet import Fernet

    crypto_available = True
except ImportError:
    crypto_available = False

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for AI models."""

    name: str
    provider: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 2000
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 30


@dataclass
class ProviderConfig:
    """Configuration for AI providers."""

    name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    models: List[str] = field(default_factory=list)
    default_model: Optional[str] = None
    rate_limit: Optional[Dict[str, Any]] = None


@dataclass
class CLIConfig:
    """Configuration for CLI settings."""

    default_model: str = "gpt-4"
    shell_interaction: bool = True
    use_functions: bool = True
    debug_mode: bool = False
    theme: str = "default"
    max_history: int = 100
    auto_save: bool = True


class ConfigManager:
    """Configuration management system."""

    def __init__(self, config_dir: Optional[str] = None):
        if config_dir is None:
            config_dir = os.path.expanduser("~/.config/llamaagent")

        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Configuration files
        self.config_file = self.config_dir / "config.json"
        self.user_prefs_file = self.config_dir / "preferences.json"
        self.secrets_file = self.config_dir / "secrets.encrypted"
        self.key_file = self.config_dir / "key.bin"

        # Initialize encryption if available
        if crypto_available:
            self._setup_encryption()

        # Load configurations
        self.config = self._load_config()
        self.user_prefs = self._load_user_preferences()
        self.secrets = self._load_secrets() if crypto_available else {}

    def _setup_encryption(self) -> None:
        """Setup encryption for sensitive data."""
        if not CRYPTO_AVAILABLE:
            logger.warning("Cryptography not available - secrets will not be encrypted")
            return

        if self.key_file.exists():
            with open(self.key_file, "rb") as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_file, "wb") as f:
                f.write(key)
            # Make key file readable only by owner
            os.chmod(self.key_file, 0o600)

        self.cipher = Fernet(key)

    def _load_config(self) -> Dict[str, Any]:
        """Load main configuration."""
        default_config = {
            "cli": {
                "default_model": "gpt-4",
                "shell_interaction": True,
                "use_functions": True,
                "debug_mode": False,
                "theme": "default",
                "max_history": 100,
                "auto_save": True,
            },
            "providers": {
                "openai": {
                    "name": "openai",
                    "models": ["gpt-4", "gpt-3.5-turbo"],
                    "default_model": "gpt-4",
                },
                "anthropic": {
                    "name": "anthropic",
                    "models": ["claude-3-opus", "claude-3-sonnet"],
                    "default_model": "claude-3-sonnet",
                },
            },
            "models": {
                "gpt-4": {
                    "name": "gpt-4",
                    "provider": "openai",
                    "max_tokens": 2000,
                    "temperature": 0.7,
                },
                "claude-3-sonnet": {
                    "name": "claude-3-sonnet",
                    "provider": "anthropic",
                    "max_tokens": 2000,
                    "temperature": 0.7,
                },
            },
        }

        if not self.config_file.exists():
            self._save_config(default_config)
            return default_config

        try:
            with open(self.config_file, "r") as f:
                if self.config_file.suffix == ".yaml" and YAML_AVAILABLE:
                    loaded_config = yaml.safe_load(f)
                else:
                    loaded_config = json.load(f)

            # Merge with defaults
            return self._merge_config(default_config, loaded_config)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return default_config

    def _load_user_preferences(self) -> Dict[str, Any]:
        """Load user preferences."""
        if not self.user_prefs_file.exists():
            return {}

        try:
            with open(self.user_prefs_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading user preferences: {e}")
            return {}

    def _load_secrets(self) -> Dict[str, Any]:
        """Load encrypted secrets."""
        if not CRYPTO_AVAILABLE or not self.secrets_file.exists():
            return {}

        try:
            with open(self.secrets_file, "rb") as f:
                encrypted_data = f.read()

            decrypted_data = self.cipher.decrypt(encrypted_data)
            return json.loads(decrypted_data.decode())
        except Exception as e:
            logger.error(f"Error loading secrets: {e}")
            return {}

    def _save_config(self, config: Dict[str, Any]) -> None:
        """Save main configuration."""
        try:
            with open(self.config_file, "w") as f:
                if self.config_file.suffix == ".yaml" and YAML_AVAILABLE:
                    yaml.dump(config, f, default_flow_style=False)
                else:
                    json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {e}")

    def _save_user_preferences(self, prefs: Dict[str, Any]) -> None:
        """Save user preferences."""
        try:
            with open(self.user_prefs_file, "w") as f:
                json.dump(prefs, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving user preferences: {e}")

    def _save_secrets(self, secrets: Dict[str, Any]) -> None:
        """Save encrypted secrets."""
        if not CRYPTO_AVAILABLE:
            logger.warning("Cannot save secrets - cryptography not available")
            return

        try:
            data = json.dumps(secrets)
            encrypted_data = self.cipher.encrypt(data.encode())

            with open(self.secrets_file, "wb") as f:
                f.write(encrypted_data)

            # Make secrets file readable only by owner
            os.chmod(self.secrets_file, 0o600)
        except Exception as e:
            logger.error(f"Error saving secrets: {e}")

    def _merge_config(
        self, base: Dict[str, Any], overlay: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge loaded config with defaults."""
        result = base.copy()

        for key, value in overlay.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value

        return result

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key path (e.g., 'cli.default_model')."""
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key path."""
        keys = key.split(".")
        target = self.config

        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]

        target[keys[-1]] = value
        self._save_config(self.config)

    def get_user_pref(self, key: str, default: Any = None) -> Any:
        """Get user preference."""
        return self.user_prefs.get(key, default)

    def set_user_pref(self, key: str, value: Any) -> None:
        """Set user preference."""
        self.user_prefs[key] = value
        self._save_user_preferences(self.user_prefs)

    def get_secret(self, key: str, default: Any = None) -> Any:
        """Get secret value."""
        return self.secrets.get(key, default)

    def set_secret(self, key: str, value: Any) -> None:
        """Set secret value."""
        self.secrets[key] = value
        self._save_secrets(self.secrets)

    def get_cli_config(self) -> CLIConfig:
        """Get CLI configuration."""
        cli_data = self.config.get("cli", {})
        return CLIConfig(**cli_data)

    def get_provider_config(self, provider: str) -> Optional[ProviderConfig]:
        """Get provider configuration."""
        provider_data = self.config.get("providers", {}).get(provider)
        if not provider_data:
            return None

        # Add API key from secrets if available
        api_key = self.get_secret(f"{provider}_api_key")
        if api_key:
            provider_data["api_key"] = api_key

        return ProviderConfig(**provider_data)

    def get_model_config(self, model: str) -> Optional[ModelConfig]:
        """Get model configuration."""
        model_data = self.config.get("models", {}).get(model)
        if not model_data:
            return None

        # Add API key from secrets if available
        provider = model_data.get("provider")
        if provider:
            api_key = self.get_secret(f"{provider}_api_key")
            if api_key:
                model_data["api_key"] = api_key

        return ModelConfig(**model_data)

    def set_api_key(self, provider: str, api_key: str) -> None:
        """Set up API key for a provider."""
        self.set_secret(f"{provider}_api_key", api_key)
        logger.info(f"API key set for {provider}")

    def validate_config(self) -> List[str]:
        """Validate configuration and return any issues."""
        issues = []

        # Check required providers
        providers = self.config.get("providers", {})
        if not providers:
            issues.append("No providers configured")

        # Check for API keys
        for provider_name in providers.keys():
            api_key = self.get_secret(f"{provider_name}_api_key")
            if not api_key:
                issues.append(f"No API key configured for {provider_name}")

        # Check models
        models = self.config.get("models", {})
        if not models:
            issues.append("No models configured")

        # Check CLI config
        cli_config = self.get_cli_config()
        if cli_config.default_model not in models:
            issues.append(
                f"Default model '{cli_config.default_model}' not found in configured models"
            )

        return issues

    def export_config(self, export_path: str, include_secrets: bool = False) -> bool:
        """Export configuration to a file."""
        export_data = {"config": self.config, "user_preferences": self.user_prefs}

        if include_secrets:
            export_data["secrets"] = self.secrets

        try:
            with open(export_path, "w") as f:
                if export_path.endswith(".yaml") and YAML_AVAILABLE:
                    yaml.dump(export_data, f, default_flow_style=False)
                else:
                    json.dump(export_data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error exporting config: {e}")
            return False

    def import_config(self, import_path: str, merge: bool = True) -> bool:
        """Import configuration from a file."""
        try:
            with open(import_path, "r") as f:
                if import_path.endswith(".yaml") or import_path.endswith(".yml"):
                    import_data = yaml.safe_load(f) if YAML_AVAILABLE else {}
                else:
                    import_data = json.load(f)

            if "config" in import_data:
                if merge:
                    self.config = self._merge_config(self.config, import_data["config"])
                else:
                    self.config = import_data["config"]
                self._save_config(self.config)

            if "user_preferences" in import_data:
                if merge:
                    self.user_prefs.update(import_data["user_preferences"])
                else:
                    self.user_prefs = import_data["user_preferences"]
                self._save_user_preferences(self.user_prefs)

            if "secrets" in import_data and CRYPTO_AVAILABLE:
                if merge:
                    self.secrets.update(import_data["secrets"])
                else:
                    self.secrets = import_data["secrets"]
                self._save_secrets(self.secrets)

            return True
        except Exception as e:
            logger.error(f"Error importing config: {e}")
            return False


def setup_interactive_config() -> None:
    """Interactive configuration setup."""
    print("Configuration LlamaAgent Configuration Setup")
    print("=" * 40)

    manager = ConfigManager()

    # API Keys setup
    print("\nLIST: API Keys Setup:")

    # OpenAI
    openai_key = input("OpenAI API Key (leave empty to skip): ").strip()
    if openai_key:
        manager.set_api_key("openai", openai_key)
        print("PASS OpenAI API key set")

    # Anthropic
    anthropic_key = input("Anthropic API Key (leave empty to skip): ").strip()
    if anthropic_key:
        manager.set_api_key("anthropic", anthropic_key)
        print("PASS Anthropic API key set")

    # CLI preferences
    print("\nLIST: CLI Preferences:")

    default_model = input(
        f"Default model [{manager.get('cli.default_model')}]: "
    ).strip()
    if default_model:
        manager.set("cli.default_model", default_model)

    shell_interaction = input("Enable shell interaction? [Y/n]: ").strip().lower()
    manager.set("cli.shell_interaction", shell_interaction not in ['n', 'no'])

    use_functions = input("Enable function calling? [Y/n]: ").strip().lower()
    manager.set("cli.use_functions", use_functions not in ['n', 'no'])

    debug_mode = input("Enable debug mode? [y/N]: ").strip().lower()
    manager.set("cli.debug_mode", debug_mode in ['y', 'yes'])

    print("\nPASS Configuration setup complete!")

    # Validate configuration
    issues = manager.validate_config()
    if issues:
        print("\nWARNING:  Configuration issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nPASS Configuration is valid")


def setup_environment() -> None:
    """Set up environment variables from configuration."""
    manager = ConfigManager()

    # Set up API keys as environment variables
    for provider in ["openai", "anthropic"]:
        api_key = manager.get_secret(f"{provider}_api_key")
        if api_key:
            env_var = f"{provider.upper()}_API_KEY"
            os.environ[env_var] = api_key


# Global config manager instance
_config_manager = None


def get_config_manager() -> ConfigManager:
    """Get global config manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value using global config manager."""
    return get_config_manager().get(key, default)


def set_config(key: str, value: Any) -> None:
    """Set configuration value using global config manager."""
    get_config_manager().set(key, value)


def main() -> None:
    """Example usage of the configuration manager."""
    manager = ConfigManager()

    # Display current configuration
    print("Current configuration:")
    print(f"Default model: {manager.get('cli.default_model')}")
    print(f"Shell interaction: {manager.get('cli.shell_interaction')}")

    # Check for API keys
    openai_key = manager.get_secret("openai_api_key")
    print(f"OpenAI API key configured: {'Yes' if openai_key else 'No'}")

    # Validate configuration
    issues = manager.validate_config()
    if issues:
        print("\nConfiguration issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nPASS Configuration is valid")


if __name__ == "__main__":
    main()

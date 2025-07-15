#!/usr/bin/env python3
"""
Example usage of the LlamaAgent configuration system.
"""

import os
from pathlib import Path

# Set environment variables for demonstration
os.environ["LLAMAAGENT_LLM_PROVIDER"] = "openai"
os.environ["LLAMAAGENT_LLM_MODEL"] = "gpt-4o-mini"
os.environ["LLAMAAGENT_DEBUG"] = "true"
os.environ["LOG_LEVEL"] = "DEBUG"

from llamaagent.config import (
    get_config,
    get_config_manager,
    set_config_file,
    ConfigManager,
    LlamaAgentConfig,
)


def main():
    """Demonstrate configuration usage."""
    
    print("=== LlamaAgent Configuration System Demo ===\n")
    
    # 1. Get default configuration
    print("1. Default Configuration:")
    config = get_config()
    print(f"   Provider: {config.llm.provider}")
    print(f"   Model: {config.llm.model}")
    print(f"   Debug: {config.agent.debug}")
    print(f"   Log Level: {config.logging.level}\n")
    
    # 2. Load from specific file
    config_file = Path(__file__).parent.parent / "config" / "example_config.yaml"
    if config_file.exists():
        print("2. Loading from example_config.yaml:")
        config = get_config(str(config_file))
        print(f"   Provider: {config.llm.provider}")
        print(f"   Model: {config.llm.model}")
        print(f"   Environment: {config.environment}\n")
    
    # 3. Using ConfigManager directly
    print("3. Using ConfigManager directly:")
    manager = ConfigManager()
    config = manager.load_config()
    
    # Get values using dot notation
    print(f"   llm.provider: {manager.get('llm.provider')}")
    print(f"   agent.max_iterations: {manager.get('agent.max_iterations')}")
    print(f"   api.port: {manager.get('api.port')}\n")
    
    # 4. Update configuration at runtime
    print("4. Updating configuration at runtime:")
    manager.set("llm.temperature", 0.9)
    manager.set("agent.max_iterations", 20)
    
    print(f"   New temperature: {manager.get('llm.temperature')}")
    print(f"   New max_iterations: {manager.get('agent.max_iterations')}\n")
    
    # 5. Save configuration
    output_file = Path("/tmp/llamaagent_config_example.yaml")
    print(f"5. Saving configuration to {output_file}:")
    manager.save_config(str(output_file))
    print("   Configuration saved!\n")
    
    # 6. Accessing nested configuration
    print("6. Accessing nested configuration objects:")
    config = get_config()
    print(f"   LLM Config: {config.llm}")
    print(f"   API Config: {config.api}")
    print(f"   Cache Config: {config.cache}\n")
    
    # 7. Environment variable override
    print("7. Environment variable overrides:")
    print("   (Set via os.environ at the top of this script)")
    print(f"   LLAMAAGENT_LLM_PROVIDER -> {config.llm.provider}")
    print(f"   LLAMAAGENT_DEBUG -> {config.agent.debug}")
    print(f"   LOG_LEVEL -> {config.logging.level}\n")


if __name__ == "__main__":
    main()
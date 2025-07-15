"""
Configuration for Ground Truth Data generation system.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from typing import Any, Dict

from src.llamaagent.data.gdt import DataType

DEFAULT_GDT_CONFIG: Dict[str, Any] = {
    "validation_rules": {
        "min_content_length": 10,
        "max_content_length": 10000,
        "required_fields": ["id", "data_type", "content"],
        "allowed_data_types": [dt.value for dt in DataType]
    },
    "generation_defaults": {
        "text": {
            "min_length": 50,
            "max_length": 500,
            "topics": [
                "artificial intelligence",
                "machine learning", 
                "data science",
                "technology",
                "programming",
                "software development",
                "automation",
                "digital transformation"
            ]
        },
        "conversation": {
            "min_turns": 2,
            "max_turns": 10,
            "contexts": ["general", "technical", "support", "educational"]
        },
        "qa_pair": {
            "difficulty_levels": ["easy", "medium", "hard"],
            "domains": ["general", "technical", "academic", "business"]
        }
    },
    "transformations": {
        "normalize_text": {"enabled": True},
        "add_metadata": {"enabled": True},
        "anonymize": {"enabled": False},
        "format_conversation": {"enabled": True}
    },
    "output": {
        "format": "json",
        "include_metadata": True,
        "include_validation": True,
        "compression": False
    },
    "quality_control": {
        "auto_validate": True,
        "strict_mode": False,
        "retry_failed_generation": True,
        "max_retries": 3
    }
}

def get_gdt_config(environment: str = "default") -> Dict[str, Any]:
    """Get GDT configuration for specific environment."""
    config = DEFAULT_GDT_CONFIG.copy()
    
    if environment == "test":
        config["validation_rules"]["min_content_length"] = 1
        config["generation_defaults"]["text"]["min_length"] = 10
        config["quality_control"]["strict_mode"] = False
    elif environment == "production":
        config["validation_rules"]["min_content_length"] = 20
        config["transformations"]["anonymize"]["enabled"] = True
        config["quality_control"]["strict_mode"] = True
        config["output"]["compression"] = True
    elif environment == "development":
        config["quality_control"]["auto_validate"] = False
        config["output"]["include_metadata"] = True
    
    return config

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate GDT configuration."""
    required_sections = [
        "validation_rules",
        "generation_defaults", 
        "transformations",
        "output"
    ]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate validation rules
    validation_rules = config["validation_rules"]
    if validation_rules["min_content_length"] < 0:
        raise ValueError("min_content_length must be non-negative")
    
    if validation_rules["max_content_length"] <= validation_rules["min_content_length"]:
        raise ValueError("max_content_length must be greater than min_content_length")
    
    return True

def get_generator_config(data_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Get generator-specific configuration."""
    if data_type in config["generation_defaults"]:
        return config["generation_defaults"][data_type]
    return {}

def get_validation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get validation-specific configuration."""
    return config["validation_rules"]

def get_transformation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get transformation-specific configuration."""
    return config["transformations"] 
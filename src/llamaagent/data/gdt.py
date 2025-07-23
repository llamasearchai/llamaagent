"""
Ground Truth Data (GDT) Generation System

This module provides comprehensive data generation, validation, and transformation
capabilities for AI agent training and evaluation.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar, Union

# Type variables for generic types
T = TypeVar('T')


class DataType(str, Enum):
    """Types of data that can be generated and processed."""

    TEXT = "text"
    CONVERSATION = "conversation"
    QA_PAIR = "qa_pair"
    INSTRUCTION = "instruction"
    CLASSIFICATION = "classification"
    EMBEDDING = "embedding"
    STRUCTURED = "structured"


class ValidationStatus(str, Enum):
    """Validation status for data items."""

    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    NEEDS_REVIEW = "needs_review"


@dataclass
class GDTItem:
    """Individual ground truth data item."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    data_type: DataType = DataType.TEXT
    content: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    validation_status: ValidationStatus = ValidationStatus.VALID
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert item to dictionary."""
        return {
            "id": self.id,
            "data_type": self.data_type.value,
            "content": self.content,
            "metadata": self.metadata,
            "tags": self.tags,
            "validation_status": self.validation_status.value,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GDTItem:
        """Create item from dictionary."""
        created_at = (
            datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(timezone.utc)
        )
        return cls(
            id=data["id"],
            data_type=DataType(data["data_type"]),
            content=data["content"],
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
            validation_status=ValidationStatus(data.get("validation_status", "valid")),
            created_at=created_at,
        )


class GDTDataset:
    """Collection of ground truth data items."""

    def __init__(
        self, name: str, description: str = "", items: Optional[List[GDTItem]] = None
    ):
        self.name = name
        self.description = description
        self.items: List[GDTItem] = items or []
        self.created_at = datetime.now(timezone.utc)
        self.metadata: Dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, index: int) -> GDTItem:
        return self.items[index]

    def append(self, item: GDTItem) -> None:
        """Add item to dataset."""
        self.items.append(item)

    def extend(self, items: List[GDTItem]) -> None:
        """Add multiple items to dataset."""
        self.items.extend(items)

    def filter_by_type(self, data_type: DataType) -> List[GDTItem]:
        """Filter items by data type."""
        return [item for item in self.items if item.data_type == data_type]

    def filter_by_tag(self, tag: str) -> List[GDTItem]:
        """Filter items by tag."""
        return [item for item in self.items if tag in item.tags]

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataset to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "items": [item.to_dict() for item in self.items],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GDTDataset:
        """Create dataset from dictionary."""
        dataset = cls(data["name"], data.get("description", ""))
        dataset.created_at = (
            datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(timezone.utc)
        )
        dataset.metadata = data.get("metadata", {})
        dataset.items = [
            GDTItem.from_dict(item_data) for item_data in data.get("items", [])
        ]
        return dataset


class GDTValidator:
    """Validates ground truth data items and datasets."""

    def __init__(self, rules: Optional[Dict[str, Any]] = None):
        self.rules = rules or {
            "min_content_length": 10,
            "max_content_length": 10000,
            "required_fields": ["id", "data_type", "content"],
            "allowed_data_types": [dt.value for dt in DataType],
        }

    def validate_item(self, item: GDTItem) -> Dict[str, Any]:
        """Validate a single item."""
        errors: List[str] = []
        warnings: List[str] = []

        # Check required fields
        if not item.id:
            errors.append("ID is required")

        if not item.content:
            errors.append("Content is required")

        # Check content length for text-based types
        if item.data_type in [DataType.TEXT, DataType.INSTRUCTION]:
            text_content = item.content.get("text", "")
            if len(text_content) < self.rules["min_content_length"]:
                warnings.append(
                    f"Content length ({len(text_content)}) below minimum ({self.rules['min_content_length']})"
                )
            elif len(text_content) > self.rules["max_content_length"]:
                errors.append(
                    f"Content length ({len(text_content)}) exceeds maximum ({self.rules['max_content_length']})"
                )

        # Determine status
        if errors:
            status = ValidationStatus.INVALID
        elif warnings:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.VALID

        return {"status": status, "errors": errors, "warnings": warnings}

    def validate_data(
        self, data: Union[GDTItem, GDTDataset, List[GDTItem]]
    ) -> List[Dict[str, Any]]:
        """Validate data items or datasets."""
        results: List[Dict[str, Any]] = []

        if isinstance(data, GDTItem):
            results.append(self.validate_item(data))
        elif isinstance(data, GDTDataset):
            for item in data.items:
                results.append(self.validate_item(item))
        else:
            # Must be List[GDTItem] since Union only has three types
            for item in data:
                results.append(self.validate_item(item))

        return results


class GDTTransformer:
    """Transforms ground truth data items."""

    def __init__(self):
        self.transformations = {
            "normalize_text": self._normalize_text,
            "add_metadata": self._add_metadata,
            "anonymize": self._anonymize,
            "format_conversation": self._format_conversation,
        }

    def transform_item(
        self, item: GDTItem, transformation: str, **kwargs: Any
    ) -> GDTItem:
        """Transform a single item."""
        if transformation not in self.transformations:
            raise ValueError(f"Unknown transformation: {transformation}")

        # Create a copy of the item
        new_item = GDTItem(
            id=item.id,
            data_type=item.data_type,
            content=item.content.copy(),
            metadata=item.metadata.copy(),
            tags=item.tags.copy(),
            validation_status=item.validation_status,
            created_at=item.created_at,
        )

        # Apply transformation
        self.transformations[transformation](new_item, **kwargs)
        return new_item

    def transform_data(
        self, data: Union[GDTItem, GDTDataset], transformation: str, **kwargs: Any
    ) -> Union[GDTItem, GDTDataset]:
        """Transform data items or datasets."""
        if isinstance(data, GDTItem):
            return self.transform_item(data, transformation, **kwargs)
        else:
            # Must be GDTDataset since Union only has two types
            new_dataset = GDTDataset(data.name, data.description)
            new_dataset.metadata = data.metadata.copy()
            new_dataset.created_at = data.created_at

            for item in data.items:
                transformed_item = self.transform_item(item, transformation, **kwargs)
                new_dataset.append(transformed_item)

            return new_dataset

    def _normalize_text(self, item: GDTItem, **kwargs: Any) -> None:
        """Normalize text content."""
        if "text" in item.content:
            text = item.content["text"]
            # Basic normalization
            text = text.strip().lower()
            item.content["text"] = text

    def _add_metadata(self, item: GDTItem, **kwargs: Any) -> None:
        """Add metadata to item."""
        item.metadata.update(kwargs)

    def _anonymize(self, item: GDTItem, **kwargs: Any) -> None:
        """Anonymize sensitive content."""
        # Basic anonymization - can be extended
        if "text" in item.content:
            # This is a placeholder - real anonymization would be more sophisticated
            item.content["text"] = "[ANONYMIZED]"

    def _format_conversation(self, item: GDTItem, **kwargs: Any) -> None:
        """Format conversation content."""
        if item.data_type == DataType.CONVERSATION and "messages" in item.content:
            # Ensure messages have required fields
            for message in item.content["messages"]:
                if "role" not in message:
                    message["role"] = "user"
                if "content" not in message:
                    message["content"] = ""


class DataGenerator(ABC):
    """Abstract base class for data generators."""

    @abstractmethod
    def generate_item(self, **kwargs: Any) -> GDTItem:
        """Generate a single data item."""


class TextDataGenerator(DataGenerator):
    """Generates text data items."""

    def generate_item(self, **kwargs: Any) -> GDTItem:
        """Generate a text data item."""
        topic = kwargs.get("topic", "general")
        length = kwargs.get("length", 100)

        # This is a simple mock implementation
        # In practice, this would use an LLM or other text generation method
        content = {
            "text": f"This is generated text about {topic} with approximately {length} characters.",
            "topic": topic,
            "length": length,
        }

        return GDTItem(
            data_type=DataType.TEXT,
            content=content,
            tags=["generated", "text", topic],
            metadata={"generator": "TextDataGenerator", "generation_params": kwargs},
        )


class ConversationDataGenerator(DataGenerator):
    """Generates conversation data items."""

    def generate_item(self, **kwargs: Any) -> GDTItem:
        """Generate a conversation data item."""
        turns = kwargs.get("turns", 4)
        context = kwargs.get("context", "general")

        # Generate mock conversation
        messages: List[Dict[str, str]] = []
        for i in range(turns):
            role = "user" if i % 2 == 0 else "assistant"
            content = f"This is turn {i+1} in a {context} conversation."
            messages.append({"role": role, "content": content})

        return GDTItem(
            data_type=DataType.CONVERSATION,
            content={"messages": messages, "context": context},
            tags=["generated", "conversation", context],
            metadata={
                "generator": "ConversationDataGenerator",
                "generation_params": kwargs,
            },
        )


class GDTGenerator:
    """Main generator for ground truth data."""

    def __init__(self):
        self.validator = GDTValidator()
        self.transformer = GDTTransformer()
        self.generators: Dict[DataType, DataGenerator] = {
            DataType.TEXT: TextDataGenerator(),
            DataType.CONVERSATION: ConversationDataGenerator(),
        }

    def register_generator(self, data_type: DataType, generator: DataGenerator) -> None:
        """Register a custom generator for a data type."""
        self.generators[data_type] = generator

    def generate_dataset(
        self, name: str, data_type: DataType, count: int, **kwargs: Any
    ) -> GDTDataset:
        """Generate a dataset of the specified type and size."""
        if data_type not in self.generators:
            raise ValueError(f"No generator registered for data type: {data_type}")

        generator = self.generators[data_type]
        dataset = GDTDataset(
            name, f"Generated dataset of {count} {data_type.value} items"
        )

        for _ in range(count):
            item = generator.generate_item(**kwargs)
            dataset.append(item)

        return dataset

    def validate_data(
        self, data: Union[GDTItem, GDTDataset, List[GDTItem]]
    ) -> List[Dict[str, Any]]:
        """Validate data using the configured validator."""
        return self.validator.validate_data(data)

    def transform_data(
        self, data: Union[GDTItem, GDTDataset], transformation: str, **kwargs: Any
    ) -> Union[GDTItem, GDTDataset]:
        """Transform data using the configured transformer."""
        return self.transformer.transform_data(data, transformation, **kwargs)

    def save_dataset(self, dataset: GDTDataset, path: Path) -> None:
        """Save dataset to file."""
        with open(path, 'w') as f:
            json.dump(dataset.to_dict(), f, indent=2)

    def load_dataset(self, path: Path) -> GDTDataset:
        """Load dataset from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return GDTDataset.from_dict(data)

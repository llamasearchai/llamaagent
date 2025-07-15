"""
Comprehensive tests for the GDT (Ground Truth Data) system.
Tests data generation, validation, transformation, and persistence.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.llamaagent.data.gdt import (
    ConversationDataGenerator,
    DataType,
    GDTDataset,
    GDTGenerator,
    GDTItem,
    GDTTransformer,
    GDTValidator,
    TextDataGenerator,
    ValidationStatus,
)


class MockGDTGenerator:
    """Mock GDT generator for testing."""

    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self.responses = responses
        self.call_count = 0
        self.validator = GDTValidator()
        self.transformer = GDTTransformer()

    def generate_dataset(
        self, name: str, data_type: DataType, count: int, **kwargs: Any
    ) -> GDTDataset:
        """Generate mock dataset."""
        items: List[GDTItem] = []
        for i in range(count):
            response_data = self.responses[i % len(self.responses)]
            item = GDTItem(
                id=f"test_item_{i}",
                data_type=data_type,
                content=response_data,
                tags=["test", "mock"],
            )
            items.append(item)

        self.call_count += 1
        return GDTDataset(name=name, description="Mock dataset", items=items)

    def validate_data(self, data: Any) -> List[Dict[str, Any]]:
        """Mock validation method."""
        return self.validator.validate_data(data)

    def transform_data(self, data: Any, transformation: str, **kwargs: Any) -> Any:
        """Mock transformation method."""
        return self.transformer.transform_data(data, transformation, **kwargs)


@pytest.fixture
def mock_gdt_generator() -> MockGDTGenerator:
    """Provide mock GDT generator for testing."""
    mock_responses = [
        {"text": "Sample text content", "topic": "AI"},
        {"text": "Another sample text", "topic": "ML"},
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        },
    ]
    return MockGDTGenerator(mock_responses)


@pytest.fixture
def sample_gdt_items() -> List[GDTItem]:
    """Provide sample GDT items for testing."""
    return [
        GDTItem(
            id="item_1",
            data_type=DataType.TEXT,
            content={"text": "Sample text", "topic": "testing"},
            tags=["test", "sample"],
        ),
        GDTItem(
            id="item_2",
            data_type=DataType.CONVERSATION,
            content={
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ]
            },
            tags=["test", "conversation"],
        ),
    ]


@pytest.fixture
def sample_dataset(sample_gdt_items: List[GDTItem]) -> GDTDataset:
    """Provide sample dataset for testing."""
    return GDTDataset(
        name="test_dataset",
        description="Sample dataset for testing",
        items=sample_gdt_items,
    )


class TestGDTGenerator:
    """Test suite for GDT generator functionality."""

    def test_generate_dataset_basic(self, mock_gdt_generator: MockGDTGenerator) -> None:
        """Test basic dataset generation."""
        dataset = mock_gdt_generator.generate_dataset("test", DataType.TEXT, 3)

        assert len(dataset) == 3
        assert dataset.name == "test"
        assert all(isinstance(item, GDTItem) for item in dataset)
        validation_results = mock_gdt_generator.validate_data(dataset)
        assert len(validation_results) > 0

    def test_generate_dataset_with_validation(
        self, mock_gdt_generator: MockGDTGenerator
    ) -> None:
        """Test dataset generation with validation."""
        dataset = mock_gdt_generator.generate_dataset(
            "validated_test", DataType.TEXT, 2
        )

        # Generate dataset
        assert len(dataset) == 2

        # Validate all items
        validation_results = mock_gdt_generator.validate_data(dataset)
        assert len(validation_results) == len(dataset)
        assert all(
            result["status"] in [ValidationStatus.VALID, ValidationStatus.WARNING]
            for result in validation_results
        )

    def test_data_transformation(self, mock_gdt_generator: MockGDTGenerator) -> None:
        """Test data transformation functionality."""
        # Create test dataset
        dataset = mock_gdt_generator.generate_dataset(
            "transform_test", DataType.TEXT, 2
        )

        # Transform data
        transformed = mock_gdt_generator.transform_data(
            dataset, "add_metadata", test_flag=True
        )

        # Verify transformation
        assert isinstance(transformed, GDTDataset)
        assert len(transformed) == len(dataset)

    def test_validation_edge_cases(self, mock_gdt_generator: MockGDTGenerator) -> None:
        """Test validation with edge cases."""
        # Test empty dataset
        empty_dataset = GDTDataset("empty", "Empty dataset")
        validation_results = mock_gdt_generator.validate_data(empty_dataset)
        assert len(validation_results) == 0

        # Test invalid item
        invalid_item = GDTItem(
            id="",  # Empty ID should trigger validation error
            data_type=DataType.TEXT,
            content={},  # Empty content
        )
        validation_results = mock_gdt_generator.validate_data(invalid_item)
        assert len(validation_results) == 1

    def test_bulk_generation_performance(
        self, mock_gdt_generator: MockGDTGenerator
    ) -> None:
        """Test bulk data generation performance."""
        # Generate larger dataset
        large_dataset = mock_gdt_generator.generate_dataset(
            "bulk_test", DataType.TEXT, 100
        )

        assert len(large_dataset) == 100

        # Verify uniqueness
        item_ids = [item.id for item in large_dataset]
        unique_ids = set(item_ids)
        assert len(unique_ids) > 90  # Allow some duplicates in mock data


class TestGDTDataStructures:
    """Test suite for GDT data structures."""

    def test_gdt_item_creation(self) -> None:
        """Test GDTItem creation and properties."""
        item = GDTItem(
            id="test_item",
            data_type=DataType.TEXT,
            content={"text": "Test content"},
            tags=["test"],
        )

        assert item.id == "test_item"
        assert item.data_type == DataType.TEXT
        assert item.content["text"] == "Test content"
        assert "test" in item.tags
        assert item.validation_status == ValidationStatus.VALID

    def test_gdt_item_serialization(self) -> None:
        """Test GDTItem serialization and deserialization."""
        original_item = GDTItem(
            id="serialize_test",
            data_type=DataType.CONVERSATION,
            content={"messages": [{"role": "user", "content": "Hello"}]},
            tags=["serialization", "test"],
        )

        # Serialize to dict
        item_dict = original_item.to_dict()
        assert isinstance(item_dict, dict)
        assert item_dict["id"] == "serialize_test"

        # Deserialize from dict
        restored_item = GDTItem.from_dict(item_dict)
        assert restored_item.id == original_item.id
        assert restored_item.data_type == original_item.data_type
        assert restored_item.content == original_item.content

    def test_gdt_dataset_operations(self, sample_gdt_items: List[GDTItem]) -> None:
        """Test GDTDataset operations."""
        dataset = GDTDataset("test_ops", "Test operations")

        # Test append
        dataset.append(sample_gdt_items[0])
        assert len(dataset) == 1

        # Test extend
        dataset.extend(sample_gdt_items[1:])
        assert len(dataset) == len(sample_gdt_items)

        # Test iteration
        items_from_iteration = list(dataset)
        assert len(items_from_iteration) == len(sample_gdt_items)

        # Test filtering
        text_items = dataset.filter_by_type(DataType.TEXT)
        assert all(item.data_type == DataType.TEXT for item in text_items)


class TestRealGDTGenerator:
    """Test suite for real GDT generator (not mocked)."""

    def test_real_generator_creation(self) -> None:
        """Test creating real GDT generator."""
        generator = GDTGenerator()
        assert isinstance(generator, GDTGenerator)
        assert isinstance(generator.validator, GDTValidator)
        assert isinstance(generator.transformer, GDTTransformer)

    def test_text_data_generation(self) -> None:
        """Test text data generation."""
        generator = TextDataGenerator()
        item = generator.generate_item(topic="machine learning")

        assert isinstance(item, GDTItem)
        assert item.data_type == DataType.TEXT
        assert "machine learning" in item.content["topic"]

    def test_conversation_data_generation(self) -> None:
        """Test conversation data generation."""
        generator = ConversationDataGenerator()
        item = generator.generate_item(turns=4)

        assert isinstance(item, GDTItem)
        assert item.data_type == DataType.CONVERSATION
        assert len(item.content["messages"]) == 4

    def test_dataset_persistence(self, sample_dataset: GDTDataset) -> None:
        """Test dataset saving and loading."""
        generator = GDTGenerator()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            # Save dataset
            generator.save_dataset(sample_dataset, tmp_path)
            assert tmp_path.exists()

            # Load dataset
            loaded_dataset = generator.load_dataset(tmp_path)
            assert loaded_dataset.name == sample_dataset.name
            assert len(loaded_dataset) == len(sample_dataset)

        finally:
            # Cleanup
            if tmp_path.exists():
                tmp_path.unlink()


class TestGDTValidation:
    """Test suite for GDT validation functionality."""

    def test_validator_creation(self) -> None:
        """Test validator creation with custom rules."""
        custom_rules = {
            "min_content_length": 10,
            "max_content_length": 10000,
            "required_fields": ["id", "data_type", "content"],
        }
        validator = GDTValidator(custom_rules)
        assert validator.rules["min_content_length"] == 10

    def test_item_validation(self) -> None:
        """Test individual item validation."""
        validator = GDTValidator()

        valid_item = GDTItem(
            id="valid_test",
            data_type=DataType.TEXT,
            content={"text": "This is valid content with sufficient length."},
        )

        result = validator.validate_item(valid_item)
        assert result["status"] == ValidationStatus.VALID
        assert len(result["errors"]) == 0


class TestGDTTransformation:
    """Test suite for GDT transformation functionality."""

    def test_transformer_creation(self) -> None:
        """Test transformer creation."""
        transformer = GDTTransformer()
        assert "normalize_text" in transformer.transformations
        assert "add_metadata" in transformer.transformations

    def test_text_normalization(self) -> None:
        """Test text normalization transformation."""
        transformer = GDTTransformer()

        item = GDTItem(
            id="norm_test",
            data_type=DataType.TEXT,
            content={"text": "  UPPERCASE TEXT  "},
        )

        normalized = transformer.transform_item(item, "normalize_text")
        assert normalized.content["text"] == "uppercase text"

    def test_metadata_addition(self) -> None:
        """Test metadata addition transformation."""
        transformer = GDTTransformer()

        item = GDTItem(
            id="meta_test", data_type=DataType.TEXT, content={"text": "Test content"}
        )

        transformed = transformer.transform_item(
            item, "add_metadata", test_flag=True, version="1.0"
        )
        assert transformed.metadata["test_flag"] is True
        assert transformed.metadata["version"] == "1.0"


# Integration tests
class TestGDTIntegration:
    """Integration tests for the complete GDT system."""

    def test_end_to_end_workflow(self) -> None:
        """Test complete end-to-end GDT workflow."""
        # Create generator
        generator = GDTGenerator()

        # Generate dataset
        dataset = generator.generate_dataset(
            name="integration_test",
            data_type=DataType.TEXT,
            count=5,
            topic="integration testing",
        )

        # Validate dataset
        validation_results = generator.validate_data(dataset)
        assert len(validation_results) == 5

        # Transform dataset
        transformed_dataset = generator.transform_data(
            dataset, "add_metadata", test_run=True
        )
        assert isinstance(transformed_dataset, GDTDataset)

        # Verify transformation applied
        for item in transformed_dataset.items:
            assert item.metadata.get("test_run") is True

    def test_custom_generator_registration(self) -> None:
        """Test registering custom data generators."""

        class CustomGenerator(TextDataGenerator):
            def generate_item(self, **kwargs: Any) -> GDTItem:
                return GDTItem(
                    id="custom_item",
                    data_type=DataType.TEXT,
                    content={"custom": "content"},
                    tags=["custom"],
                )

        generator = GDTGenerator()
        custom_gen = CustomGenerator()
        generator.register_generator(DataType.TEXT, custom_gen)

        dataset = generator.generate_dataset("custom_test", DataType.TEXT, 1)
        assert dataset.items[0].content["custom"] == "content"


class TestGDTPerformance:
    """Performance tests for GDT system."""

    def test_large_dataset_generation(self) -> None:
        """Test generation of large datasets."""
        generator = GDTGenerator()

        # Generate large text dataset
        large_dataset = generator.generate_dataset(
            name="large_test", data_type=DataType.TEXT, count=1000
        )

        assert len(large_dataset) == 1000
        assert all(isinstance(item, GDTItem) for item in large_dataset)

    def test_batch_validation_performance(self) -> None:
        """Test validation performance on large datasets."""
        generator = GDTGenerator()

        # Generate dataset
        dataset = generator.generate_dataset(
            name="validation_perf_test", data_type=DataType.TEXT, count=500
        )

        # Validate dataset
        validation_results = generator.validate_data(dataset)

        assert len(validation_results) == 500
        assert all("status" in result for result in validation_results)

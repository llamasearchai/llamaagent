#!/usr/bin/env python3
"""
Comprehensive System Integration Test Suite

This test suite verifies that all components of the LlamaAgent system
work perfectly together.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llamaagent.data.gdt import DataType, GDTDataset, GDTGenerator, GDTItem
# Import our modules
from llamaagent.storage.database import DatabaseConfig, DatabaseManager


class SystemIntegrationTest:
    """Comprehensive system integration test suite."""

    def __init__(self):
        self.test_results = []
        self.db_manager = None
        self.gdt_generator = None

    def log_test(self, test_name: str, passed: bool, message: str = ""):
        """Log test result."""
        status = "PASS" if passed else "FAIL"
        result = {
            "test": test_name,
            "status": status,
            "message": message,
            "timestamp": time.time(),
        }
        self.test_results.append(result)
        print(f"[{status}] {test_name}: {message}")

    def test_imports(self):
        """Test that all imports work correctly."""
        try:
            # Test database imports
            # Test GDT imports
            from llamaagent.data.gdt import (DataType, GDTDataset,
                                             GDTGenerator, GDTItem,
                                             GDTTransformer, GDTValidator,
                                             ValidationStatus)
            from llamaagent.storage.database import (DatabaseConfig,
                                                     DatabaseManager,
                                                     QueryResult)

            self.log_test("Import Test", True, "All modules imported successfully")
            return True
        except Exception as e:
            self.log_test("Import Test", False, f"Import failed: {e}")
            return False

    def test_database_config(self):
        """Test database configuration."""
        try:
            # Test default config
            config = DatabaseConfig()
            assert config.host == "localhost"
            assert config.port == 5432
            assert config.database == "llamaagent"

            # Test custom config
            custom_config = DatabaseConfig(
                host="testhost", port=5433, database="testdb"
            )
            assert custom_config.host == "testhost"
            assert custom_config.port == 5433

            self.log_test(
                "Database Config Test", True, "Database configuration working correctly"
            )
            return True
        except Exception as e:
            self.log_test("Database Config Test", False, f"Config test failed: {e}")
            return False

    def test_database_manager_initialization(self):
        """Test database manager initialization."""
        try:
            # Test initialization without actual database connection
            self.db_manager = DatabaseManager()
            assert self.db_manager.config is not None
            assert self.db_manager.pool is None  # Should be None before initialization

            # Test connection string building
            conn_str = self.db_manager._build_connection_string()
            assert "postgresql://" in conn_str
            assert "llamaagent" in conn_str

            self.log_test(
                "Database Manager Init Test",
                True,
                "Database manager initialized correctly",
            )
            return True
        except Exception as e:
            self.log_test(
                "Database Manager Init Test", False, f"Initialization failed: {e}"
            )
            return False

    def test_gdt_generator_initialization(self):
        """Test GDT generator initialization."""
        try:
            self.gdt_generator = GDTGenerator()
            assert self.gdt_generator.validator is not None
            assert self.gdt_generator.transformer is not None
            assert len(self.gdt_generator.generators) >= 2  # TEXT and CONVERSATION

            self.log_test(
                "GDT Generator Init Test", True, "GDT generator initialized correctly"
            )
            return True
        except Exception as e:
            self.log_test(
                "GDT Generator Init Test", False, f"GDT initialization failed: {e}"
            )
            return False

    def test_gdt_item_creation(self):
        """Test GDT item creation and manipulation."""
        try:
            # Test basic item creation
            item = GDTItem(
                data_type=DataType.TEXT,
                content={"text": "Test content"},
                tags=["test", "sample"],
            )

            assert item.data_type == DataType.TEXT
            assert item.content["text"] == "Test content"
            assert "test" in item.tags
            assert item.id is not None

            # Test serialization
            item_dict = item.to_dict()
            assert "id" in item_dict
            assert item_dict["data_type"] == "text"
            assert item_dict["content"]["text"] == "Test content"

            # Test deserialization
            restored_item = GDTItem.from_dict(item_dict)
            assert restored_item.data_type == DataType.TEXT
            assert restored_item.content["text"] == "Test content"

            self.log_test(
                "GDT Item Creation Test",
                True,
                "GDT items created and serialized correctly",
            )
            return True
        except Exception as e:
            self.log_test("GDT Item Creation Test", False, f"Item creation failed: {e}")
            return False

    def test_gdt_dataset_operations(self):
        """Test GDT dataset operations."""
        try:
            # Create dataset
            dataset = GDTDataset("test_dataset", "Test dataset for integration testing")

            # Add items
            for i in range(5):
                item = GDTItem(
                    data_type=DataType.TEXT,
                    content={"text": f"Test content {i}"},
                    tags=["test", f"item_{i}"],
                )
                dataset.append(item)

            assert len(dataset) == 5

            # Test filtering
            text_items = dataset.filter_by_type(DataType.TEXT)
            assert len(text_items) == 5

            tag_items = dataset.filter_by_tag("test")
            assert len(tag_items) == 5

            # Test serialization
            dataset_dict = dataset.to_dict()
            assert dataset_dict["name"] == "test_dataset"
            assert len(dataset_dict["items"]) == 5

            # Test deserialization
            restored_dataset = GDTDataset.from_dict(dataset_dict)
            assert restored_dataset.name == "test_dataset"
            assert len(restored_dataset) == 5

            self.log_test(
                "GDT Dataset Operations Test",
                True,
                "Dataset operations working correctly",
            )
            return True
        except Exception as e:
            self.log_test(
                "GDT Dataset Operations Test", False, f"Dataset operations failed: {e}"
            )
            return False

    def test_gdt_validation(self):
        """Test GDT validation system."""
        try:
            # Create validator
            validator = self.gdt_generator.validator

            # Test valid item
            valid_item = GDTItem(
                data_type=DataType.TEXT,
                content={
                    "text": "This is a valid test content with sufficient length."
                },
                tags=["valid", "test"],
            )

            validation_result = validator.validate_item(valid_item)
            assert validation_result["status"].value == "valid"
            assert len(validation_result["errors"]) == 0

            # Test invalid item
            invalid_item = GDTItem(
                data_type=DataType.TEXT,
                content={"text": "Short"},  # Too short
                tags=["invalid"],
            )

            validation_result = validator.validate_item(invalid_item)
            assert validation_result["status"].value in ["warning", "invalid"]

            self.log_test(
                "GDT Validation Test", True, "Validation system working correctly"
            )
            return True
        except Exception as e:
            self.log_test("GDT Validation Test", False, f"Validation failed: {e}")
            return False

    def test_gdt_transformation(self):
        """Test GDT transformation system."""
        try:
            transformer = self.gdt_generator.transformer

            # Test text normalization
            item = GDTItem(
                data_type=DataType.TEXT,
                content={"text": "  TEST CONTENT WITH SPACES  "},
                tags=["transform", "test"],
            )

            transformed_item = transformer.transform_item(item, "normalize_text")
            assert transformed_item.content["text"] == "test content with spaces"

            # Test metadata addition
            transformed_item = transformer.transform_item(
                item, "add_metadata", test_key="test_value"
            )
            assert transformed_item.metadata["test_key"] == "test_value"

            self.log_test(
                "GDT Transformation Test",
                True,
                "Transformation system working correctly",
            )
            return True
        except Exception as e:
            self.log_test(
                "GDT Transformation Test", False, f"Transformation failed: {e}"
            )
            return False

    def test_gdt_generation(self):
        """Test GDT data generation."""
        try:
            # Generate text dataset
            text_dataset = self.gdt_generator.generate_dataset(
                "test_text_dataset", DataType.TEXT, 3, topic="testing", length=100
            )

            assert len(text_dataset) == 3
            assert text_dataset.name == "test_text_dataset"

            for item in text_dataset:
                assert item.data_type == DataType.TEXT
                assert "testing" in item.content["text"]
                assert "generated" in item.tags

            # Generate conversation dataset
            conv_dataset = self.gdt_generator.generate_dataset(
                "test_conversation_dataset",
                DataType.CONVERSATION,
                2,
                turns=4,
                context="testing",
            )

            assert len(conv_dataset) == 2
            for item in conv_dataset:
                assert item.data_type == DataType.CONVERSATION
                assert len(item.content["messages"]) == 4
                assert item.content["context"] == "testing"

            self.log_test(
                "GDT Generation Test", True, "Data generation working correctly"
            )
            return True
        except Exception as e:
            self.log_test("GDT Generation Test", False, f"Generation failed: {e}")
            return False

    def test_json_serialization(self):
        """Test JSON serialization of all components."""
        try:
            # Create and serialize dataset
            dataset = self.gdt_generator.generate_dataset(
                "json_test_dataset", DataType.TEXT, 2, topic="json_test"
            )

            # Serialize to JSON
            json_str = json.dumps(dataset.to_dict(), indent=2, default=str)
            assert len(json_str) > 0

            # Deserialize from JSON
            data_dict = json.loads(json_str)
            restored_dataset = GDTDataset.from_dict(data_dict)

            assert restored_dataset.name == "json_test_dataset"
            assert len(restored_dataset) == 2

            self.log_test(
                "JSON Serialization Test", True, "JSON serialization working correctly"
            )
            return True
        except Exception as e:
            self.log_test(
                "JSON Serialization Test", False, f"JSON serialization failed: {e}"
            )
            return False

    def test_error_handling(self):
        """Test error handling in various scenarios."""
        try:
            # Test invalid data type
            try:
                self.gdt_generator.generate_dataset("test", "invalid_type", 1)
                assert False, "Should have raised an error"
            except (ValueError, TypeError):
                pass  # Expected

            # Test invalid transformation
            item = GDTItem(data_type=DataType.TEXT, content={"text": "test"})
            try:
                self.gdt_generator.transformer.transform_item(
                    item, "invalid_transformation"
                )
                assert False, "Should have raised an error"
            except ValueError:
                pass  # Expected

            # Test validation with invalid data
            try:
                self.gdt_generator.validator.validate_data("invalid_data")
                assert False, "Should have raised an error"
            except TypeError:
                pass  # Expected

            self.log_test(
                "Error Handling Test", True, "Error handling working correctly"
            )
            return True
        except Exception as e:
            self.log_test("Error Handling Test", False, f"Error handling failed: {e}")
            return False

    def test_performance(self):
        """Test performance with larger datasets."""
        try:
            start_time = time.time()

            # Generate larger dataset
            large_dataset = self.gdt_generator.generate_dataset(
                "performance_test", DataType.TEXT, 100, topic="performance"
            )

            generation_time = time.time() - start_time

            # Validate all items
            start_time = time.time()
            validation_results = self.gdt_generator.validate_data(large_dataset)
            validation_time = time.time() - start_time

            assert len(large_dataset) == 100
            assert len(validation_results) == 100
            assert generation_time < 10.0  # Should complete within 10 seconds
            assert validation_time < 5.0  # Should complete within 5 seconds

            self.log_test(
                "Performance Test",
                True,
                f"Generated 100 items in {generation_time:.2f}s, validated in {validation_time:.2f}s",
            )
            return True
        except Exception as e:
            self.log_test("Performance Test", False, f"Performance test failed: {e}")
            return False

    def run_all_tests(self):
        """Run all tests in the suite."""
        print("=" * 60)
        print("LLAMAAGENT SYSTEM INTEGRATION TEST SUITE")
        print("=" * 60)

        tests = [
            self.test_imports,
            self.test_database_config,
            self.test_database_manager_initialization,
            self.test_gdt_generator_initialization,
            self.test_gdt_item_creation,
            self.test_gdt_dataset_operations,
            self.test_gdt_validation,
            self.test_gdt_transformation,
            self.test_gdt_generation,
            self.test_json_serialization,
            self.test_error_handling,
            self.test_performance,
        ]

        passed = 0
        failed = 0

        for test in tests:
            try:
                if test():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                print(f"[FAIL] {test.__name__}: Unexpected error: {e}")

        print("=" * 60)
        print(f"TEST RESULTS: {passed} PASSED, {failed} FAILED")
        print("=" * 60)

        if failed == 0:
            print("SUCCESS ALL TESTS PASSED! System is working perfectly!")
        else:
            print("FAIL Some tests failed. Please check the output above.")

        return failed == 0


def main():
    """Run the integration test suite."""
    test_suite = SystemIntegrationTest()
    success = test_suite.run_all_tests()

    # Save test results
    with open("test_results.json", "w") as f:
        json.dump(
            {
                "timestamp": time.time(),
                "total_tests": len(test_suite.test_results),
                "passed": sum(
                    1 for r in test_suite.test_results if r["status"] == "PASS"
                ),
                "failed": sum(
                    1 for r in test_suite.test_results if r["status"] == "FAIL"
                ),
                "results": test_suite.test_results,
            },
            f,
            indent=2,
        )

    print(f"\nDetailed test results saved to test_results.json")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

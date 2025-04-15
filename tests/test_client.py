"""
Tests for the Client class.
"""

import pytest

from llamaagent import Client, Config


def test_client_initialization():
    """Test that a Client can be initialized with default values."""
    client = Client()
    assert client.api_key is None
    assert "api.llamasearch.ai" in client.base_url
    assert isinstance(client.config, Config)


def test_client_with_custom_config():
    """Test that a Client can be initialized with custom configuration."""
    config = Config(timeout=30, retries=5, verbose=True)
    client = Client(api_key="REDACTED", config=config)

    assert client.api_key == "test_key"
    assert client.config.timeout == 30
    assert client.config.retries == 5
    assert client.config.verbose is True


def test_process_data():
    """Test the process_data method."""
    client = Client()
    result = client.process_data("test data")

    assert isinstance(result, dict)
    assert result["status"] == "success"
    assert "Processed: test data" in result["data"]
    assert "metadata" in result


def test_batch_process():
    """Test the batch_process method."""
    client = Client()
    data_items = ["item1", "item2", "item3"]

    results = client.batch_process(data_items)

    assert isinstance(results, list)
    assert len(results) == 3
    for i, result in enumerate(results):
        assert f"Processed: {data_items[i]}" in result["data"]

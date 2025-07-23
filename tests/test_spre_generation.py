"""Tests for SPRE dataset generation paths.
Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio

import llamaagent.types  # noqa: F401 – imported for coverage
from llamaagent.data_generation.spre import DataType, SPREGenerator

ASYNC_TIMEOUT = 30


def test_sync_generation_compiles():
    """Ensure synchronous generation returns the expected number of items."""
    generator = SPREGenerator()
    dataset = generator.generate_dataset("Smoke", 3, data_type=DataType.TEXT)
    assert len(dataset.items) == 3
    assert all(item.validation_status for item in dataset.items)


def test_async_generation_event_loop():
    """Ensure async generation works inside an already running event loop."""

    async def _run():
        generator = SPREGenerator()
        dataset = await generator.generate_dataset_async(
            "Async",
            2,
            data_type=DataType.CONVERSATION,
        )
        assert len(dataset.get_valid_items()) == len(dataset.items)

    asyncio.run(asyncio.wait_for(_run(), ASYNC_TIMEOUT))

"""ML module for LlamaAgent."""

from typing import Any, List


class MLModel:
    """Basic ML model interface."""

    def predict(self, data: Any) -> Any:
        """Make prediction."""
        return None


__all__ = ['MLModel']

"""MLX provider for Apple Silicon optimization."""

from typing import List

from .base import LLMMessage, LLMProvider, LLMResponse


class MlxProvider(LLMProvider):
    """MLX provider for Apple Silicon - simplified version."""

    def __init__(self, model: str = "llama3.2:3b"):
        """Initialize MLX provider."""
        self.model = model
        # For now, we'll use Ollama as backend until MLX is properly set up
        self.fallback_to_ollama = True

    async def complete(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Complete using MLX (fallback to Ollama for now)."""
        if self.fallback_to_ollama:
            # Import here to avoid circular imports
            from .ollama_provider import OllamaProvider
            
            ollama = OllamaProvider(model=self.model)
            return await ollama.complete(messages, **kwargs)

        # TODO: Implement actual MLX when dependencies are resolved
        return LLMResponse(
            content="MLX provider not yet implemented - using Ollama fallback",
            model=f"mlx-{self.model}",
            tokens_used=0,
            **kwargs
        )

    async def shutdown(self) -> None:
        """Shutdown the provider."""
        pass
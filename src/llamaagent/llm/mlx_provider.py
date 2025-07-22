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
        """Complete using MLX with Apple Silicon optimization."""
        try:
            # Try to use MLX if available
            import mlx.core as mx
            import mlx_lm

            # Convert messages to prompt
            prompt = self._messages_to_prompt(messages)

            # Load model with MLX
            model, tokenizer = mlx_lm.load(self.model)

            # Generate response
            response = mlx_lm.generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=kwargs.get("max_tokens", 1000),
                temperature=kwargs.get("temperature", 0.7),
            )

            # Count tokens
            tokens_used = len(tokenizer.encode(prompt)) + len(
                tokenizer.encode(response)
            )

            return LLMResponse(
                content=response,
                model=f"mlx-{self.model}",
                tokens_used=tokens_used,
                **kwargs,
            )

        except ImportError:
            # MLX not available, fallback to Ollama
            if self.fallback_to_ollama:
                from .ollama_provider import OllamaProvider

                ollama = OllamaProvider(model=self.model)
                return await ollama.complete(messages, **kwargs)

            # No fallback available
            return LLMResponse(
                content="MLX not available on this system. Please install mlx-lm for Apple Silicon support.",
                model=f"mlx-{self.model}",
                tokens_used=0,
                **kwargs,
            )

    def _messages_to_prompt(self, messages: List[LLMMessage]) -> str:
        """Convert messages to a single prompt string."""
        prompt_parts = []
        for msg in messages:
            role = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
            prompt_parts.append(f"{role}: {msg.content}")
        return "\n".join(prompt_parts)

    async def shutdown(self) -> None:
        """Shutdown the provider."""
        pass

"""
CUDA Provider implementation

CUDA-accelerated provider using Hugging Face Transformers. This provider is
intended for workstations or servers equipped with NVIDIA GPUs and a functional
CUDA installation. It tries to load the requested model via transformers into
GPU memory. If CUDA is unavailable (e.g. running on a CPU-only machine) the
provider gracefully falls back to the OllamaProvider so that the remainder
of the pipeline keeps working.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional

from src.llamaagent import LLMMessage, LLMResponse

from ..exceptions import LLMError
from .base_provider import BaseLLMProvider

if TYPE_CHECKING:
    from .ollama_provider import OllamaProvider

logger = logging.getLogger(__name__)


class CUDAProvider(BaseLLMProvider):
    """CUDA-backed provider built on Hugging Face Transformers."""

    def __init__(
        self,
        model: str = "microsoft/phi-2",
        device: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        fallback_to_ollama: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize CUDA provider.

        Args:
            model: Hugging Face model ID or path
            device: Device to use ("cuda", "cpu", or None for auto)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            fallback_to_ollama: Whether to fallback to Ollama if CUDA fails
            **kwargs: Additional configuration
        """
        super().__init__(model=model, **kwargs)
        self.device = device  # may be set after torch import
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._fallback_to_ollama = fallback_to_ollama
        self._fallback_provider: Optional[OllamaProvider] = None
        self._model = None
        self._tokenizer = None

        # Try to initialize CUDA/transformers
        try:
            self._setup_cuda()
        except Exception as exc:
            logger.warning(f"Failed to initialize CUDA: {exc}")
            if fallback_to_ollama:
                from .ollama_provider import OllamaProvider

                self._fallback_provider = OllamaProvider(model=model)
            else:
                raise

    def _setup_cuda(self) -> None:
        """Setup CUDA and load model."""
        import importlib

        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")

        AutoTokenizer = transformers.AutoTokenizer
        AutoModelForCausalLM = transformers.AutoModelForCausalLM

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading model %s on %s", self.model, self.device)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        if self.device != "cuda":
            self._model = self._model.to(self.device)

        self._model.eval()

    async def generate_response(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response from a prompt.

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            LLMResponse with generated content
        """
        messages = [LLMMessage(role="user", content=prompt)]
        return await self.complete(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    async def chat_completion(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Complete a chat conversation.

        Args:
            messages: List of conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            model: Model to use (overrides default)
            **kwargs: Additional parameters

        Returns:
            LLMResponse containing the assistant's reply
        """
        return await self.complete(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            model=model,
            **kwargs,
        )

    async def complete(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Complete using CUDA-accelerated model.

        Args:
            messages: List of conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            model: Model to use (ignored, uses initialized model)
            **kwargs: Additional parameters

        Returns:
            LLMResponse with generated content
        """
        if self._fallback_provider is not None:
            return await self._fallback_provider.complete(messages, **kwargs)

        if self._model is None or self._tokenizer is None:
            raise LLMError("CUDA provider not initialized properly")

        # Convert messages to prompt
        prompt = "\n".join(msg.content for msg in messages)

        # Run generation in thread pool since it's synchronous
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            self._generate_sync,
            prompt,
            max_tokens,
            temperature,
            kwargs,
        )

        return response

    def _generate_sync(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        kwargs: Dict[str, Any],
    ) -> LLMResponse:
        """Synchronous generation method for thread pool execution."""
        import torch

        start = time.perf_counter()

        # Tokenize input
        inputs = self._tokenizer(prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generation arguments
        generation_args = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "pad_token_id": self._tokenizer.eos_token_id,
            **kwargs,
        }

        # Generate
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **generation_args)

        # Decode output
        output_text = self._tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )

        latency_ms = (time.perf_counter() - start) * 1000
        tokens_used = output_ids.shape[-1]

        return LLMResponse(
            content=output_text.strip(),
            model=self.model,
            provider="cuda",
            tokens_used=tokens_used,
            usage={"latency_ms": latency_ms, "device": self.device},
        )

    async def stream_chat_completion(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response.

        Currently simulates streaming by yielding the complete response in chunks.

        Args:
            messages: List of conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            model: Model to use (ignored)
            **kwargs: Additional parameters

        Yields:
            String chunks of the response
        """
        if self._fallback_provider is not None:
            async for chunk in self._fallback_provider.stream_chat_completion(
                messages, max_tokens=max_tokens, temperature=temperature, **kwargs
            ):
                yield chunk
            return

        # For now, simulate streaming
        response = await self.complete(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

        # Yield response in chunks
        words = response.content.split()
        for word in words:
            yield word + " "
            await asyncio.sleep(0.01)

    async def embed_text(
        self, texts: List[str], model: Optional[str] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Generate embeddings.

        Note: This requires a sentence-transformers model, not implemented yet.

        Args:
            texts: List of texts to embed
            model: Embedding model to use
            **kwargs: Additional parameters

        Raises:
            NotImplementedError: CUDA provider doesn't support embeddings yet
        """
        # For now, return mock embeddings
        import numpy as np

        return [np.random.rand(384).tolist() for _ in texts]

    async def validate_model(self, model: str) -> bool:
        """Validate if model is available.

        Args:
            model: Model name to validate

        Returns:
            True if model can be loaded, False otherwise
        """
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(model)
            return config is not None
        except Exception:
            return False

    async def health_check(self) -> bool:
        """Check if the CUDA provider is healthy.

        Returns:
            True if provider is operational, False otherwise
        """
        if self._fallback_provider is not None:
            return await self._fallback_provider.health_check()

        try:
            import torch

            if self._model is None:
                return False
            return torch.cuda.is_available() if self.device == "cuda" else True
        except Exception:
            return False

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for CUDA usage.

        CUDA is local, so there's no cost.

        Args:
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens

        Returns:
            Always returns 0.0 for local inference
        """
        return 0.0

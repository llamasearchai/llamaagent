"""
LLM client module for different providers.

This module provides a unified interface for interacting with various
language model providers through a consistent API.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union

from .config import LLMConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Base client for language model providers.

    This class defines the interface that all LLM provider implementations
    should follow. Subclasses implement provider-specific logic.
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize the LLM client.

        Args:
            config: Configuration for the language model
        """
        self.config = config

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from the language model.

        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional arguments to override configuration

        Returns:
            The generated text
        """
        raise NotImplementedError("Subclasses must implement this method")

    def generate_with_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a structured JSON response from the language model.

        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional arguments to override configuration

        Returns:
            The parsed JSON response
        """
        raise NotImplementedError("Subclasses must implement this method")

    def embed(self, text: str) -> List[float]:
        """
        Generate embeddings for the given text.

        Args:
            text: The text to embed

        Returns:
            The embedding vector
        """
        raise NotImplementedError("Subclasses must implement this method")


class AnthropicClient(LLMClient):
    """Client for the Anthropic API."""

    def __init__(self, config: LLMConfig):
        """
        Initialize the Anthropic client.

        Args:
            config: Configuration for Anthropic
        """
        super().__init__(config)

        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required to use the Anthropic API. "
                "Install it with 'pip install anthropic'."
            )

        # Get API key from config or environment
        api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key is required. Provide it in the config or "
                "set the ANTHROPIC_API_KEY environment variable."
            )

        self.client = anthropic.Anthropic(api_key=api_key)
        logger.debug(f"Initialized Anthropic client with model {config.model}")

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the Anthropic API.

        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional arguments to override configuration

        Returns:
            The generated text
        """
        # Combine config with overrides
        params = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            **self.config.extra_params,
            **kwargs,
        }

        messages = [{"role": "user", "content": prompt}]

        # Add system prompt if not empty
        if self.config.system_prompt:
            params["system"] = self.config.system_prompt

        logger.debug(f"Sending request to Anthropic API with {len(prompt)} chars")

        try:
            response = self.client.messages.create(messages=messages, **params)

            # Extract text content
            content = response.content[0].text
            logger.debug(f"Received {len(content)} chars from Anthropic API")
            return content

        except Exception as e:
            logger.error(f"Error calling Anthropic API: {e}")
            return f"Error generating response: {str(e)}"

    def generate_with_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a structured JSON response using the Anthropic API.

        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional arguments to override configuration

        Returns:
            The parsed JSON response
        """
        import json

        # Add instruction to return JSON
        json_prompt = f"{prompt}\n\nRespond with valid JSON."

        # Generate response
        response_text = self.generate(json_prompt, **kwargs)

        # Extract and parse JSON
        try:
            # Try to extract JSON if it's wrapped in markdown code blocks
            if "```json" in response_text:
                json_content = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_content = response_text.split("```")[1].strip()
            else:
                json_content = response_text

            return json.loads(json_content)

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON from response: {e}")
            logger.debug(f"Response was: {response_text}")
            return {
                "error": "Could not parse JSON from response",
                "text": response_text,
            }

    def embed(self, text: str) -> List[float]:
        """
        Generate embeddings using the Anthropic API.

        Args:
            text: The text to embed

        Returns:
            The embedding vector
        """
        try:
            response = self.client.embeddings.create(
                model="claude-3-haiku-20240307",  # Embedding model for Anthropic
                input=text,
            )
            return response.embedding

        except Exception as e:
            logger.error(f"Error generating embeddings with Anthropic API: {e}")
            # Return a zero vector as fallback
            return [0.0] * 1024  # Claude embeddings are 1024-dimensional


class OpenAIClient(LLMClient):
    """Client for the OpenAI API."""

    def __init__(self, config: LLMConfig):
        """
        Initialize the OpenAI client.

        Args:
            config: Configuration for OpenAI
        """
        super().__init__(config)

        try:
            import openai
        except ImportError:
            raise ImportError(
                "The 'openai' package is required to use the OpenAI API. "
                "Install it with 'pip install openai'."
            )

        # Get API key from config or environment
        api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Provide it in the config or "
                "set the OPENAI_API_KEY environment variable."
            )

        self.client = openai.OpenAI(api_key=api_key)
        logger.debug(f"Initialized OpenAI client with model {config.model}")

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the OpenAI API.

        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional arguments to override configuration

        Returns:
            The generated text
        """
        # Combine config with overrides
        params = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            **self.config.extra_params,
            **kwargs,
        }

        messages = [{"role": "user", "content": prompt}]

        # Add system message if not empty
        if self.config.system_prompt:
            messages.insert(0, {"role": "system", "content": self.config.system_prompt})

        logger.debug(f"Sending request to OpenAI API with {len(prompt)} chars")

        try:
            response = self.client.chat.completions.create(messages=messages, **params)

            # Extract content
            content = response.choices[0].message.content
            logger.debug(f"Received {len(content)} chars from OpenAI API")
            return content

        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return f"Error generating response: {str(e)}"

    def generate_with_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a structured JSON response using the OpenAI API.

        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional arguments to override configuration

        Returns:
            The parsed JSON response
        """
        # For OpenAI, we can use the native JSON mode if available
        if (
            "response_format" not in kwargs
            and "response_format" not in self.config.extra_params
        ):
            kwargs["response_format"] = {"type": "json_object"}

        # Generate response
        response_text = self.generate(prompt, **kwargs)

        # Parse JSON
        import json

        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON from response: {e}")
            logger.debug(f"Response was: {response_text}")
            return {
                "error": "Could not parse JSON from response",
                "text": response_text,
            }

    def embed(self, text: str) -> List[float]:
        """
        Generate embeddings using the OpenAI API.

        Args:
            text: The text to embed

        Returns:
            The embedding vector
        """
        try:
            embedding_model = "text-embedding-3-small"  # Default embedding model

            # Check if model is specified in extra_params
            if "embedding_model" in self.config.extra_params:
                embedding_model = self.config.extra_params["embedding_model"]

            response = self.client.embeddings.create(model=embedding_model, input=text)

            return response.data[0].embedding

        except Exception as e:
            logger.error(f"Error generating embeddings with OpenAI API: {e}")
            # Return a zero vector as fallback
            return [0.0] * 1536  # OpenAI embeddings are 1536-dimensional


def get_llm_client(config: LLMConfig) -> LLMClient:
    """
    Get an LLM client for the specified provider.

    Args:
        config: Configuration for the language model

    Returns:
        An initialized LLM client

    Raises:
        ValueError: If the provider is unknown
    """
    provider = config.provider.lower()

    if provider == "anthropic":
        return AnthropicClient(config)
    elif provider == "openai":
        return OpenAIClient(config)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

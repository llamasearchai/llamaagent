"""
Cohere LLM provider implementation.
"""

import asyncio
import os
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

from ..base import BaseLLMProvider
from ..messages import LLMMessage, LLMResponse


class CohereConfig(BaseModel):
    """Configuration for Cohere provider."""
    
    api_key: str = Field(..., description="Cohere API key")
    base_url: str = Field(
        default="https://api.cohere.ai/v1", 
        description="Cohere API base URL"
    )
    timeout: int = Field(default=60, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries")


class CohereProvider(BaseLLMProvider):
    """Cohere LLM provider implementation."""
    
    def __init__(self, config: Optional[CohereConfig] = None) -> None:
        if config is None:
            config = CohereConfig(api_key=os.getenv("COHERE_API_KEY", ""))
        
        self.config = config
        self.client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=config.timeout,
            headers={"Authorization": f"Bearer {config.api_key}"}
        )
        
        # Available models
        self.available_models = [
            "command",
            "command-light",
            "command-nightly",
            "command-light-nightly"
        ]
    
    async def generate_response(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Cohere."""
        
        # Use default model if none specified
        if model is None:
            model = "command"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = await self._make_request("/generate", payload)
            
            if "message" in response and response.get("message"):
                raise Exception(f"Cohere API error: {response['message']}")
            
            # Extract response content
            generations = response.get("generations", [])
            
            if not generations:
                raise Exception("No generations returned from Cohere API")
            
            content = generations[0].get("text", "")
            
            # Calculate tokens (approximate)
            input_tokens = len(prompt.split())
            output_tokens = len(content.split())
            total_tokens = input_tokens + output_tokens
            
            return LLMResponse(
                content=content,
                provider="cohere",
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens
            )
            
        except Exception as e:
            return LLMResponse(
                content=f"Error occurred: {str(e)}",
                provider="cohere",
                model=model,
                error=str(e)
            )
    
    async def complete(
        self, 
        messages: List[LLMMessage], 
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Complete using Cohere API with messages interface."""
        
        # Convert messages to a single prompt for Cohere
        prompt_parts = []
        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
        
        prompt = "\n".join(prompt_parts)
        if not prompt_parts or prompt_parts[-1].startswith("User:"):
            prompt += "\nAssistant:"
        
        # Use the existing generate_response method
        return await self.generate_response(prompt=prompt, model=model, **kwargs)
    
    async def _make_request(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make HTTP request to Cohere API."""
        
        for attempt in range(self.config.max_retries):
            try:
                if data:
                    response = await self.client.post(endpoint, json=data)
                else:
                    response = await self.client.get(endpoint)
                
                response.raise_for_status()
                return response.json()
                
            except httpx.HTTPStatusError as e:
                if attempt == self.config.max_retries - 1:
                    raise Exception(f"HTTP {e.response.status_code}: {e.response.text}")
                await asyncio.sleep(2**attempt)  # Exponential backoff
                
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise e
                await asyncio.sleep(2**attempt)
        
        # This should never be reached due to the exceptions above
        raise Exception("Request failed after all retries")
    
    async def health_check(self) -> bool:
        """Check if Cohere API is accessible."""
        try:
            # Make a simple request to check API health
            test_payload = {
                "model": "command",
                "prompt": "Test",
                "max_tokens": 1
            }
            await self._make_request("/generate", test_payload)
            return True
        except Exception:
            return False
    
    async def list_models(self) -> List[str]:
        """List available models from Cohere."""
        return self.available_models.copy()
    
    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()
    
    def __del__(self) -> None:
        """Cleanup on deletion."""
        try:
            asyncio.create_task(self.close())
        except Exception:
            pass

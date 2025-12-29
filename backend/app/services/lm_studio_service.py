"""
LM Studio integration service.

Handles communication with LM Studio's OpenAI-compatible API for model operations.
LM Studio provides an OpenAI-compatible API interface on port 1234 by default.
"""

from typing import Any

import httpx

from app.core.config import settings


class LMStudioServiceError(Exception):
    """Base exception for LM Studio service errors."""
    pass


class LMStudioConnectionError(LMStudioServiceError):
    """Raised when connection to LM Studio fails."""
    pass


class LMStudioModelError(LMStudioServiceError):
    """Raised when model operations fail."""
    pass


class LMStudioService:
    """
    Service for interacting with LM Studio's OpenAI-compatible API.
    
    Handles model operations including status checks, model listing,
    and inference for training purposes. Uses OpenAI-compatible endpoints.
    
    Attributes:
        base_url: LM Studio API base URL (typically http://localhost:1234).
        model: Default model name (typically "local-model").
        timeout: Request timeout in seconds.
    """
    
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ):
        """
        Initialize the LM Studio service.
        
        Args:
            base_url: LM Studio API base URL. Defaults to settings or http://localhost:1234.
            model: Default model name. Defaults to "local-model".
            timeout: Request timeout in seconds. Defaults to settings or 300.
        """
        self.base_url = base_url or getattr(settings, "ollama_base_url", "http://localhost:1234").replace(":11434", ":1234")
        self.model = model or "local-model"
        self.timeout = timeout or getattr(settings, "ollama_timeout", 300)
    
    async def check_health(self) -> bool:
        """
        Check if LM Studio service is healthy.
        
        Uses the /v1/models endpoint to verify the service is responding.
        
        Returns:
            True if LM Studio is reachable and responding.
        """
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.base_url}/v1/models")
                return response.status_code == 200
        except Exception:
            return False
    
    async def list_models(self) -> list[dict[str, Any]]:
        """
        List all available models.
        
        Transforms LM Studio's OpenAI-compatible model list format to match
        Ollama's format for compatibility.
        
        Returns:
            List of model information dictionaries with 'name' key.
            
        Raises:
            LMStudioConnectionError: If connection fails.
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/v1/models")
                response.raise_for_status()
                data = response.json()
                
                # Transform OpenAI format to Ollama-compatible format
                models = []
                for model_data in data.get("data", []):
                    models.append({
                        "name": model_data.get("id", ""),
                        "object": model_data.get("object", "model"),
                        "created": model_data.get("created", 0),
                    })
                return models
        except httpx.HTTPError as e:
            raise LMStudioConnectionError(f"Failed to list models: {str(e)}")
    
    async def model_exists(self, model_name: str | None = None) -> bool:
        """
        Check if a model exists.
        
        Args:
            model_name: Model name to check. Defaults to configured model.
            
        Returns:
            True if model exists.
        """
        model = model_name or self.model
        try:
            models = await self.list_models()
            return any(m.get("name", "") == model for m in models)
        except LMStudioConnectionError:
            return False
    
    async def generate(
        self,
        prompt: str,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text using the model via OpenAI-compatible chat completions.
        
        Converts the prompt to OpenAI message format and extracts the response.
        
        Args:
            prompt: Input prompt.
            model_name: Model to use. Defaults to configured model.
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.).
            
        Returns:
            Generated text response.
            
        Raises:
            LMStudioModelError: If generation fails.
        """
        model = model_name or self.model
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Prepare OpenAI-compatible request
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                    **kwargs,
                }
                
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                
                # Extract response from OpenAI format
                choices = data.get("choices", [])
                if choices and "message" in choices[0]:
                    return choices[0]["message"].get("content", "")
                return ""
        except httpx.HTTPError as e:
            raise LMStudioModelError(f"Generation failed: {str(e)}")
    
    def sync_check_health(self) -> bool:
        """
        Synchronous health check.
        
        Returns:
            True if LM Studio is reachable and responding.
        """
        try:
            with httpx.Client(timeout=10) as client:
                response = client.get(f"{self.base_url}/v1/models")
                return response.status_code == 200
        except Exception:
            return False
    
    def sync_generate(
        self,
        prompt: str,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Synchronous text generation.
        
        Args:
            prompt: Input prompt.
            model_name: Model to use. Defaults to configured model.
            **kwargs: Additional generation parameters.
            
        Returns:
            Generated text response.
        """
        model = model_name or self.model
        try:
            with httpx.Client(timeout=self.timeout) as client:
                # Prepare OpenAI-compatible request
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                    **kwargs,
                }
                
                response = client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                
                # Extract response from OpenAI format
                choices = data.get("choices", [])
                if choices and "message" in choices[0]:
                    return choices[0]["message"].get("content", "")
                return ""
        except httpx.HTTPError as e:
            raise LMStudioModelError(f"Generation failed: {str(e)}")

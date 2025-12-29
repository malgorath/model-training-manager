"""
Ollama integration service.

Handles communication with the Ollama API for model operations.
"""

import asyncio
from typing import Any

import httpx

from app.core.config import settings


class OllamaServiceError(Exception):
    """Base exception for Ollama service errors."""
    pass


class OllamaConnectionError(OllamaServiceError):
    """Raised when connection to Ollama fails."""
    pass


class OllamaModelError(OllamaServiceError):
    """Raised when model operations fail."""
    pass


class OllamaService:
    """
    Service for interacting with the Ollama API.
    
    Handles model operations including status checks, model pulling,
    and inference for training purposes.
    
    Attributes:
        base_url: Ollama API base URL.
        model: Default model name.
        timeout: Request timeout in seconds.
    """
    
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ):
        """
        Initialize the Ollama service.
        
        Args:
            base_url: Ollama API base URL. Defaults to settings.
            model: Default model name. Defaults to settings.
            timeout: Request timeout in seconds. Defaults to settings.
        """
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.ollama_model
        self.timeout = timeout or settings.ollama_timeout
    
    async def check_health(self) -> bool:
        """
        Check if Ollama service is healthy.
        
        Returns:
            True if Ollama is reachable and responding.
        """
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False
    
    async def list_models(self) -> list[dict[str, Any]]:
        """
        List all available models.
        
        Returns:
            List of model information dictionaries.
            
        Raises:
            OllamaConnectionError: If connection fails.
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
                return data.get("models", [])
        except httpx.HTTPError as e:
            raise OllamaConnectionError(f"Failed to list models: {str(e)}")
    
    async def model_exists(self, model_name: str | None = None) -> bool:
        """
        Check if a model exists locally.
        
        Args:
            model_name: Model name to check. Defaults to configured model.
            
        Returns:
            True if model exists locally.
        """
        model = model_name or self.model
        try:
            models = await self.list_models()
            return any(m.get("name", "").startswith(model.split(":")[0]) for m in models)
        except OllamaConnectionError:
            return False
    
    async def pull_model(
        self,
        model_name: str | None = None,
        progress_callback: Any = None,
    ) -> bool:
        """
        Pull a model from Ollama registry.
        
        Args:
            model_name: Model name to pull. Defaults to configured model.
            progress_callback: Optional callback for progress updates.
            
        Returns:
            True if model was pulled successfully.
            
        Raises:
            OllamaModelError: If pull fails.
        """
        model = model_name or self.model
        try:
            async with httpx.AsyncClient(timeout=self.timeout * 10) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/pull",
                    json={"name": model},
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if progress_callback and line:
                            import json
                            try:
                                data = json.loads(line)
                                progress_callback(data)
                            except json.JSONDecodeError:
                                pass
            return True
        except httpx.HTTPError as e:
            raise OllamaModelError(f"Failed to pull model {model}: {str(e)}")
    
    async def generate(
        self,
        prompt: str,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text using the model.
        
        Args:
            prompt: Input prompt.
            model_name: Model to use. Defaults to configured model.
            **kwargs: Additional generation parameters.
            
        Returns:
            Generated text response.
            
        Raises:
            OllamaModelError: If generation fails.
        """
        model = model_name or self.model
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        **kwargs,
                    },
                )
                response.raise_for_status()
                data = response.json()
                return data.get("response", "")
        except httpx.HTTPError as e:
            raise OllamaModelError(f"Generation failed: {str(e)}")
    
    async def generate_stream(
        self,
        prompt: str,
        model_name: str | None = None,
        **kwargs: Any,
    ):
        """
        Generate text with streaming response.
        
        Args:
            prompt: Input prompt.
            model_name: Model to use. Defaults to configured model.
            **kwargs: Additional generation parameters.
            
        Yields:
            Generated text chunks.
            
        Raises:
            OllamaModelError: If generation fails.
        """
        model = model_name or self.model
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": True,
                        **kwargs,
                    },
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            import json
                            try:
                                data = json.loads(line)
                                if "response" in data:
                                    yield data["response"]
                            except json.JSONDecodeError:
                                pass
        except httpx.HTTPError as e:
            raise OllamaModelError(f"Streaming generation failed: {str(e)}")
    
    async def get_model_info(self, model_name: str | None = None) -> dict[str, Any]:
        """
        Get information about a model.
        
        Args:
            model_name: Model name. Defaults to configured model.
            
        Returns:
            Model information dictionary.
            
        Raises:
            OllamaModelError: If model info cannot be retrieved.
        """
        model = model_name or self.model
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/show",
                    json={"name": model},
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            raise OllamaModelError(f"Failed to get model info: {str(e)}")
    
    def sync_check_health(self) -> bool:
        """
        Synchronous health check.
        
        Returns:
            True if Ollama is reachable and responding.
        """
        try:
            with httpx.Client(timeout=10) as client:
                response = client.get(f"{self.base_url}/api/tags")
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
                response = client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        **kwargs,
                    },
                )
                response.raise_for_status()
                data = response.json()
                return data.get("response", "")
        except httpx.HTTPError as e:
            raise OllamaModelError(f"Generation failed: {str(e)}")


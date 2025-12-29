"""
Model service factory.

Factory function to create the appropriate model service (OllamaService or LMStudioService)
based on provider configuration.
"""

from app.services.ollama_service import OllamaService
from app.services.lm_studio_service import LMStudioService


def get_model_service(
    provider: str,
    api_url: str,
    model: str | None = None,
    timeout: int | None = None,
):
    """
    Get the appropriate model service based on provider.
    
    Args:
        provider: Model provider type ("ollama" or "lm_studio").
        api_url: Base URL for the model API server.
        model: Model name (optional, uses service default if not provided).
        timeout: Request timeout in seconds (optional, uses service default if not provided).
        
    Returns:
        OllamaService or LMStudioService instance based on provider.
        
    Raises:
        ValueError: If provider is not supported.
        
    Example:
        >>> service = get_model_service(
        ...     provider="ollama",
        ...     api_url="http://localhost:11434",
        ...     model="llama3.2:3b",
        ...     timeout=300
        ... )
        >>> isinstance(service, OllamaService)
        True
    """
    if provider == "ollama":
        return OllamaService(
            base_url=api_url,
            model=model,
            timeout=timeout,
        )
    elif provider == "lm_studio":
        return LMStudioService(
            base_url=api_url,
            model=model,
            timeout=timeout,
        )
    else:
        raise ValueError(
            f"Unsupported model provider: {provider}. "
            "Supported providers are: 'ollama', 'lm_studio'"
        )

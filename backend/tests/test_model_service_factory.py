"""
Tests for the model service factory.

Tests the factory function that creates the appropriate model service
(OllamaService or LMStudioService) based on provider configuration.
"""

import pytest
from unittest.mock import patch

from app.services.model_service_factory import get_model_service
from app.services.ollama_service import OllamaService
from app.services.lm_studio_service import LMStudioService


class TestModelServiceFactory:
    """Tests for model service factory."""
    
    def test_get_model_service_ollama(self):
        """Test factory returns OllamaService for ollama provider."""
        service = get_model_service(
            provider="ollama",
            api_url="http://localhost:11434",
            model="llama3.2:3b",
            timeout=300,
        )
        
        assert isinstance(service, OllamaService)
        assert service.base_url == "http://localhost:11434"
        assert service.model == "llama3.2:3b"
        assert service.timeout == 300
    
    def test_get_model_service_lm_studio(self):
        """Test factory returns LMStudioService for lm_studio provider."""
        service = get_model_service(
            provider="lm_studio",
            api_url="http://localhost:1234",
            model="local-model",
            timeout=300,
        )
        
        assert isinstance(service, LMStudioService)
        assert service.base_url == "http://localhost:1234"
        assert service.model == "local-model"
        assert service.timeout == 300
    
    def test_get_model_service_defaults(self):
        """Test factory with default parameters."""
        service = get_model_service(
            provider="ollama",
            api_url="http://localhost:11434",
        )
        
        assert isinstance(service, OllamaService)
        assert service.base_url == "http://localhost:11434"
        # Model and timeout should use service defaults
    
    def test_get_model_service_invalid_provider(self):
        """Test factory raises ValueError for invalid provider."""
        with pytest.raises(ValueError, match="Unsupported model provider"):
            get_model_service(
                provider="invalid_provider",
                api_url="http://localhost:11434",
            )
    
    def test_get_model_service_ollama_custom_timeout(self):
        """Test factory passes custom timeout to OllamaService."""
        service = get_model_service(
            provider="ollama",
            api_url="http://localhost:11434",
            model="llama3.2:3b",
            timeout=600,
        )
        
        assert isinstance(service, OllamaService)
        assert service.timeout == 600
    
    def test_get_model_service_lm_studio_custom_model(self):
        """Test factory passes custom model to LMStudioService."""
        service = get_model_service(
            provider="lm_studio",
            api_url="http://localhost:1234",
            model="custom-model",
            timeout=300,
        )
        
        assert isinstance(service, LMStudioService)
        assert service.model == "custom-model"
    
    def test_get_model_service_different_urls(self):
        """Test factory handles different URLs for each provider."""
        ollama_service = get_model_service(
            provider="ollama",
            api_url="http://192.168.1.100:11434",
            model="test-model",
        )
        
        lm_service = get_model_service(
            provider="lm_studio",
            api_url="http://192.168.1.100:1234",
            model="test-model",
        )
        
        assert ollama_service.base_url == "http://192.168.1.100:11434"
        assert lm_service.base_url == "http://192.168.1.100:1234"

"""
Tests for the Ollama service.

Tests Ollama API client operations with mocked HTTP responses.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from app.services.ollama_service import (
    OllamaService,
    OllamaConnectionError,
    OllamaModelError,
)


class TestOllamaService:
    """Tests for OllamaService."""
    
    @pytest.fixture
    def service(self) -> OllamaService:
        """Create an OllamaService instance for testing."""
        return OllamaService(
            base_url="http://localhost:11434",
            model="llama3.2:3b",
            timeout=30,
        )
    
    @pytest.mark.asyncio
    async def test_check_health_success(self, service: OllamaService):
        """Test health check when Ollama is healthy."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance
            
            result = await service.check_health()
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_check_health_failure(self, service: OllamaService):
        """Test health check when Ollama is unreachable."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance
            
            result = await service.check_health()
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_list_models(self, service: OllamaService):
        """Test listing available models."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "models": [
                    {"name": "llama3.2:3b", "size": 1234567890},
                    {"name": "mistral:7b", "size": 9876543210},
                ]
            }
            mock_response.raise_for_status = MagicMock()
            
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance
            
            models = await service.list_models()
            
            assert len(models) == 2
            assert models[0]["name"] == "llama3.2:3b"
    
    @pytest.mark.asyncio
    async def test_list_models_connection_error(self, service: OllamaService):
        """Test list models when connection fails."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance
            
            with pytest.raises(OllamaConnectionError):
                await service.list_models()
    
    @pytest.mark.asyncio
    async def test_model_exists_true(self, service: OllamaService):
        """Test model exists when model is available."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "models": [
                    {"name": "llama3.2:3b"},
                ]
            }
            mock_response.raise_for_status = MagicMock()
            
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance
            
            result = await service.model_exists("llama3.2:3b")
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_model_exists_false(self, service: OllamaService):
        """Test model exists when model is not available."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "models": [
                    {"name": "mistral:7b"},
                ]
            }
            mock_response.raise_for_status = MagicMock()
            
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance
            
            result = await service.model_exists("llama3.2:3b")
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_generate(self, service: OllamaService):
        """Test text generation."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": "Hello, this is a test response.",
                "done": True,
            }
            mock_response.raise_for_status = MagicMock()
            
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance
            
            result = await service.generate("Hello, world!")
            
            assert result == "Hello, this is a test response."
    
    @pytest.mark.asyncio
    async def test_generate_error(self, service: OllamaService):
        """Test text generation when API call fails."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "Server Error",
                    request=MagicMock(),
                    response=MagicMock(status_code=500),
                )
            )
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance
            
            with pytest.raises(OllamaModelError):
                await service.generate("Hello")
    
    @pytest.mark.asyncio
    async def test_get_model_info(self, service: OllamaService):
        """Test getting model information."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "name": "llama3.2:3b",
                "parameters": "3B",
                "template": "...",
            }
            mock_response.raise_for_status = MagicMock()
            
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance
            
            info = await service.get_model_info()
            
            assert info["name"] == "llama3.2:3b"
            assert info["parameters"] == "3B"
    
    def test_sync_check_health_success(self, service: OllamaService):
        """Test synchronous health check when Ollama is healthy."""
        with patch("httpx.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            
            mock_instance = MagicMock()
            mock_instance.get = MagicMock(return_value=mock_response)
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=None)
            mock_client.return_value = mock_instance
            
            result = service.sync_check_health()
            
            assert result is True
    
    def test_sync_check_health_failure(self, service: OllamaService):
        """Test synchronous health check when Ollama is unreachable."""
        with patch("httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_instance.get = MagicMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=None)
            mock_client.return_value = mock_instance
            
            result = service.sync_check_health()
            
            assert result is False
    
    def test_sync_generate(self, service: OllamaService):
        """Test synchronous text generation."""
        with patch("httpx.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": "Sync response",
                "done": True,
            }
            mock_response.raise_for_status = MagicMock()
            
            mock_instance = MagicMock()
            mock_instance.post = MagicMock(return_value=mock_response)
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=None)
            mock_client.return_value = mock_instance
            
            result = service.sync_generate("Test prompt")
            
            assert result == "Sync response"
    
    def test_sync_generate_error(self, service: OllamaService):
        """Test sync generation when API call fails."""
        with patch("httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_instance.post = MagicMock(
                side_effect=httpx.HTTPStatusError(
                    "Server Error",
                    request=MagicMock(),
                    response=MagicMock(status_code=500),
                )
            )
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=None)
            mock_client.return_value = mock_instance
            
            with pytest.raises(OllamaModelError):
                service.sync_generate("Hello")
    
    @pytest.mark.asyncio
    async def test_model_exists_connection_error(self, service: OllamaService):
        """Test model_exists when connection fails."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance
            
            result = await service.model_exists("llama3.2:3b")
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_pull_model_success(self, service: OllamaService):
        """Test pulling a model successfully."""
        with patch("httpx.AsyncClient") as mock_client:
            # Create an async iterator for streaming
            async def mock_aiter_lines():
                yield '{"status": "downloading", "completed": 50}'
                yield '{"status": "downloading", "completed": 100}'
                yield '{"status": "success"}'
            
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.aiter_lines = mock_aiter_lines
            
            mock_stream = AsyncMock()
            mock_stream.__aenter__ = AsyncMock(return_value=mock_response)
            mock_stream.__aexit__ = AsyncMock(return_value=None)
            
            mock_instance = AsyncMock()
            mock_instance.stream = MagicMock(return_value=mock_stream)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance
            
            progress_updates = []
            def progress_callback(data):
                progress_updates.append(data)
            
            result = await service.pull_model("llama3.2:3b", progress_callback)
            
            assert result is True
            assert len(progress_updates) > 0
    
    @pytest.mark.asyncio
    async def test_pull_model_error(self, service: OllamaService):
        """Test pull_model when request fails."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_stream = AsyncMock()
            mock_stream.__aenter__ = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "Not Found",
                    request=MagicMock(),
                    response=MagicMock(status_code=404),
                )
            )
            mock_stream.__aexit__ = AsyncMock(return_value=None)
            
            mock_instance = AsyncMock()
            mock_instance.stream = MagicMock(return_value=mock_stream)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance
            
            with pytest.raises(OllamaModelError):
                await service.pull_model("nonexistent-model")
    
    @pytest.mark.asyncio
    async def test_generate_stream(self, service: OllamaService):
        """Test streaming text generation."""
        with patch("httpx.AsyncClient") as mock_client:
            async def mock_aiter_lines():
                yield '{"response": "Hello"}'
                yield '{"response": " world"}'
                yield '{"response": "!", "done": true}'
            
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.aiter_lines = mock_aiter_lines
            
            mock_stream = AsyncMock()
            mock_stream.__aenter__ = AsyncMock(return_value=mock_response)
            mock_stream.__aexit__ = AsyncMock(return_value=None)
            
            mock_instance = AsyncMock()
            mock_instance.stream = MagicMock(return_value=mock_stream)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance
            
            chunks = []
            async for chunk in service.generate_stream("Hello"):
                chunks.append(chunk)
            
            assert len(chunks) == 3
            assert chunks[0] == "Hello"
            assert chunks[1] == " world"
            assert chunks[2] == "!"
    
    @pytest.mark.asyncio
    async def test_generate_stream_error(self, service: OllamaService):
        """Test streaming generation when request fails."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_stream = AsyncMock()
            mock_stream.__aenter__ = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "Server Error",
                    request=MagicMock(),
                    response=MagicMock(status_code=500),
                )
            )
            mock_stream.__aexit__ = AsyncMock(return_value=None)
            
            mock_instance = AsyncMock()
            mock_instance.stream = MagicMock(return_value=mock_stream)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance
            
            with pytest.raises(OllamaModelError):
                async for _ in service.generate_stream("Hello"):
                    pass
    
    @pytest.mark.asyncio
    async def test_get_model_info_error(self, service: OllamaService):
        """Test get_model_info when request fails."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "Not Found",
                    request=MagicMock(),
                    response=MagicMock(status_code=404),
                )
            )
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance
            
            with pytest.raises(OllamaModelError):
                await service.get_model_info()
    
    def test_service_default_initialization(self):
        """Test OllamaService with default settings."""
        service = OllamaService()
        
        assert service.base_url is not None
        assert service.model is not None
        assert service.timeout is not None
    
    @pytest.mark.asyncio
    async def test_generate_with_kwargs(self, service: OllamaService):
        """Test generation with additional parameters."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": "Response with custom params",
                "done": True,
            }
            mock_response.raise_for_status = MagicMock()
            
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance
            
            result = await service.generate(
                "Hello",
                temperature=0.7,
                top_p=0.9,
            )
            
            assert result == "Response with custom params"
            
            # Verify the call was made with kwargs
            call_args = mock_instance.post.call_args
            json_body = call_args.kwargs.get("json") or call_args[1].get("json")
            assert json_body["temperature"] == 0.7
            assert json_body["top_p"] == 0.9


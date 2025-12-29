"""
Tests for the LM Studio service.

Tests LM Studio OpenAI-compatible API client operations with mocked HTTP responses.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from app.services.lm_studio_service import (
    LMStudioService,
    LMStudioConnectionError,
    LMStudioModelError,
)


class TestLMStudioService:
    """Tests for LMStudioService."""
    
    @pytest.fixture
    def service(self) -> LMStudioService:
        """Create an LMStudioService instance for testing."""
        return LMStudioService(
            base_url="http://localhost:1234",
            model="local-model",
            timeout=30,
        )
    
    @pytest.mark.asyncio
    async def test_check_health_success(self, service: LMStudioService):
        """Test health check when LM Studio is healthy."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [
                    {"id": "local-model", "object": "model"},
                ]
            }
            
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance
            
            result = await service.check_health()
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_check_health_failure(self, service: LMStudioService):
        """Test health check when LM Studio is unreachable."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance
            
            result = await service.check_health()
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_list_models(self, service: LMStudioService):
        """Test listing available models."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [
                    {"id": "local-model", "object": "model", "created": 1234567890},
                    {"id": "another-model", "object": "model", "created": 1234567891},
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
            assert models[0]["name"] == "local-model"
            assert models[1]["name"] == "another-model"
    
    @pytest.mark.asyncio
    async def test_list_models_connection_error(self, service: LMStudioService):
        """Test list models when connection fails."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance
            
            with pytest.raises(LMStudioConnectionError):
                await service.list_models()
    
    @pytest.mark.asyncio
    async def test_model_exists_true(self, service: LMStudioService):
        """Test model exists when model is available."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [
                    {"id": "local-model"},
                ]
            }
            mock_response.raise_for_status = MagicMock()
            
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance
            
            result = await service.model_exists("local-model")
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_model_exists_false(self, service: LMStudioService):
        """Test model exists when model is not available."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [
                    {"id": "other-model"},
                ]
            }
            mock_response.raise_for_status = MagicMock()
            
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance
            
            result = await service.model_exists("local-model")
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_generate(self, service: LMStudioService):
        """Test text generation using OpenAI-compatible chat completions."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Hello, this is a test response.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 10},
            }
            mock_response.raise_for_status = MagicMock()
            
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance
            
            result = await service.generate("Hello, world!")
            
            assert result == "Hello, this is a test response."
            
            # Verify OpenAI-compatible format was used
            call_args = mock_instance.post.call_args
            json_body = call_args.kwargs.get("json") or call_args[1].get("json")
            assert json_body["model"] == "local-model"
            assert "messages" in json_body
            assert json_body["messages"][0]["role"] == "user"
            assert json_body["messages"][0]["content"] == "Hello, world!"
    
    @pytest.mark.asyncio
    async def test_generate_error(self, service: LMStudioService):
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
            
            with pytest.raises(LMStudioModelError):
                await service.generate("Hello")
    
    def test_sync_check_health_success(self, service: LMStudioService):
        """Test synchronous health check when LM Studio is healthy."""
        with patch("httpx.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [
                    {"id": "local-model", "object": "model"},
                ]
            }
            
            mock_instance = MagicMock()
            mock_instance.get = MagicMock(return_value=mock_response)
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=None)
            mock_client.return_value = mock_instance
            
            result = service.sync_check_health()
            
            assert result is True
    
    def test_sync_check_health_failure(self, service: LMStudioService):
        """Test synchronous health check when LM Studio is unreachable."""
        with patch("httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_instance.get = MagicMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=None)
            mock_client.return_value = mock_instance
            
            result = service.sync_check_health()
            
            assert result is False
    
    def test_sync_generate(self, service: LMStudioService):
        """Test synchronous text generation."""
        with patch("httpx.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Sync response",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 5},
            }
            mock_response.raise_for_status = MagicMock()
            
            mock_instance = MagicMock()
            mock_instance.post = MagicMock(return_value=mock_response)
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=None)
            mock_client.return_value = mock_instance
            
            result = service.sync_generate("Test prompt")
            
            assert result == "Sync response"
    
    def test_sync_generate_error(self, service: LMStudioService):
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
            
            with pytest.raises(LMStudioModelError):
                service.sync_generate("Hello")
    
    @pytest.mark.asyncio
    async def test_model_exists_connection_error(self, service: LMStudioService):
        """Test model_exists when connection fails."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance
            
            result = await service.model_exists("local-model")
            
            assert result is False
    
    def test_service_default_initialization(self):
        """Test LMStudioService with default settings."""
        service = LMStudioService()
        
        assert service.base_url is not None
        assert service.model is not None
        assert service.timeout is not None
    
    @pytest.mark.asyncio
    async def test_generate_with_kwargs(self, service: LMStudioService):
        """Test generation with additional parameters."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Response with custom params",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 10},
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
                max_tokens=500,
            )
            
            assert result == "Response with custom params"
            
            # Verify the call was made with kwargs
            call_args = mock_instance.post.call_args
            json_body = call_args.kwargs.get("json") or call_args[1].get("json")
            assert json_body["temperature"] == 0.7
            assert json_body["max_tokens"] == 500

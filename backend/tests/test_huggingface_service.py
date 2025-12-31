"""
Tests for the Hugging Face service.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from app.services.huggingface_service import HuggingFaceService


class TestHuggingFaceService:
    """Tests for HuggingFaceService class."""

    def test_init(self):
        """Test service initialization."""
        service = HuggingFaceService(timeout=60)
        assert service.timeout == 60

    @pytest.mark.asyncio
    async def test_search_datasets_success(self):
        """Test successful dataset search."""
        mock_response = [
            {
                "id": "squad",
                "author": "rajpurkar",
                "description": "Stanford Question Answering Dataset",
                "downloads": 100000,
                "likes": 500,
                "tags": ["nlp", "qa"],
                "lastModified": "2024-01-01",
                "private": False,
            },
            {
                "id": "imdb",
                "author": "amaury",
                "description": "IMDB Movie Reviews",
                "downloads": 50000,
                "likes": 200,
                "tags": ["nlp", "sentiment"],
                "lastModified": "2024-01-02",
                "private": False,
            },
        ]

        with patch("httpx.AsyncClient") as mock_client:
            mock_async_client = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_async_client

            mock_http_response = MagicMock()
            mock_http_response.json.return_value = mock_response
            mock_http_response.raise_for_status = MagicMock()
            mock_async_client.get.return_value = mock_http_response

            service = HuggingFaceService()
            results = await service.search_datasets("qa", limit=10, offset=0)

            assert results["query"] == "qa"
            assert results["limit"] == 10
            assert results["offset"] == 0
            assert len(results["items"]) == 2
            assert results["items"][0]["id"] == "squad"
            assert results["items"][1]["id"] == "imdb"

    @pytest.mark.asyncio
    async def test_search_datasets_error(self):
        """Test dataset search with HTTP error."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_async_client = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_async_client
            mock_async_client.get.side_effect = httpx.HTTPError("Connection failed")

            service = HuggingFaceService()
            
            with pytest.raises(ValueError, match="Failed to search datasets"):
                await service.search_datasets("test")

    @pytest.mark.asyncio
    async def test_get_dataset_info_success(self):
        """Test getting dataset info."""
        mock_response = {
            "id": "squad",
            "author": "rajpurkar",
            "description": "Stanford QA Dataset",
            "downloads": 100000,
            "likes": 500,
            "tags": ["qa"],
            "cardData": {},
            "siblings": [],
            "private": False,
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_async_client = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_async_client

            mock_http_response = MagicMock()
            mock_http_response.json.return_value = mock_response
            mock_http_response.raise_for_status = MagicMock()
            mock_async_client.get.return_value = mock_http_response

            service = HuggingFaceService()
            info = await service.get_dataset_info("squad")

            assert info["id"] == "squad"
            assert info["author"] == "rajpurkar"

    @pytest.mark.asyncio
    async def test_get_dataset_info_error(self):
        """Test getting dataset info with HTTP error."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_async_client = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_async_client
            mock_async_client.get.side_effect = httpx.HTTPError("Not found")

            service = HuggingFaceService()
            
            with pytest.raises(ValueError, match="Failed to get dataset info"):
                await service.get_dataset_info("nonexistent")

    def test_download_dataset_success(self, tmp_path):
        """Test successful dataset download."""
        # Create mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_dataset.column_names = ["text", "label"]
        mock_dataset.__iter__ = MagicMock(
            return_value=iter([{"text": "sample", "label": 1}] * 100)
        )
        mock_dataset.select = MagicMock(return_value=mock_dataset)

        with patch("app.services.huggingface_service.load_dataset") as mock_load:
            mock_load.return_value = mock_dataset

            with patch("app.services.huggingface_service.settings") as mock_settings:
                mock_settings.get_dataset_path.return_value = tmp_path / "test" / "dataset"

                service = HuggingFaceService()

                file_path, row_count, columns = service.download_dataset(
                    dataset_id="test/dataset",
                    split="train",
                    max_rows=50,
                )

                assert file_path.exists()
                assert row_count == 100
                assert columns == ["text", "label"]
                assert "train.json" in str(file_path)
                assert "test" in str(file_path)
                assert "dataset" in str(file_path)

    def test_download_dataset_with_config(self, tmp_path):
        """Test dataset download with config."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)
        mock_dataset.column_names = ["question", "answer"]
        mock_dataset.__iter__ = MagicMock(
            return_value=iter([{"question": "q", "answer": "a"}] * 10)
        )

        with patch("app.services.huggingface_service.load_dataset") as mock_load:
            mock_load.return_value = mock_dataset

            with patch("app.services.huggingface_service.settings") as mock_settings:
                mock_settings.get_dataset_path.return_value = tmp_path / "test" / "dataset_en"

                service = HuggingFaceService()

                file_path, row_count, columns = service.download_dataset(
                    dataset_id="test/dataset",
                    split="test",
                    config="en",
                )

                mock_load.assert_called_once_with(
                    "test/dataset", "en", split="test", trust_remote_code=True
                )
                assert "test.json" in str(file_path)
                assert "test" in str(file_path)
                assert "dataset_en" in str(file_path)

    def test_download_dataset_error(self, tmp_path):
        """Test dataset download with error."""
        with patch("app.services.huggingface_service.load_dataset") as mock_load:
            mock_load.side_effect = Exception("Dataset not found")

            with patch("app.services.huggingface_service.settings") as mock_settings:
                mock_settings.get_dataset_path.return_value = tmp_path / "huggingface" / "nonexistent_dataset"

                service = HuggingFaceService()

                with pytest.raises(ValueError, match="Failed to download dataset"):
                    service.download_dataset("nonexistent/dataset", "train")

    @pytest.mark.asyncio
    async def test_list_dataset_configs(self):
        """Test listing dataset configs."""
        mock_response = {
            "id": "squad",
            "cardData": {
                "configs": [
                    {"config_name": "plain_text"},
                    {"name": "v2"},
                ]
            },
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_async_client = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_async_client

            mock_http_response = MagicMock()
            mock_http_response.json.return_value = mock_response
            mock_http_response.raise_for_status = MagicMock()
            mock_async_client.get.return_value = mock_http_response

            service = HuggingFaceService()
            configs = await service.list_dataset_configs("squad")

            assert "plain_text" in configs

    @pytest.mark.asyncio
    async def test_list_dataset_configs_error(self):
        """Test listing configs with HTTP error."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_async_client = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_async_client
            mock_async_client.get.side_effect = httpx.HTTPError("Error")

            service = HuggingFaceService()
            configs = await service.list_dataset_configs("test")

            assert configs == []


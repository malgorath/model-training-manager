"""
Hugging Face Hub integration service.

Provides functionality to search and download datasets and models from Hugging Face.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import httpx
from datasets import load_dataset

from app.core.config import settings

logger = logging.getLogger(__name__)

# Hugging Face API base URL
HF_API_URL = "https://huggingface.co/api"


class HuggingFaceService:
    """
    Service for interacting with the Hugging Face Hub.
    
    Provides search and download capabilities for datasets and models.
    """
    
    def __init__(self, timeout: int = 30, hf_token: Optional[str] = None):
        """
        Initialize the Hugging Face service.
        
        Args:
            timeout: Request timeout in seconds.
            hf_token: HuggingFace API token for authenticated requests.
        """
        self.timeout = timeout
        self.hf_token = hf_token
    
    def _get_headers(self) -> dict[str, str]:
        """
        Get HTTP headers with authentication token if available.
        
        Returns:
            Dictionary with headers including Authorization if token is set.
        """
        headers = {}
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"
        return headers
    
    async def search_datasets(
        self,
        query: str,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        Search for datasets on Hugging Face Hub.
        
        Args:
            query: Search query string.
            limit: Maximum number of results.
            offset: Offset for pagination.
            
        Returns:
            Dictionary with search results and metadata.
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Search datasets using HF API
                params = {
                    "search": query,
                    "limit": limit,
                    "offset": offset,
                    "sort": "downloads",
                    "direction": "-1",
                }
                
                response = await client.get(
                    f"{HF_API_URL}/datasets",
                    params=params,
                    headers=self._get_headers(),
                )
                response.raise_for_status()
                
                datasets = response.json()
                
                # Format results
                results = []
                for ds in datasets:
                    results.append({
                        "id": ds.get("id", ""),
                        "name": ds.get("id", "").split("/")[-1],
                        "author": ds.get("author", ds.get("id", "").split("/")[0] if "/" in ds.get("id", "") else ""),
                        "description": ds.get("description", ""),
                        "downloads": ds.get("downloads", 0),
                        "likes": ds.get("likes", 0),
                        "tags": ds.get("tags", []),
                        "last_modified": ds.get("lastModified", ""),
                        "private": ds.get("private", False),
                    })
                
                return {
                    "items": results,
                    "query": query,
                    "limit": limit,
                    "offset": offset,
                }
                
        except httpx.HTTPError as e:
            logger.error(f"Failed to search HuggingFace datasets: {e}")
            raise ValueError(f"Failed to search datasets: {str(e)}")
    
    async def get_dataset_info(self, dataset_id: str) -> dict[str, Any]:
        """
        Get detailed information about a dataset.
        
        Args:
            dataset_id: The dataset ID (e.g., "squad", "imdb", "user/dataset").
            
        Returns:
            Dictionary with dataset information.
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{HF_API_URL}/datasets/{dataset_id}",
                )
                response.raise_for_status()
                
                data = response.json()
                
                return {
                    "id": data.get("id", ""),
                    "name": data.get("id", "").split("/")[-1],
                    "author": data.get("author", ""),
                    "description": data.get("description", ""),
                    "downloads": data.get("downloads", 0),
                    "likes": data.get("likes", 0),
                    "tags": data.get("tags", []),
                    "card_data": data.get("cardData", {}),
                    "siblings": data.get("siblings", []),
                    "private": data.get("private", False),
                }
                
        except httpx.HTTPError as e:
            logger.error(f"Failed to get dataset info for {dataset_id}: {e}")
            raise ValueError(f"Failed to get dataset info: {str(e)}")
    
    def download_dataset(
        self,
        dataset_id: str,
        split: str = "train",
        config: str | None = None,
        max_rows: int | None = None,
    ) -> tuple[Path, int, list[str]]:
        """
        Download a dataset from Hugging Face Hub.
        
        Args:
            dataset_id: The dataset ID (e.g., "squad", "imdb").
            split: Dataset split to download (train, test, validation).
            config: Optional dataset configuration/subset.
            max_rows: Maximum number of rows to download (None for all).
            
        Returns:
            Tuple of (file_path, row_count, column_names).
        """
        try:
            logger.info(f"Downloading dataset {dataset_id} (split={split}, config={config})")
            
            # Load dataset from HuggingFace
            if config:
                ds = load_dataset(dataset_id, config, split=split, trust_remote_code=True)
            else:
                ds = load_dataset(dataset_id, split=split, trust_remote_code=True)
            
            # Limit rows if specified
            if max_rows and len(ds) > max_rows:
                ds = ds.select(range(max_rows))
            
            # Get column names
            columns = ds.column_names
            
            # Parse dataset_id to extract author and dataset name
            # Format: "author/datasetname" or just "datasetname"
            if "/" in dataset_id:
                author, dataset_name = dataset_id.split("/", 1)
            else:
                author = "huggingface"
                dataset_name = dataset_id
            
            # Add config to dataset name if present
            if config:
                dataset_name = f"{dataset_name}_{config}"
            
            # Create directory structure: ./data/{author}/{datasetname}
            dataset_dir = settings.get_dataset_path(author, dataset_name)
            
            # Generate filename
            filename = f"{split}.json"
            file_path = dataset_dir / filename
            
            # Convert to list of dicts and save as JSON
            data = [dict(row) for row in ds]
            
            # Handle non-serializable types
            def make_serializable(obj):
                if isinstance(obj, bytes):
                    return obj.decode('utf-8', errors='replace')
                elif isinstance(obj, (list, tuple)):
                    return [make_serializable(item) for item in obj]
                elif isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                else:
                    try:
                        json.dumps(obj)
                        return obj
                    except (TypeError, ValueError):
                        return str(obj)
            
            data = [make_serializable(row) for row in data]
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            row_count = len(data)
            logger.info(f"Downloaded {row_count} rows to {file_path}")
            
            return file_path, row_count, columns
            
        except Exception as e:
            logger.error(f"Failed to download dataset {dataset_id}: {e}")
            raise ValueError(f"Failed to download dataset: {str(e)}")
    
    async def list_dataset_configs(self, dataset_id: str) -> list[str]:
        """
        List available configurations for a dataset.
        
        Args:
            dataset_id: The dataset ID.
            
        Returns:
            List of configuration names.
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{HF_API_URL}/datasets/{dataset_id}",
                    headers=self._get_headers(),
                )
                response.raise_for_status()
                
                data = response.json()
                card_data = data.get("cardData", {})
                configs = card_data.get("configs", [])
                
                if configs:
                    return [c.get("config_name", c.get("name", "")) for c in configs if isinstance(c, dict)]
                
                # Try to get from dataset info
                return []
                
        except httpx.HTTPError as e:
            logger.error(f"Failed to list configs for {dataset_id}: {e}")
            return []
    
    async def search_models(
        self,
        query: str,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        Search for models on Hugging Face Hub.
        
        Args:
            query: Search query string.
            limit: Maximum number of results.
            offset: Offset for pagination.
            
        Returns:
            Dictionary with search results and metadata.
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                params = {
                    "search": query,
                    "limit": limit,
                    "offset": offset,
                    "sort": "downloads",
                    "direction": "-1",
                }
                
                response = await client.get(
                    f"{HF_API_URL}/models",
                    params=params,
                    headers=self._get_headers(),
                )
                response.raise_for_status()
                
                models = response.json()
                
                # Format results
                results = []
                for model in models:
                    model_id = model.get("id", "")
                    parts = model_id.split("/") if "/" in model_id else ["", model_id]
                    results.append({
                        "id": model_id,
                        "name": parts[-1],
                        "author": parts[0] if len(parts) > 1 else "",
                        "description": model.get("description", ""),
                        "downloads": model.get("downloads", 0),
                        "likes": model.get("likes", 0),
                        "tags": model.get("tags", []),
                        "model_type": model.get("pipeline_tag", ""),
                        "private": model.get("private", False),
                        "last_modified": model.get("lastModified", ""),
                    })
                
                return {
                    "items": results,
                    "query": query,
                    "limit": limit,
                    "offset": offset,
                }
                
        except httpx.HTTPError as e:
            logger.error(f"Failed to search HuggingFace models: {e}")
            raise ValueError(f"Failed to search models: {str(e)}")
    
    async def get_model_info(self, model_id: str) -> dict[str, Any]:
        """
        Get detailed information about a model.
        
        Args:
            model_id: The model ID (e.g., "meta-llama/Llama-3.2-3B-Instruct").
            
        Returns:
            Dictionary with model information.
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{HF_API_URL}/models/{model_id}",
                    headers=self._get_headers(),
                )
                response.raise_for_status()
                
                data = response.json()
                model_id_val = data.get("id", model_id)
                parts = model_id_val.split("/") if "/" in model_id_val else ["", model_id_val]
                
                return {
                    "id": model_id_val,
                    "name": parts[-1],
                    "author": parts[0] if len(parts) > 1 else "",
                    "description": data.get("description", ""),
                    "downloads": data.get("downloads", 0),
                    "likes": data.get("likes", 0),
                    "tags": data.get("tags", []),
                    "model_type": data.get("pipeline_tag", ""),
                    "private": data.get("private", False),
                    "card_data": data.get("cardData", {}),
                    "siblings": data.get("siblings", []),
                }
                
        except httpx.HTTPError as e:
            logger.error(f"Failed to get model info for {model_id}: {e}")
            raise ValueError(f"Failed to get model info: {str(e)}")
    
    def download_model(
        self,
        model_id: str,
        local_path: Path,
    ) -> Path:
        """
        Download a model from Hugging Face Hub.
        
        Uses huggingface_hub library to download the model files.
        
        Args:
            model_id: The model ID (e.g., "meta-llama/Llama-3.2-3B-Instruct").
            local_path: Local directory path to save the model.
            
        Returns:
            Path to the downloaded model directory.
        """
        try:
            from huggingface_hub import snapshot_download
            
            logger.info(f"Downloading model {model_id} to {local_path}")
            
            # Ensure directory exists
            local_path.mkdir(parents=True, exist_ok=True)
            
            # Download model using huggingface_hub
            downloaded_path = snapshot_download(
                repo_id=model_id,
                local_dir=str(local_path),
                token=self.hf_token,
                local_dir_use_symlinks=False,
            )
            
            logger.info(f"Successfully downloaded model {model_id} to {downloaded_path}")
            return Path(downloaded_path)
            
        except ImportError:
            logger.error("huggingface_hub library is not installed")
            raise ValueError("huggingface_hub library is required for model downloads. Please install it: pip install huggingface_hub")
        except Exception as e:
            logger.error(f"Failed to download model {model_id}: {e}")
            raise ValueError(f"Failed to download model: {str(e)}")


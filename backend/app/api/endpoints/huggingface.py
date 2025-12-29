"""
Hugging Face Hub API endpoints.

Provides endpoints for searching and downloading datasets from Hugging Face.
"""

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.dataset import Dataset
from app.services.huggingface_service import HuggingFaceService

router = APIRouter(prefix="/huggingface", tags=["huggingface"])


class HFDatasetSearchResult(BaseModel):
    """Search result for a Hugging Face dataset."""
    id: str
    name: str
    author: str
    description: str | None = None
    downloads: int = 0
    likes: int = 0
    tags: list[str] = []
    last_modified: str | None = None
    private: bool = False


class HFSearchResponse(BaseModel):
    """Response for dataset search."""
    items: list[HFDatasetSearchResult]
    query: str
    limit: int
    offset: int


class HFDatasetInfo(BaseModel):
    """Detailed information about a dataset."""
    id: str
    name: str
    author: str
    description: str | None = None
    downloads: int = 0
    likes: int = 0
    tags: list[str] = []
    private: bool = False


class HFDownloadRequest(BaseModel):
    """Request to download a dataset from Hugging Face."""
    dataset_id: str = Field(..., description="HuggingFace dataset ID (e.g., 'squad', 'imdb')")
    name: str | None = Field(None, description="Custom name for the dataset")
    split: str = Field("train", description="Dataset split (train, test, validation)")
    config: str | None = Field(None, description="Dataset configuration/subset")
    max_rows: int | None = Field(None, description="Maximum rows to download", ge=1, le=1000000)


class HFDownloadResponse(BaseModel):
    """Response after downloading a dataset."""
    id: int
    name: str
    dataset_id: str
    row_count: int
    columns: list[str]
    file_path: str
    message: str


@router.get("/search", response_model=HFSearchResponse)
async def search_datasets(
    query: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
):
    """
    Search for datasets on Hugging Face Hub.
    
    Returns a list of datasets matching the search query.
    """
    service = HuggingFaceService()
    
    try:
        results = await service.search_datasets(query, limit, offset)
        return results
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/{dataset_id:path}/configs")
async def get_dataset_configs(dataset_id: str):
    """
    Get available configurations for a dataset.
    
    Some datasets have multiple configurations (e.g., different languages or subsets).
    """
    service = HuggingFaceService()
    
    try:
        configs = await service.list_dataset_configs(dataset_id)
        return {"configs": configs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/{dataset_id:path}", response_model=HFDatasetInfo)
async def get_dataset_info(dataset_id: str):
    """
    Get detailed information about a Hugging Face dataset.
    
    The dataset_id can be in format 'dataset_name' or 'author/dataset_name'.
    """
    service = HuggingFaceService()
    
    try:
        info = await service.get_dataset_info(dataset_id)
        return info
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/download", response_model=HFDownloadResponse)
def download_dataset(
    request: HFDownloadRequest,
    db: Session = Depends(get_db),
):
    """
    Download a dataset from Hugging Face Hub.
    
    This downloads the dataset and saves it to the uploads directory,
    then creates a database entry for it.
    """
    service = HuggingFaceService()
    
    try:
        # Download the dataset
        file_path, row_count, columns = service.download_dataset(
            dataset_id=request.dataset_id,
            split=request.split,
            config=request.config,
            max_rows=request.max_rows,
        )
        
        # Create dataset name
        name = request.name or f"{request.dataset_id.split('/')[-1]} ({request.split})"
        if request.config:
            name = f"{name} - {request.config}"
        
        # Get file size
        file_size = file_path.stat().st_size
        
        # Create database entry
        dataset = Dataset(
            name=name,
            description=f"Downloaded from Hugging Face: {request.dataset_id}",
            filename=file_path.name,
            file_path=str(file_path),
            file_type="json",
            file_size=file_size,
            row_count=row_count,
            column_count=len(columns),
            columns=json.dumps(columns),
        )
        
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        
        return HFDownloadResponse(
            id=dataset.id,
            name=dataset.name,
            dataset_id=request.dataset_id,
            row_count=row_count,
            columns=columns,
            file_path=str(file_path),
            message=f"Successfully downloaded {row_count} rows from {request.dataset_id}",
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


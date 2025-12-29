"""
Dataset API endpoints.

Handles dataset upload, listing, retrieval, and deletion.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.dataset import (
    DatasetCreate,
    DatasetUpdate,
    DatasetResponse,
    DatasetListResponse,
)
from app.services.dataset_service import DatasetService

router = APIRouter()


def get_dataset_service(db: Session = Depends(get_db)) -> DatasetService:
    """Dependency to get DatasetService instance."""
    return DatasetService(db)


@router.post("/", response_model=DatasetResponse, status_code=201)
async def upload_dataset(
    file: Annotated[UploadFile, File(description="CSV or JSON file to upload")],
    name: Annotated[str, Form(description="Dataset name")],
    description: Annotated[str | None, Form(description="Dataset description")] = None,
    service: DatasetService = Depends(get_dataset_service),
) -> DatasetResponse:
    """
    Upload a new dataset.
    
    Upload a CSV or JSON file to create a new dataset for training.
    """
    dataset_create = DatasetCreate(name=name, description=description)
    try:
        dataset = await service.create_dataset(file, dataset_create)
        return DatasetResponse.model_validate(dataset)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/", response_model=DatasetListResponse)
async def list_datasets(
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100, description="Items per page")] = 10,
    service: DatasetService = Depends(get_dataset_service),
) -> DatasetListResponse:
    """
    List all datasets with pagination.
    """
    result = service.list_datasets(page=page, page_size=page_size)
    return DatasetListResponse(
        items=[DatasetResponse.model_validate(d) for d in result["items"]],
        total=result["total"],
        page=result["page"],
        page_size=result["page_size"],
        pages=result["pages"],
    )


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: int,
    service: DatasetService = Depends(get_dataset_service),
) -> DatasetResponse:
    """
    Get a specific dataset by ID.
    """
    dataset = service.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return DatasetResponse.model_validate(dataset)


@router.patch("/{dataset_id}", response_model=DatasetResponse)
async def update_dataset(
    dataset_id: int,
    update_data: DatasetUpdate,
    service: DatasetService = Depends(get_dataset_service),
) -> DatasetResponse:
    """
    Update a dataset's metadata.
    """
    dataset = service.update_dataset(dataset_id, update_data)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return DatasetResponse.model_validate(dataset)


@router.delete("/{dataset_id}", status_code=204)
async def delete_dataset(
    dataset_id: int,
    service: DatasetService = Depends(get_dataset_service),
) -> None:
    """
    Delete a dataset.
    """
    success = service.delete_dataset(dataset_id)
    if not success:
        raise HTTPException(status_code=404, detail="Dataset not found")


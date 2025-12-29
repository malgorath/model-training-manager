"""
Dataset Pydantic schemas for request/response validation.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


class DatasetBase(BaseModel):
    """Base schema for dataset with common fields."""
    
    name: str = Field(..., min_length=1, max_length=255, description="Dataset name")
    description: Optional[str] = Field(None, description="Dataset description")


class DatasetCreate(DatasetBase):
    """Schema for creating a new dataset (used with file upload)."""
    pass


class DatasetUpdate(BaseModel):
    """Schema for updating an existing dataset."""
    
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Dataset name")
    description: Optional[str] = Field(None, description="Dataset description")


class DatasetResponse(DatasetBase):
    """Schema for dataset response."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: int = Field(..., description="Dataset ID")
    filename: str = Field(..., description="Original filename")
    file_path: str = Field(..., description="Storage path")
    file_type: str = Field(..., description="File type (csv or json)")
    file_size: int = Field(..., description="File size in bytes")
    row_count: int = Field(..., description="Number of rows")
    column_count: int = Field(..., description="Number of columns")
    columns: Optional[str] = Field(None, description="JSON array of column names")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class DatasetListResponse(BaseModel):
    """Schema for paginated dataset list response."""
    
    items: list[DatasetResponse] = Field(..., description="List of datasets")
    total: int = Field(..., description="Total number of datasets")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")
    pages: int = Field(..., description="Total number of pages")


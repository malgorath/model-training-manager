"""
Model-related Pydantic schemas for request/response validation.
"""

import json
from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, Field, ConfigDict, field_validator


class ModelSearchResponse(BaseModel):
    """Schema for model search response."""
    
    items: List[dict] = Field(..., description="List of model search results")
    query: str = Field(..., description="Search query")
    limit: int = Field(..., description="Results limit")
    offset: int = Field(..., description="Results offset")


class HuggingFaceModelInfo(BaseModel):
    """Schema for HuggingFace model information."""
    
    id: str = Field(..., description="Model ID")
    name: str = Field(..., description="Model name")
    author: str = Field(..., description="Model author/organization")
    description: str = Field("", description="Model description")
    downloads: int = Field(0, description="Number of downloads")
    likes: int = Field(0, description="Number of likes")
    tags: List[str] = Field(default_factory=list, description="Model tags")
    model_type: str = Field("", description="Model type/pipeline tag")
    private: bool = Field(False, description="Whether model is private")
    last_modified: Optional[str] = Field(None, description="Last modified date")
    card_data: Optional[dict] = Field(None, description="Model card data")
    siblings: Optional[List[dict]] = Field(None, description="Model file siblings")


class ModelDownloadRequest(BaseModel):
    """Schema for model download request."""
    
    model_id: str = Field(..., description="HuggingFace model ID to download", min_length=1)


class ModelDownloadResponse(BaseModel):
    """Schema for model download response."""
    
    id: int = Field(..., description="Downloaded model database ID")
    model_id: str = Field(..., description="HuggingFace model ID")
    name: str = Field(..., description="Model name")
    author: str = Field(..., description="Model author")
    local_path: str = Field(..., description="Local path to downloaded model")
    file_size: int = Field(..., description="Model file size in bytes")
    message: str = Field(..., description="Success message")


class LocalModelResponse(BaseModel):
    """Schema for local model response."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: int = Field(..., description="Model database ID")
    model_id: str = Field(..., description="HuggingFace model ID")
    name: str = Field(..., description="Model name")
    author: str = Field(..., description="Model author")
    description: Optional[str] = Field(None, description="Model description")
    local_path: str = Field(..., description="Local path to model")
    file_size: int = Field(..., description="Model file size in bytes")
    downloaded_at: datetime = Field(..., description="Download timestamp")
    tags: Optional[List[str]] = Field(default_factory=list, description="Model tags")
    model_type: Optional[str] = Field(None, description="Model type")
    is_private: bool = Field(False, description="Whether model is private")
    created_at: datetime = Field(..., description="Record creation timestamp")
    updated_at: datetime = Field(..., description="Record update timestamp")
    
    @field_validator('tags', mode='before')
    @classmethod
    def parse_tags(cls, v):
        """Parse tags from JSON string to list."""
        if v is None:
            return []
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
                return []
            except (json.JSONDecodeError, TypeError):
                return []
        if isinstance(v, list):
            return v
        return []
"""
Pydantic schemas for project API.

Defines request and response models for project endpoints.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict


class DatasetAllocation(BaseModel):
    """Dataset allocation within a trait."""
    
    dataset_id: int = Field(..., description="Dataset ID")
    percentage: float = Field(..., ge=0.0, le=100.0, description="Percentage of training data (0-100)")


class TraitConfiguration(BaseModel):
    """Trait configuration with dataset allocations."""
    
    trait_type: str = Field(..., description="Type of trait (reasoning, coding, general_tools)")
    datasets: List[DatasetAllocation] = Field(..., description="List of dataset allocations")


class ProjectCreate(BaseModel):
    """Schema for creating a new project."""
    
    name: str = Field(..., min_length=1, max_length=255, description="Project name")
    description: Optional[str] = Field(None, description="Project description")
    base_model: str = Field(..., description="Base model identifier")
    training_type: str = Field(..., description="Training type (qlora, unsloth, rag, standard)")
    max_rows: int = Field(..., description="Maximum rows for training (50000, 100000, 250000, 500000, 1000000)")
    output_directory: str = Field(..., description="Output directory path for trained model")
    traits: List[TraitConfiguration] = Field(..., min_length=1, description="List of trait configurations")
    
    model_config = ConfigDict(from_attributes=True)


class ProjectUpdate(BaseModel):
    """Schema for updating a project."""
    
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Project name")
    description: Optional[str] = Field(None, description="Project description")
    status: Optional[str] = Field(None, description="Project status")
    
    model_config = ConfigDict(from_attributes=True)


class ProjectTraitResponse(BaseModel):
    """Response schema for project trait."""
    
    id: int
    trait_type: str
    datasets: List[dict] = Field(..., description="List of dataset allocations with percentages")
    
    model_config = ConfigDict(from_attributes=True)


class ProjectResponse(BaseModel):
    """Response schema for project."""
    
    id: int
    name: str
    description: Optional[str]
    base_model: str
    training_type: str
    max_rows: int
    output_directory: str
    status: str
    progress: float
    current_epoch: int
    current_loss: Optional[float]
    error_message: Optional[str]
    model_path: Optional[str]
    worker_id: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    traits: List[ProjectTraitResponse] = Field(default_factory=list)
    
    model_config = ConfigDict(from_attributes=True)


class ProjectListResponse(BaseModel):
    """Response schema for project list."""
    
    items: List[ProjectResponse]
    total: int
    page: int = 1
    page_size: int = 50


class OutputDirectoryValidationRequest(BaseModel):
    """Request schema for output directory validation."""
    
    output_directory: str = Field(..., description="Directory path to validate")


class OutputDirectoryValidationResponse(BaseModel):
    """Response schema for output directory validation."""
    
    valid: bool
    writable: bool
    path: str
    errors: List[str] = Field(default_factory=list)


class ModelValidationRequest(BaseModel):
    """Request schema for model availability check."""
    
    model_name: str = Field(..., description="Model identifier to check")


class ModelValidationResponse(BaseModel):
    """Response schema for model availability."""
    
    available: bool
    model_name: str
    path: Optional[str] = None
    errors: List[str] = Field(default_factory=list)

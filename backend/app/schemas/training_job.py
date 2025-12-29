"""
Training job Pydantic schemas for request/response validation.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict

from app.models.training_job import TrainingStatus, TrainingType


class TrainingJobBase(BaseModel):
    """Base schema for training job with common fields."""
    
    name: str = Field(..., min_length=1, max_length=255, description="Job name")
    description: Optional[str] = Field(None, description="Job description")
    training_type: TrainingType = Field(
        TrainingType.QLORA,
        description="Type of training"
    )
    model_name: str = Field("llama3.2:3b", description="Model to train")


class TrainingJobCreate(TrainingJobBase):
    """Schema for creating a new training job."""
    
    dataset_id: int = Field(..., description="ID of dataset to use for training")
    
    # Training parameters (optional, will use defaults if not provided)
    batch_size: Optional[int] = Field(None, ge=1, le=64, description="Batch size")
    learning_rate: Optional[float] = Field(
        None,
        ge=1e-6,
        le=1.0,
        description="Learning rate"
    )
    epochs: Optional[int] = Field(None, ge=1, le=100, description="Number of epochs")
    lora_r: Optional[int] = Field(None, ge=1, le=256, description="LoRA rank")
    lora_alpha: Optional[int] = Field(None, ge=1, le=512, description="LoRA alpha")
    lora_dropout: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="LoRA dropout"
    )


class TrainingJobUpdate(BaseModel):
    """Schema for updating an existing training job."""
    
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Job name")
    description: Optional[str] = Field(None, description="Job description")


class TrainingJobProgress(BaseModel):
    """Schema for training job progress update."""
    
    progress: float = Field(..., ge=0.0, le=100.0, description="Progress percentage")
    current_epoch: int = Field(..., ge=0, description="Current epoch")
    current_loss: Optional[float] = Field(None, description="Current loss value")
    status: Optional[TrainingStatus] = Field(None, description="Job status")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class TrainingJobResponse(TrainingJobBase):
    """Schema for training job response."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: int = Field(..., description="Job ID")
    status: TrainingStatus = Field(..., description="Job status")
    dataset_id: int = Field(..., description="Dataset ID")
    
    # Training parameters
    batch_size: int = Field(..., description="Batch size")
    learning_rate: float = Field(..., description="Learning rate")
    epochs: int = Field(..., description="Number of epochs")
    lora_r: int = Field(..., description="LoRA rank")
    lora_alpha: int = Field(..., description="LoRA alpha")
    lora_dropout: float = Field(..., description="LoRA dropout")
    
    # Progress
    progress: float = Field(..., description="Progress percentage")
    current_epoch: int = Field(..., description="Current epoch")
    current_loss: Optional[float] = Field(None, description="Current loss")
    error_message: Optional[str] = Field(None, description="Error message")
    
    # Training logs and output
    log: Optional[str] = Field(None, description="Training log output")
    model_path: Optional[str] = Field(None, description="Path to saved model")
    
    # Worker
    worker_id: Optional[str] = Field(None, description="Assigned worker ID")
    
    # Timestamps
    started_at: Optional[datetime] = Field(None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class TrainingJobListResponse(BaseModel):
    """Schema for paginated training job list response."""
    
    items: list[TrainingJobResponse] = Field(..., description="List of training jobs")
    total: int = Field(..., description="Total number of jobs")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")
    pages: int = Field(..., description="Total number of pages")


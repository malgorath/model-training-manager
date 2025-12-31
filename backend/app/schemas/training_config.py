"""
Training configuration Pydantic schemas for request/response validation.
"""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


class TrainingConfigUpdate(BaseModel):
    """Schema for updating training configuration."""
    
    max_concurrent_workers: Optional[int] = Field(
        None,
        ge=1,
        le=32,
        description="Maximum concurrent workers"
    )
    default_model: Optional[str] = Field(None, description="Default model name")
    default_training_type: Optional[str] = Field(None, description="Default training type")
    
    # Training parameters
    default_batch_size: Optional[int] = Field(
        None,
        ge=1,
        le=64,
        description="Default batch size"
    )
    default_learning_rate: Optional[float] = Field(
        None,
        ge=1e-6,
        le=1.0,
        description="Default learning rate"
    )
    default_epochs: Optional[int] = Field(
        None,
        ge=1,
        le=100,
        description="Default epochs"
    )
    default_lora_r: Optional[int] = Field(
        None,
        ge=1,
        le=256,
        description="Default LoRA rank"
    )
    default_lora_alpha: Optional[int] = Field(
        None,
        ge=1,
        le=512,
        description="Default LoRA alpha"
    )
    default_lora_dropout: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Default LoRA dropout"
    )
    
    auto_start_workers: Optional[bool] = Field(
        None,
        description="Auto-start workers on startup"
    )
    
    model_provider: Optional[Literal["ollama", "lm_studio"]] = Field(
        None,
        description="Model provider type (ollama or lm_studio)"
    )
    model_api_url: Optional[str] = Field(
        None,
        description="Base URL for the model API server",
        min_length=1,
        max_length=500,
    )
    
    output_directory_base: Optional[str] = Field(
        None,
        description="Base directory for project output (models will be saved here)",
        max_length=512,
    )
    
    model_cache_path: Optional[str] = Field(
        None,
        description="Base path for HuggingFace model cache",
        max_length=512,
    )
    
    hf_token: Optional[str] = Field(
        None,
        description="HuggingFace API token (starts with hf_)",
        max_length=100,
        min_length=3,
    )
    
    selected_gpus: Optional[list[int]] = Field(
        None,
        description="List of GPU IDs to use for training (e.g., [0, 1, 2])"
    )
    gpu_auto_detect: Optional[bool] = Field(
        None,
        description="Automatically detect and use all available GPUs"
    )
    
    @field_validator("model_api_url")
    @classmethod
    def validate_model_api_url(cls, v: Optional[str]) -> Optional[str]:
        """Validate that model_api_url is a valid URL format."""
        if v is None:
            return v
        if not v.startswith(("http://", "https://")):
            raise ValueError("model_api_url must start with http:// or https://")
        return v
    
    @field_validator("hf_token")
    @classmethod
    def validate_hf_token(cls, v: Optional[str]) -> Optional[str]:
        """Validate that hf_token format is correct."""
        if v is None or v == "":
            return None
        if not v.startswith("hf_"):
            raise ValueError("HuggingFace token must start with 'hf_'")
        return v


class TrainingConfigResponse(BaseModel):
    """Schema for training configuration response."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: int = Field(..., description="Config ID")
    
    @model_validator(mode='after')
    def mask_hf_token(self):
        """Mask HuggingFace token for security (show only last 4 chars if present)."""
        if self.hf_token and len(self.hf_token) > 4:
            self.hf_token = f"***{self.hf_token[-4:]}"
        elif self.hf_token:
            self.hf_token = "***"
        return self
    
    # Worker settings
    max_concurrent_workers: int = Field(..., description="Maximum concurrent workers")
    active_workers: int = Field(..., description="Currently active workers")
    
    # Model settings
    default_model: str = Field(..., description="Default model name")
    default_training_type: str = Field(..., description="Default training type")
    
    # Training parameters
    default_batch_size: int = Field(..., description="Default batch size")
    default_learning_rate: float = Field(..., description="Default learning rate")
    default_epochs: int = Field(..., description="Default epochs")
    default_lora_r: int = Field(..., description="Default LoRA rank")
    default_lora_alpha: int = Field(..., description="Default LoRA alpha")
    default_lora_dropout: float = Field(..., description="Default LoRA dropout")
    
    auto_start_workers: bool = Field(..., description="Auto-start workers on startup")
    
    # Model API settings
    model_provider: str = Field(..., description="Model provider type")
    model_api_url: str = Field(..., description="Base URL for the model API server")
    
    # Directory settings
    output_directory_base: Optional[str] = Field(None, description="Base directory for project output")
    model_cache_path: Optional[str] = Field(None, description="Base path for HuggingFace model cache")
    
    # HuggingFace settings
    hf_token: Optional[str] = Field(None, description="HuggingFace API token (masked for security)")
    
    # GPU settings
    selected_gpus: Optional[list[int]] = Field(None, description="List of selected GPU IDs for training")
    gpu_auto_detect: bool = Field(True, description="Auto-detect GPUs or use manual selection")
    
    @field_validator('selected_gpus', mode='before')
    @classmethod
    def parse_selected_gpus(cls, v):
        """Parse selected_gpus from JSON string to list."""
        import json
        if isinstance(v, str):
            try:
                if v:
                    return json.loads(v)
                return None
            except (json.JSONDecodeError, TypeError):
                return None
        return v
    
    # Timestamps
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


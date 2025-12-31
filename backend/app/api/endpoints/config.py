"""
Configuration API endpoints.

Handles training configuration retrieval and updates.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.training_config import (
    TrainingConfigUpdate,
    TrainingConfigResponse,
)
from app.services.training_service import TrainingService
from app.services.gpu_service import GPUService

router = APIRouter()


def get_training_service(db: Session = Depends(get_db)) -> TrainingService:
    """Dependency to get TrainingService instance."""
    return TrainingService(db)


def get_gpu_service() -> GPUService:
    """Dependency to get GPUService instance."""
    return GPUService()


@router.get("/", response_model=TrainingConfigResponse)
async def get_config(
    service: TrainingService = Depends(get_training_service),
) -> TrainingConfigResponse:
    """
    Get the current training configuration.
    """
    config = service.get_config()
    return TrainingConfigResponse.model_validate(config)


@router.patch("/", response_model=TrainingConfigResponse)
async def update_config(
    update_data: TrainingConfigUpdate,
    service: TrainingService = Depends(get_training_service),
) -> TrainingConfigResponse:
    """
    Update the training configuration.
    
    Updates the global training configuration with the provided values.
    Only provided fields will be updated.
    """
    config = service.update_config(update_data)
    return TrainingConfigResponse.model_validate(config)


@router.get("/gpus")
async def get_gpus(
    gpu_service: GPUService = Depends(get_gpu_service),
) -> list[dict]:
    """
    Get list of available GPUs.
    
    Returns information about all detected GPUs including ID, name, and memory.
    """
    return gpu_service.detect_gpus()


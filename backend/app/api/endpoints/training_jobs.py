"""
Training job API endpoints.

Handles training job creation, listing, status updates, cancellation, and model download.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.training_job import TrainingStatus
from app.schemas.training_job import (
    TrainingJobCreate,
    TrainingJobUpdate,
    TrainingJobResponse,
    TrainingJobListResponse,
)
from app.services.training_service import TrainingService

router = APIRouter()


def get_training_service(db: Session = Depends(get_db)) -> TrainingService:
    """Dependency to get TrainingService instance."""
    return TrainingService(db)


@router.post("/", response_model=TrainingJobResponse, status_code=201)
async def create_training_job(
    job_data: TrainingJobCreate,
    service: TrainingService = Depends(get_training_service),
) -> TrainingJobResponse:
    """
    Create a new training job.
    
    Creates a new training job with the specified configuration.
    The job will be queued for processing by available workers.
    """
    try:
        job = service.create_job(job_data)
        return TrainingJobResponse.model_validate(job)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/", response_model=TrainingJobListResponse)
async def list_training_jobs(
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100, description="Items per page")] = 10,
    status: Annotated[TrainingStatus | None, Query(description="Filter by status")] = None,
    service: TrainingService = Depends(get_training_service),
) -> TrainingJobListResponse:
    """
    List all training jobs with pagination and optional status filter.
    """
    result = service.list_jobs(page=page, page_size=page_size, status=status)
    return TrainingJobListResponse(
        items=[TrainingJobResponse.model_validate(j) for j in result["items"]],
        total=result["total"],
        page=result["page"],
        page_size=result["page_size"],
        pages=result["pages"],
    )


@router.get("/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(
    job_id: int,
    service: TrainingService = Depends(get_training_service),
) -> TrainingJobResponse:
    """
    Get a specific training job by ID.
    """
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return TrainingJobResponse.model_validate(job)


@router.patch("/{job_id}", response_model=TrainingJobResponse)
async def update_training_job(
    job_id: int,
    update_data: TrainingJobUpdate,
    service: TrainingService = Depends(get_training_service),
) -> TrainingJobResponse:
    """
    Update a training job's metadata.
    
    Only name and description can be updated. Training parameters
    cannot be modified after creation.
    """
    job = service.update_job(job_id, update_data)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return TrainingJobResponse.model_validate(job)


@router.post("/{job_id}/start", response_model=TrainingJobResponse)
async def start_training_job(
    job_id: int,
    service: TrainingService = Depends(get_training_service),
) -> TrainingJobResponse:
    """
    Manually start a training job.
    
    Submits a pending or queued job to the worker pool for processing.
    If workers are not running, starts them automatically.
    """
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    # Check if job can be started
    if job.status not in [TrainingStatus.PENDING.value, TrainingStatus.QUEUED.value]:
        raise HTTPException(
            status_code=400,
            detail=f"Job cannot be started (status: {job.status})"
        )
    
    # Auto-start workers if not running
    if not service.worker_pool.is_running:
        config = service.get_config()
        service.start_workers(config.max_concurrent_workers)
    
    # Submit job to worker pool
    if job.status == TrainingStatus.PENDING.value or (
        job.status == TrainingStatus.QUEUED.value and 
        job_id not in service.worker_pool.get_all_jobs_in_pool()
    ):
        if service.worker_pool.submit_job(job_id):
            job.status = TrainingStatus.QUEUED.value
            service.db.commit()
        else:
            raise HTTPException(
                status_code=400, 
                detail="Failed to submit job to worker pool"
            )
    
    return TrainingJobResponse.model_validate(job)


@router.post("/{job_id}/cancel", response_model=TrainingJobResponse)
async def cancel_training_job(
    job_id: int,
    service: TrainingService = Depends(get_training_service),
) -> TrainingJobResponse:
    """
    Cancel a training job.
    
    Cancels a pending or running training job. Completed or already
    cancelled jobs cannot be cancelled.
    """
    try:
        job = service.cancel_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")
        return TrainingJobResponse.model_validate(job)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{job_id}/status", response_model=dict)
async def get_training_job_status(
    job_id: int,
    service: TrainingService = Depends(get_training_service),
) -> dict:
    """
    Get the current status of a training job.
    
    Returns a simplified status response for quick polling.
    """
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    return {
        "id": job.id,
        "status": job.status,
        "progress": job.progress,
        "current_epoch": job.current_epoch,
        "current_loss": job.current_loss,
        "error_message": job.error_message,
    }


@router.get("/{job_id}/download")
async def download_training_model(
    job_id: int,
    service: TrainingService = Depends(get_training_service),
) -> FileResponse:
    """
    Download the trained model for a completed job.
    
    For multi-file models, returns a tar.gz archive.
    For single-file models, returns the file directly.
    
    Returns:
        FileResponse with the model file or archive.
    """
    try:
        result = service.get_model_download_path(job_id)
        if not result:
            raise HTTPException(
                status_code=404,
                detail="Model not available for download"
            )
        
        file_path, filename = result
        
        # Determine media type
        if filename.endswith(".tar.gz"):
            media_type = "application/gzip"
        elif filename.endswith(".json"):
            media_type = "application/json"
        elif filename.endswith(".safetensors"):
            media_type = "application/octet-stream"
        elif filename.endswith(".bin"):
            media_type = "application/octet-stream"
        else:
            media_type = "application/octet-stream"
        
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{job_id}/model-info", response_model=dict)
async def get_model_info(
    job_id: int,
    service: TrainingService = Depends(get_training_service),
) -> dict:
    """
    Get information about the trained model files.
    
    Returns file type, size, and list of files for directories.
    """
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    info = service.get_model_file_info(job_id)
    if not info:
        raise HTTPException(
            status_code=404,
            detail="Model information not available"
        )
    
    return info


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
from app.models.project import Project
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
    
    Includes both TrainingJob records and Project records that are in training.
    Projects are converted to TrainingJobResponse format for unified display.
    """
    result = service.list_jobs(page=page, page_size=page_size, status=status)
    
    # Convert items to TrainingJobResponse format
    # Handle both TrainingJob and Project objects
    converted_items = []
    for item in result["items"]:
        if isinstance(item, Project):
            # Convert Project to TrainingJobResponse-like format
            # Use first dataset from project traits, or create a synthetic dataset_id
            dataset_id = 0
            if item.traits and len(item.traits) > 0:
                trait = item.traits[0]
                if trait.datasets and len(trait.datasets) > 0:
                    dataset_id = trait.datasets[0].dataset_id
            
            # Get training config for defaults
            from app.models.training_config import TrainingConfig
            config = service.db.query(TrainingConfig).first()
            if not config:
                # Create default config if it doesn't exist
                config = TrainingConfig(
                    default_batch_size=4,
                    default_learning_rate=2e-4,
                    default_epochs=3,
                    default_lora_r=16,
                    default_lora_alpha=32,
                    default_lora_dropout=0.05,
                )
                service.db.add(config)
                service.db.commit()
            
            # Create a dict that matches TrainingJobResponse structure
            job_dict = {
                "id": item.id,
                "name": item.name,
                "description": item.description,
                "status": item.status,
                "training_type": item.training_type,
                "model_name": item.base_model,
                "dataset_id": dataset_id,
                "batch_size": config.default_batch_size,
                "learning_rate": config.default_learning_rate,
                "epochs": config.default_epochs,
                "lora_r": config.default_lora_r,
                "lora_alpha": config.default_lora_alpha,
                "lora_dropout": config.default_lora_dropout,
                "progress": item.progress,
                "current_epoch": item.current_epoch,
                "current_loss": item.current_loss,
                "error_message": item.error_message,
                "log": item.log,
                "model_path": item.model_path,
                "worker_id": item.worker_id,
                "started_at": item.started_at,
                "completed_at": item.completed_at,
                "created_at": item.created_at,
                "updated_at": item.updated_at,
            }
            converted_items.append(TrainingJobResponse.model_validate(job_dict))
        else:
            # It's a TrainingJob, convert normally
            converted_items.append(TrainingJobResponse.model_validate(item))
    
    return TrainingJobListResponse(
        items=converted_items,
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
    
    Handles both TrainingJob IDs and Project IDs. Projects are converted
    to TrainingJobResponse format for unified display.
    """
    # First try to get as TrainingJob
    job = service.get_job(job_id)
    
    if job:
        return TrainingJobResponse.model_validate(job)
    
    # If not found, try to get as Project
    from app.models.project import Project, ProjectStatus
    from app.models.training_config import TrainingConfig
    
    project = service.db.query(Project).filter(Project.id == job_id).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    # Only return projects that are in training status
    training_statuses = [
        ProjectStatus.PENDING.value,
        ProjectStatus.RUNNING.value,
        ProjectStatus.COMPLETED.value,
        ProjectStatus.FAILED.value,
        ProjectStatus.CANCELLED.value,
    ]
    if project.status not in training_statuses:
        raise HTTPException(
            status_code=404,
            detail="Project is not in a training status"
        )
    
    # Convert Project to TrainingJobResponse format
    # Use first dataset from project traits, or create a synthetic dataset_id
    dataset_id = 0
    if project.traits and len(project.traits) > 0:
        trait = project.traits[0]
        if trait.datasets and len(trait.datasets) > 0:
            dataset_id = trait.datasets[0].dataset_id
    
    # Get training config for defaults
    config = service.db.query(TrainingConfig).first()
    if not config:
        # Create default config if it doesn't exist
        config = TrainingConfig(
            default_batch_size=4,
            default_learning_rate=2e-4,
            default_epochs=3,
            default_lora_r=16,
            default_lora_alpha=32,
            default_lora_dropout=0.05,
        )
        service.db.add(config)
        service.db.commit()
    
    # Create a dict that matches TrainingJobResponse structure
    job_dict = {
        "id": project.id,
        "name": project.name,
        "description": project.description,
        "status": project.status,
        "training_type": project.training_type,
        "model_name": project.base_model,
        "dataset_id": dataset_id,
        "batch_size": config.default_batch_size,
        "learning_rate": config.default_learning_rate,
        "epochs": config.default_epochs,
        "lora_r": config.default_lora_r,
        "lora_alpha": config.default_lora_alpha,
        "lora_dropout": config.default_lora_dropout,
        "progress": project.progress,
        "current_epoch": project.current_epoch,
        "current_loss": project.current_loss,
        "error_message": project.error_message,
        "log": project.log,
        "model_path": project.model_path,
        "worker_id": project.worker_id,
        "started_at": project.started_at,
        "completed_at": project.completed_at,
        "created_at": project.created_at,
        "updated_at": project.updated_at,
    }
    
    return TrainingJobResponse.model_validate(job_dict)


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
    db: Session = Depends(get_db),
) -> TrainingJobResponse:
    """
    Manually start a training job or project.
    
    Handles both TrainingJob IDs and Project IDs (since Projects appear in jobs list).
    Submits a pending or queued job/project to the worker pool for processing.
    If workers are not running, starts them automatically.
    """
    # First try to get as TrainingJob
    job = service.get_job(job_id)
    
    # If not found, try as Project
    if not job:
        from app.models.project import Project, ProjectStatus
        project = db.query(Project).filter(Project.id == job_id).first()
        if project:
            # Check if project can be started
            if project.status not in [ProjectStatus.PENDING.value]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Project cannot be started (status: {project.status})"
                )
            
            # Auto-start workers if not running
            if not service.worker_pool.is_running:
                config = service.get_config()
                service.start_workers(config.max_concurrent_workers)
            
            # Queue project for training
            from app.services.training_service import TrainingService
            training_service = TrainingService(db=db)
            training_service.worker_pool.queue_project(project.id)
            
            # Refresh project
            db.refresh(project)
            
            # Convert Project to TrainingJobResponse format (same as get_training_job endpoint)
            dataset_id = 0
            if project.traits and len(project.traits) > 0:
                trait = project.traits[0]
                if trait.datasets and len(trait.datasets) > 0:
                    dataset_id = trait.datasets[0].dataset_id
            
            from app.models.training_config import TrainingConfig
            config = db.query(TrainingConfig).first()
            if not config:
                config = TrainingConfig(
                    default_batch_size=4,
                    default_learning_rate=2e-4,
                    default_epochs=3,
                    default_lora_r=16,
                    default_lora_alpha=32,
                    default_lora_dropout=0.05,
                )
                db.add(config)
                db.commit()
            
            job_dict = {
                "id": project.id,
                "name": project.name,
                "description": project.description,
                "status": project.status,
                "training_type": project.training_type,
                "model_name": project.base_model,
                "dataset_id": dataset_id,
                "batch_size": config.default_batch_size,
                "learning_rate": config.default_learning_rate,
                "epochs": config.default_epochs,
                "lora_r": config.default_lora_r,
                "lora_alpha": config.default_lora_alpha,
                "lora_dropout": config.default_lora_dropout,
                "progress": project.progress,
                "current_epoch": project.current_epoch,
                "current_loss": project.current_loss,
                "error_message": project.error_message,
                "log": project.log,
                "model_path": project.model_path,
                "worker_id": project.worker_id,
                "started_at": project.started_at,
                "completed_at": project.completed_at,
                "created_at": project.created_at,
                "updated_at": project.updated_at,
            }
            return TrainingJobResponse.model_validate(job_dict)
        else:
            raise HTTPException(status_code=404, detail="Training job or project not found")
    
    # It's a TrainingJob - handle normally
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
    db: Session = Depends(get_db),
) -> TrainingJobResponse:
    """
    Cancel a training job or project.
    
    Handles both TrainingJob IDs and Project IDs (since Projects appear in jobs list).
    Cancels a pending or running training job/project. Completed or already
    cancelled items cannot be cancelled.
    """
    try:
        item = service.cancel_job(job_id)
        if not item:
            raise HTTPException(status_code=404, detail="Training job or project not found")
        
        # If it's a Project, convert to TrainingJobResponse format
        from app.models.project import Project, ProjectStatus
        from app.models.training_config import TrainingConfig
        
        if isinstance(item, Project):
            # Convert Project to TrainingJobResponse format (same as get_training_job endpoint)
            dataset_id = 0
            if item.traits and len(item.traits) > 0:
                trait = item.traits[0]
                if trait.datasets and len(trait.datasets) > 0:
                    dataset_id = trait.datasets[0].dataset_id
            
            # Get training config for defaults
            config = db.query(TrainingConfig).first()
            if not config:
                config = TrainingConfig(
                    default_batch_size=4,
                    default_learning_rate=2e-4,
                    default_epochs=3,
                    default_lora_r=16,
                    default_lora_alpha=32,
                    default_lora_dropout=0.05,
                )
                db.add(config)
                db.commit()
            
            job_dict = {
                "id": item.id,
                "name": item.name,
                "description": item.description,
                "status": item.status,
                "training_type": item.training_type,
                "model_name": item.base_model,
                "dataset_id": dataset_id,
                "batch_size": config.default_batch_size,
                "learning_rate": config.default_learning_rate,
                "epochs": config.default_epochs,
                "lora_r": config.default_lora_r,
                "lora_alpha": config.default_lora_alpha,
                "lora_dropout": config.default_lora_dropout,
                "progress": item.progress,
                "current_epoch": item.current_epoch,
                "current_loss": item.current_loss,
                "error_message": item.error_message,
                "log": item.log,
                "model_path": item.model_path,
                "worker_id": item.worker_id,
                "started_at": item.started_at,
                "completed_at": item.completed_at,
                "created_at": item.created_at,
                "updated_at": item.updated_at,
            }
            return TrainingJobResponse.model_validate(job_dict)
        else:
            # It's a TrainingJob - return directly
            return TrainingJobResponse.model_validate(item)
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


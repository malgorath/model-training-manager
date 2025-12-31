"""
Project API endpoints.

Handles project creation, listing, updates, validation, and training.
"""

from typing import Annotated, Optional, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.project import (
    ProjectCreate,
    ProjectUpdate,
    ProjectResponse,
    ProjectListResponse,
    OutputDirectoryValidationRequest,
    OutputDirectoryValidationResponse,
    ModelValidationRequest,
    ModelValidationResponse,
)
from app.services.project_service import ProjectService, ProjectValidationError, DatasetAllocationError
from app.models.project import ProjectStatus
from app.services.output_directory_service import OutputDirectoryService, DirectoryValidationError
from app.services.model_resolution_service import ModelResolutionService, ModelNotFoundError, ModelFormatError
from app.services.model_validation_service import ModelValidationService

router = APIRouter()


def get_project_service(db: Session = Depends(get_db)) -> ProjectService:
    """Dependency to get ProjectService instance."""
    return ProjectService(db=db)


def get_output_directory_service() -> OutputDirectoryService:
    """Dependency to get OutputDirectoryService instance."""
    return OutputDirectoryService()


def get_model_resolution_service(db: Session = Depends(get_db)) -> ModelResolutionService:
    """Dependency to get ModelResolutionService instance with paths from TrainingConfig."""
    from app.models.training_config import TrainingConfig
    
    from app.core.config import settings
    
    config = db.query(TrainingConfig).first()
    local_models_path = None
    cache_path = None
    
    if config:
        if config.model_cache_path:
            cache_path = config.model_cache_path
        # Use default model path from settings
        local_models_path = str(settings.get_model_path())
    
    return ModelResolutionService(
        cache_base_path=cache_path,
        local_models_path=local_models_path,
    )


def get_model_validation_service() -> ModelValidationService:
    """Dependency to get ModelValidationService instance."""
    return ModelValidationService()


@router.post("/", response_model=ProjectResponse, status_code=201)
async def create_project(
    project_data: ProjectCreate,
    service: ProjectService = Depends(get_project_service),
    output_service: OutputDirectoryService = Depends(get_output_directory_service),
) -> ProjectResponse:
    """
    Create a new training project.
    
    Creates a project with traits and dataset allocations. Validates:
    - Output directory is writable
    - Trait constraints (Reasoning/Coding = 1 dataset, General/Tools = 1+ datasets)
    - No duplicate datasets in project
    """
    # Validate output directory
    try:
        output_service.validate_directory(project_data.output_directory, create_if_missing=True)
    except DirectoryValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Create project
    try:
        project_dict = project_data.model_dump()
        project = service.create_project(project_dict)
        return ProjectResponse.model_validate(_project_to_dict(project))
    except (ProjectValidationError, DatasetAllocationError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/", response_model=ProjectListResponse)
async def list_projects(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page"),
    service: ProjectService = Depends(get_project_service),
) -> ProjectListResponse:
    """
    List all projects.
    
    Returns paginated list of all training projects.
    """
    skip = (page - 1) * page_size
    projects_list = service.list_projects(skip=skip, limit=page_size)
    
    # Get total count
    from app.models.project import Project
    total = service.db.query(Project).count()
    
    items = [ProjectResponse.model_validate(_project_to_dict(p)) for p in projects_list]
    
    return ProjectListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: int,
    service: ProjectService = Depends(get_project_service),
) -> ProjectResponse:
    """
    Get a project by ID.
    
    Returns full project details including traits and dataset allocations.
    """
    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    
    return ProjectResponse.model_validate(_project_to_dict(project))


@router.delete("/{project_id}", status_code=204)
async def delete_project(
    project_id: int,
    service: ProjectService = Depends(get_project_service),
) -> None:
    """
    Delete a project.
    
    Projects that are currently running cannot be deleted.
    Related traits and dataset allocations will be automatically deleted (cascade delete).
    """
    try:
        success = service.delete_project(project_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    except ProjectValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: int,
    project_update: ProjectUpdate,
    service: ProjectService = Depends(get_project_service),
) -> ProjectResponse:
    """
    Update a project.
    
    Updates project metadata. Cannot update traits or datasets after creation.
    """
    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    
    update_dict = project_update.model_dump(exclude_unset=True)
    for key, value in update_dict.items():
        setattr(project, key, value)
    
    service.db.commit()
    service.db.refresh(project)
    
    return ProjectResponse.model_validate(_project_to_dict(project))


@router.post("/validate-output-dir", response_model=OutputDirectoryValidationResponse)
async def validate_output_directory(
    request: OutputDirectoryValidationRequest,
    service: OutputDirectoryService = Depends(get_output_directory_service),
) -> OutputDirectoryValidationResponse:
    """
    Validate an output directory.
    
    Checks if the directory exists, is writable, and has proper permissions.
    """
    try:
        result = service.validate_directory(request.output_directory, create_if_missing=False)
        return OutputDirectoryValidationResponse(**result)
    except DirectoryValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/validate-model", response_model=ModelValidationResponse)
async def validate_model_availability(
    request: ModelValidationRequest,
    service: ModelResolutionService = Depends(get_model_resolution_service),
) -> ModelValidationResponse:
    """
    Check if a model is available locally.
    
    Validates model exists in HuggingFace cache or configured local paths.
    """
    try:
        available = service.is_model_available(request.model_name)
        if available:
            path = service.resolve_model_path(request.model_name)
            return ModelValidationResponse(
                available=True,
                model_name=request.model_name,
                path=path,
            )
        else:
            return ModelValidationResponse(
                available=False,
                model_name=request.model_name,
                errors=[f"Model '{request.model_name}' not found in local cache or configured paths"],
            )
    except (ModelNotFoundError, ModelFormatError) as e:
        return ModelValidationResponse(
            available=False,
            model_name=request.model_name,
            errors=[str(e)],
        )


@router.post("/{project_id}/start")
async def start_project_training(
    project_id: int,
    service: ProjectService = Depends(get_project_service),
    model_service: ModelResolutionService = Depends(get_model_resolution_service),
) -> ProjectResponse:
    """
    Start training for a project.
    
    Validates model availability and queues the project for training.
    """
    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    
    # Validate model availability
    try:
        if not model_service.is_model_available(project.base_model):
            raise HTTPException(
                status_code=400,
                detail=f"Model '{project.base_model}' is not available or not suitable for training. "
                       f"Please ensure it's downloaded in HuggingFace format (with config.json and tokenizer files). "
                       f"Note: GGUF models are inference-only and cannot be used for training.",
            )
    except (ModelNotFoundError, ModelFormatError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{project.base_model}' validation failed: {str(e)}. "
                   f"GGUF models (quantized inference models) cannot be used for training. "
                   f"Please download a HuggingFace format model instead."
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model validation failed: {str(e)}")
    
    # Update project status
    project.status = "pending"  # Will be picked up by worker
    service.db.commit()
    service.db.refresh(project)
    
    # Queue project for training with worker pool
    from app.services.training_service import TrainingService
    training_service = TrainingService(db=service.db)
    training_service.worker_pool.queue_project(project.id)
    
    return ProjectResponse.model_validate(_project_to_dict(project))


@router.post("/{project_id}/cancel")
async def cancel_project_training(
    project_id: int,
    service: ProjectService = Depends(get_project_service),
) -> ProjectResponse:
    """
    Cancel training for a running or pending project.
    
    Cancels the project and stops any running training.
    """
    try:
        project = service.cancel_project(project_id)
        return ProjectResponse.model_validate(_project_to_dict(project))
    except ProjectValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{project_id}/retry")
async def retry_project_training(
    project_id: int,
    service: ProjectService = Depends(get_project_service),
    model_service: ModelResolutionService = Depends(get_model_resolution_service),
) -> ProjectResponse:
    """
    Retry training for a failed project.
    
    Resets the project status to pending, clears error messages, and queues it for training again.
    """
    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    
    if project.status != ProjectStatus.FAILED.value:
        raise HTTPException(
            status_code=400,
            detail=f"Project {project_id} is not in failed status. Only failed projects can be retried."
        )
    
    # Validate model availability
    try:
        if not model_service.is_model_available(project.base_model):
            raise HTTPException(
                status_code=400,
                detail=f"Model '{project.base_model}' is not available or not suitable for training. "
                       f"Please ensure it's downloaded in HuggingFace format (with config.json and tokenizer files). "
                       f"Note: GGUF models are inference-only and cannot be used for training.",
            )
    except (ModelNotFoundError, ModelFormatError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{project.base_model}' validation failed: {str(e)}. "
                   f"GGUF models (quantized inference models) cannot be used for training. "
                   f"Please download a HuggingFace format model instead."
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model validation failed: {str(e)}")
    
    # Reset project for retry
    project.status = ProjectStatus.PENDING.value
    project.progress = 0.0
    project.current_epoch = 0
    project.current_loss = None
    project.error_message = None
    project.worker_id = None
    project.started_at = None
    project.completed_at = None
    # Keep the log for reference, but could clear it if desired
    # project.log = None
    
    service.db.commit()
    service.db.refresh(project)
    
    # Queue project for training with worker pool
    from app.services.training_service import TrainingService
    training_service = TrainingService(db=service.db)
    training_service.worker_pool.queue_project(project.id)
    
    return ProjectResponse.model_validate(_project_to_dict(project))


@router.post("/{project_id}/validate")
async def validate_trained_model(
    project_id: int,
    validation_service: ModelValidationService = Depends(get_model_validation_service),
    service: ProjectService = Depends(get_project_service),
) -> dict:
    """
    Validate a trained model after training completes.
    
    Performs complete validation including file checks and loading tests.
    """
    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    
    if not project.model_path:
        raise HTTPException(status_code=400, detail="Project has no model path. Training may not be complete.")
    
    try:
        result = validation_service.validate_model_complete(project.model_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Validation failed: {str(e)}")


@router.get("/models/available", tags=["Models"])
async def list_available_models(
    model_service: ModelResolutionService = Depends(get_model_resolution_service),
    db: Session = Depends(get_db),
) -> list[str]:
    """
    List all available models suitable for training.
    
    Returns ONLY models that have been downloaded and registered in the database.
    This ensures consistency with the models shown in the Models table.
    """
    from app.models.downloaded_model import DownloadedModel
    
    # Only return models from the database (downloaded models)
    downloaded_models = db.query(DownloadedModel.model_id).all()
    available = []
    
    for (model_id,) in downloaded_models:
        # Check if the model is valid for training using the service method
        # This filters out GGUF-only models which don't have config.json/tokenizer files
        if model_service.is_model_available(model_id):
            available.append(model_id)
    
    return sorted(available)


@router.get("/models/{model_name:path}/types", tags=["Models"])
async def get_model_types(
    model_name: str,
    model_service: ModelResolutionService = Depends(get_model_resolution_service),
) -> dict[str, Any]:
    """
    Get available model types for a specific model.
    
    Reads the model's config.json to determine the model_type and returns
    available model types from CONFIG_MAPPING that match.
    
    Args:
        model_name: Model identifier to check
        
    Returns:
        Dictionary with:
        - model_type: The detected model_type from config.json
        - available_types: List of available model types from CONFIG_MAPPING
        - recommended: The recommended model_type to use
    """
    from transformers import AutoConfig, CONFIG_MAPPING
    import json
    from pathlib import Path
    
    try:
        # Check if model is available
        if not model_service.is_model_available(model_name):
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found. Please ensure it's downloaded."
            )
        
        # Resolve model path
        model_path = model_service.resolve_model_path(model_name)
        config_path = Path(model_path) / "config.json"
        
        if not config_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Model config.json not found at {config_path}"
            )
        
        # Read model_type from config.json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
            detected_model_type = config_dict.get('model_type')
        
        if not detected_model_type:
            raise HTTPException(
                status_code=400,
                detail=f"Model config.json missing 'model_type' field"
            )
        
        # Load the actual config to get the Config class
        config = AutoConfig.from_pretrained(
            str(model_path),
            trust_remote_code=True,
        )
        config_class = type(config)
        
        # Find all model types in CONFIG_MAPPING that map to the same Config class
        # These are the compatible model types for this model
        compatible_types = [
            model_type 
            for model_type, mapped_class in CONFIG_MAPPING.items()
            if mapped_class == config_class
        ]
        
        # Sort for consistent ordering
        available_types = sorted(compatible_types) if compatible_types else [detected_model_type]
        
        # Check if detected type is in the compatible types
        recommended = detected_model_type if detected_model_type in available_types else (available_types[0] if available_types else None)
        
        return {
            "model_type": detected_model_type,
            "available_types": available_types,
            "recommended": recommended,
            "model_name": model_name,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model types: {str(e)}"
        )


def _project_to_dict(project) -> dict:
    """Convert project ORM object to dictionary for Pydantic validation."""
    traits_data = []
    for trait in project.traits:
        datasets_data = []
        for trait_dataset in trait.datasets:
            datasets_data.append({
                "dataset_id": trait_dataset.dataset_id,
                "percentage": trait_dataset.percentage,
            })
        traits_data.append({
            "id": trait.id,
            "trait_type": trait.trait_type,
            "datasets": datasets_data,
        })
    
    return {
        "id": project.id,
        "name": project.name,
        "description": project.description,
        "base_model": project.base_model,
        "model_type": project.model_type,
        "training_type": project.training_type,
        "max_rows": project.max_rows,
        "output_directory": project.output_directory,
        "status": project.status,
        "progress": project.progress,
        "current_epoch": project.current_epoch,
        "current_loss": project.current_loss,
        "error_message": project.error_message,
        "model_path": project.model_path,
        "worker_id": project.worker_id,
        "started_at": project.started_at,
        "completed_at": project.completed_at,
        "created_at": project.created_at,
        "updated_at": project.updated_at,
        "traits": traits_data,
    }

"""
Project API endpoints.

Handles project creation, listing, updates, validation, and training.
"""

from typing import Annotated, Optional

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


def get_model_resolution_service() -> ModelResolutionService:
    """Dependency to get ModelResolutionService instance."""
    return ModelResolutionService()


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
    - Dataset percentages sum to 100%
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
                detail=f"Model '{project.base_model}' not available. Please ensure it's downloaded.",
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
) -> list[str]:
    """
    List all available models in local cache.
    
    Returns list of model identifiers found in HuggingFace cache.
    """
    return model_service.list_available_models()


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

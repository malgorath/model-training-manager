"""
Models API endpoints.

Handles HuggingFace model search, download, and local model management.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.training_config import TrainingConfig
from app.schemas.model import (
    ModelSearchResponse,
    HuggingFaceModelInfo,
    ModelDownloadRequest,
    ModelDownloadResponse,
    LocalModelResponse,
)
from app.services.huggingface_service import HuggingFaceService
from app.services.downloaded_model_service import DownloadedModelService

router = APIRouter()


def get_hf_service(db: Session = Depends(get_db)) -> HuggingFaceService:
    """Get HuggingFaceService instance with token from TrainingConfig."""
    config = db.query(TrainingConfig).first()
    token = config.hf_token if config else None
    return HuggingFaceService(hf_token=token)


def get_downloaded_model_service(db: Session = Depends(get_db)) -> DownloadedModelService:
    """Get DownloadedModelService instance."""
    return DownloadedModelService(db)


@router.get("/search", response_model=ModelSearchResponse)
async def search_models(
    query: str = Query(..., description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Results offset"),
    hf_service: HuggingFaceService = Depends(get_hf_service),
) -> ModelSearchResponse:
    """
    Search for models on HuggingFace Hub.
    
    Returns a list of models matching the search query.
    """
    try:
        results = await hf_service.search_models(query=query, limit=limit, offset=offset)
        return ModelSearchResponse(**results)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/download", response_model=ModelDownloadResponse)
async def download_model(
    request: ModelDownloadRequest,
    hf_service: HuggingFaceService = Depends(get_hf_service),
    model_service: DownloadedModelService = Depends(get_downloaded_model_service),
    db: Session = Depends(get_db),
) -> ModelDownloadResponse:
    """
    Download a model from HuggingFace Hub.
    
    Downloads the model files and creates a database record to track it.
    """
    try:
        # Check if token is available for private models
        config = db.query(TrainingConfig).first()
        if not config or not config.hf_token:
            # Try download anyway - some models are public
            pass
        
        # Download model
        downloaded_model = model_service.download_model(
            model_id=request.model_id,
            hf_service=hf_service,
        )
        
        # Try to get additional model info from API and update
        try:
            model_info = await hf_service.get_model_info(request.model_id)
            model_service.update_model_info(
                model_id=request.model_id,
                description=model_info.get("description"),
                tags=model_info.get("tags"),
                model_type=model_info.get("model_type"),
                is_private=model_info.get("private", False),
            )
        except Exception as e:
            # Non-fatal - model is downloaded, just metadata update failed
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to update model metadata: {e}")
        
        return ModelDownloadResponse(
            id=downloaded_model.id,
            model_id=downloaded_model.model_id,
            name=downloaded_model.name,
            author=downloaded_model.author,
            local_path=downloaded_model.local_path,
            file_size=downloaded_model.file_size,
            message=f"Successfully downloaded model {request.model_id}",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@router.get("/", response_model=list[LocalModelResponse])
async def list_local_models(
    model_service: DownloadedModelService = Depends(get_downloaded_model_service),
) -> list[LocalModelResponse]:
    """
    List all locally downloaded models.
    
    Returns a list of models that have been downloaded and stored locally.
    """
    models = model_service.list_local_models()
    return [LocalModelResponse.model_validate(model) for model in models]


@router.get("/local/{model_id:path}", response_model=LocalModelResponse)
async def get_local_model(
    model_id: str,
    model_service: DownloadedModelService = Depends(get_downloaded_model_service),
) -> LocalModelResponse:
    """
    Get information about a locally downloaded model.
    
    Args:
        model_id: HuggingFace model ID.
    """
    model = model_service.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found locally")
    
    return LocalModelResponse.model_validate(model)


@router.delete("/local/{model_id:path}")
async def delete_local_model(
    model_id: str,
    model_service: DownloadedModelService = Depends(get_downloaded_model_service),
) -> dict:
    """
    Delete a locally downloaded model.
    
    Removes the model files and database record.
    
    Args:
        model_id: HuggingFace model ID.
    """
    try:
        model_service.delete_model(model_id)
        return {"message": f"Model {model_id} deleted successfully"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


@router.post("/scan", status_code=200)
async def scan_models(
    model_service: DownloadedModelService = Depends(get_downloaded_model_service),
) -> dict:
    """
    Scan the model directory structure and auto-add valid models to the database.
    
    Scans ./data/models/{org}/{model_name}/ directories for valid model directories
    (containing config.json and tokenizer files), and adds them to the database
    if they don't already exist.
    
    Returns a summary of the scan operation.
    """
    result = model_service.scan_models()
    return result


@router.get("/{model_id:path}", response_model=HuggingFaceModelInfo)
async def get_model_info(
    model_id: str,
    hf_service: HuggingFaceService = Depends(get_hf_service),
) -> HuggingFaceModelInfo:
    """
    Get detailed information about a model from HuggingFace Hub.
    
    Args:
        model_id: HuggingFace model ID (e.g., "meta-llama/Llama-3.2-3B-Instruct").
    """
    # Validate model_id is not empty
    if not model_id or not model_id.strip():
        raise HTTPException(status_code=400, detail="Model ID cannot be empty")
    
    try:
        info = await hf_service.get_model_info(model_id)
        return HuggingFaceModelInfo(**info)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

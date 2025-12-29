"""
Worker management API endpoints.

Handles worker pool status and control operations.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.worker import (
    WorkerPoolStatus,
    WorkerCommand,
)
from app.services.training_service import TrainingService

router = APIRouter()


def get_training_service(db: Session = Depends(get_db)) -> TrainingService:
    """Dependency to get TrainingService instance."""
    return TrainingService(db)


@router.get("/", response_model=WorkerPoolStatus)
async def get_worker_status(
    service: TrainingService = Depends(get_training_service),
) -> WorkerPoolStatus:
    """
    Get the current status of the worker pool.
    
    Returns information about all workers including their status,
    current jobs, and statistics.
    """
    return service.get_worker_pool_status()


@router.post("/", response_model=WorkerPoolStatus)
async def control_workers(
    command: WorkerCommand,
    service: TrainingService = Depends(get_training_service),
) -> WorkerPoolStatus:
    """
    Control the worker pool.
    
    Supports the following actions:
    - start: Start new workers (specify worker_count)
    - stop: Stop all workers
    - restart: Restart all workers
    """
    try:
        if command.action == "start":
            service.start_workers(command.worker_count or 1)
        elif command.action == "stop":
            service.stop_workers()
        elif command.action == "restart":
            service.restart_workers()
        else:
            raise HTTPException(status_code=400, detail="Invalid action")
        
        return service.get_worker_pool_status()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


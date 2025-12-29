"""
Worker Pydantic schemas for request/response validation.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class WorkerStatus(str, Enum):
    """Worker status enumeration."""
    
    IDLE = "idle"
    BUSY = "busy"
    STOPPED = "stopped"
    ERROR = "error"


class WorkerInfo(BaseModel):
    """Schema for individual worker information."""
    
    id: str = Field(..., description="Worker ID")
    status: WorkerStatus = Field(..., description="Worker status")
    current_job_id: Optional[int] = Field(None, description="Current job ID if busy")
    jobs_completed: int = Field(0, description="Total jobs completed")
    started_at: datetime = Field(..., description="Worker start timestamp")
    last_activity: Optional[datetime] = Field(None, description="Last activity timestamp")


class WorkerPoolStatus(BaseModel):
    """Schema for worker pool status response."""
    
    total_workers: int = Field(..., description="Total number of workers")
    active_workers: int = Field(..., description="Number of active workers")
    idle_workers: int = Field(..., description="Number of idle workers")
    busy_workers: int = Field(..., description="Number of busy workers")
    max_workers: int = Field(..., description="Maximum allowed workers")
    workers: list[WorkerInfo] = Field(..., description="List of worker details")
    jobs_in_queue: int = Field(..., description="Number of jobs in queue")


class WorkerCommand(BaseModel):
    """Schema for worker control commands."""
    
    action: str = Field(
        ...,
        pattern="^(start|stop|restart)$",
        description="Command action: start, stop, or restart"
    )
    worker_count: Optional[int] = Field(
        None,
        ge=1,
        le=32,
        description="Number of workers to start (for start action)"
    )


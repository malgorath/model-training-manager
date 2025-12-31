"""
Training service for managing training jobs and workers.

Handles job creation, worker orchestration, and configuration management.
"""

import logging
import tarfile
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import settings

logger = logging.getLogger(__name__)
from app.models.dataset import Dataset
from app.models.project import Project
from app.models.training_job import TrainingJob, TrainingStatus
from app.models.training_config import TrainingConfig
from app.schemas.training_job import TrainingJobCreate, TrainingJobUpdate, TrainingJobProgress
from app.schemas.training_config import TrainingConfigUpdate
from app.schemas.worker import WorkerPoolStatus, WorkerInfo, WorkerStatus
from app.workers.worker_pool import WorkerPool


class TrainingService:
    """
    Service for managing training jobs and workers.
    
    Handles job lifecycle, worker orchestration, and configuration
    management for the training system.
    
    Attributes:
        db: SQLAlchemy database session.
        worker_pool: Worker pool manager instance.
    """
    
    # Shared worker pool instance
    _worker_pool: WorkerPool | None = None
    
    def __init__(self, db: Session):
        """
        Initialize the training service.
        
        Args:
            db: SQLAlchemy database session.
        """
        self.db = db
        
        # Initialize worker pool if not exists
        if TrainingService._worker_pool is None:
            TrainingService._worker_pool = WorkerPool(
                max_workers=settings.max_workers,
                db_session_factory=self._get_db_session,
            )
    
    def _get_db_session(self) -> Session:
        """Get a new database session for workers."""
        from app.core.database import SessionLocal
        return SessionLocal()
    
    @property
    def worker_pool(self) -> WorkerPool:
        """Get the worker pool instance."""
        return TrainingService._worker_pool
    
    def create_job(self, job_data: TrainingJobCreate) -> TrainingJob:
        """
        Create a new training job.
        
        Args:
            job_data: Training job creation data.
            
        Returns:
            Created TrainingJob object.
            
        Raises:
            ValueError: If dataset not found or validation fails.
        """
        # Verify dataset exists
        dataset = self.db.query(Dataset).filter(Dataset.id == job_data.dataset_id).first()
        if not dataset:
            raise ValueError(f"Dataset with ID {job_data.dataset_id} not found")
        
        # Get default config
        config = self.get_config()
        
        # Create job with defaults from config
        job = TrainingJob(
            name=job_data.name,
            description=job_data.description,
            training_type=job_data.training_type.value,
            model_name=job_data.model_name,
            dataset_id=job_data.dataset_id,
            batch_size=job_data.batch_size or config.default_batch_size,
            learning_rate=job_data.learning_rate or config.default_learning_rate,
            epochs=job_data.epochs or config.default_epochs,
            lora_r=job_data.lora_r or config.default_lora_r,
            lora_alpha=job_data.lora_alpha or config.default_lora_alpha,
            lora_dropout=job_data.lora_dropout or config.default_lora_dropout,
            status=TrainingStatus.PENDING.value,
        )
        
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        
        # Queue job for processing
        if self.worker_pool.is_running:
            self.worker_pool.submit_job(job.id)
            job.status = TrainingStatus.QUEUED.value
            self.db.commit()
        
        return job
    
    def get_job(self, job_id: int) -> TrainingJob | None:
        """
        Get a training job by ID.
        
        Args:
            job_id: Training job ID.
            
        Returns:
            TrainingJob object or None if not found.
        """
        return self.db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    
    def list_jobs(
        self,
        page: int = 1,
        page_size: int = 10,
        status: TrainingStatus | None = None,
    ) -> dict[str, Any]:
        """
        List training jobs with pagination and optional filtering.
        
        Includes both TrainingJob records and Project records that are in training.
        Projects are converted to a TrainingJob-like format for unified display.
        
        Args:
            page: Page number (1-indexed).
            page_size: Number of items per page.
            status: Optional status filter.
            
        Returns:
            Dictionary with items, total, page, page_size, and pages.
            Items can be either TrainingJob or Project objects.
        """
        from app.models.project import Project, ProjectStatus
        
        # Get TrainingJob records
        job_query = self.db.query(TrainingJob).order_by(TrainingJob.created_at.desc())
        if status:
            job_query = job_query.filter(TrainingJob.status == status.value)
        jobs = job_query.all()
        
        # Get Project records that are in training status
        project_query = self.db.query(Project).order_by(Project.created_at.desc())
        # Only include projects that are actively training (not just created)
        training_statuses = [
            ProjectStatus.PENDING.value,
            ProjectStatus.RUNNING.value,
            ProjectStatus.COMPLETED.value,
            ProjectStatus.FAILED.value,
            ProjectStatus.CANCELLED.value,
        ]
        project_query = project_query.filter(Project.status.in_(training_statuses))
        
        if status:
            # Map TrainingStatus to ProjectStatus
            status_map = {
                TrainingStatus.PENDING: ProjectStatus.PENDING.value,
                TrainingStatus.QUEUED: ProjectStatus.PENDING.value,
                TrainingStatus.RUNNING: ProjectStatus.RUNNING.value,
                TrainingStatus.COMPLETED: ProjectStatus.COMPLETED.value,
                TrainingStatus.FAILED: ProjectStatus.FAILED.value,
                TrainingStatus.CANCELLED: ProjectStatus.CANCELLED.value,
            }
            project_status = status_map.get(status)
            if project_status:
                project_query = project_query.filter(Project.status == project_status)
        
        projects = project_query.all()
        
        # Combine and sort by created_at
        all_items = list(jobs) + list(projects)
        all_items.sort(key=lambda x: x.created_at, reverse=True)
        
        # Apply pagination
        total = len(all_items)
        pages = ceil(total / page_size) if total > 0 else 1
        offset = (page - 1) * page_size
        paginated_items = all_items[offset:offset + page_size]
        
        return {
            "items": paginated_items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": pages,
        }
    
    def update_job(
        self,
        job_id: int,
        update_data: TrainingJobUpdate,
    ) -> TrainingJob | None:
        """
        Update a training job's metadata.
        
        Args:
            job_id: Training job ID.
            update_data: Fields to update.
            
        Returns:
            Updated TrainingJob object or None if not found.
        """
        job = self.get_job(job_id)
        if not job:
            return None
        
        update_dict = update_data.model_dump(exclude_unset=True)
        for field, value in update_dict.items():
            setattr(job, field, value)
        
        self.db.commit()
        self.db.refresh(job)
        
        return job
    
    def update_job_progress(
        self,
        job_id: int,
        progress_data: TrainingJobProgress,
    ) -> TrainingJob | None:
        """
        Update a training job's progress.
        
        Args:
            job_id: Training job ID.
            progress_data: Progress update data.
            
        Returns:
            Updated TrainingJob object or None if not found.
        """
        job = self.get_job(job_id)
        if not job:
            return None
        
        job.progress = progress_data.progress
        job.current_epoch = progress_data.current_epoch
        
        if progress_data.current_loss is not None:
            job.current_loss = progress_data.current_loss
        
        if progress_data.status is not None:
            job.status = progress_data.status.value
            
            if progress_data.status == TrainingStatus.RUNNING and job.started_at is None:
                job.started_at = datetime.utcnow()
            elif progress_data.status in (TrainingStatus.COMPLETED, TrainingStatus.FAILED):
                job.completed_at = datetime.utcnow()
        
        if progress_data.error_message is not None:
            job.error_message = progress_data.error_message
        
        self.db.commit()
        self.db.refresh(job)
        
        return job
    
    def cancel_job(self, job_id: int) -> TrainingJob | Project | None:
        """
        Cancel a training job or project.
        
        Handles both TrainingJob IDs and Project IDs (since Projects appear in jobs list).
        Handles jobs/projects stuck in running state even when no workers are active.
        
        Args:
            job_id: Training job ID or Project ID.
            
        Returns:
            Cancelled TrainingJob or Project object.
            
        Raises:
            ValueError: If job/project not found or cannot be cancelled.
        """
        # First try to get as TrainingJob
        job = self.get_job(job_id)
        
        if job:
            # It's a TrainingJob - cancel it
            if job.status in (TrainingStatus.COMPLETED.value, TrainingStatus.CANCELLED.value):
                raise ValueError(f"Cannot cancel job with status: {job.status}")
            
            # Cancel in worker pool if running (even if no workers, this will clean up queue)
            if job.status == TrainingStatus.RUNNING.value:
                cancelled_in_pool = self.worker_pool.cancel_job(job.id)
                # If job wasn't in pool (e.g., workers stopped but job still marked running),
                # we still cancel it - this handles stuck jobs
                if not cancelled_in_pool:
                    logger.warning(f"Job {job_id} was running but not found in worker pool - cancelling anyway (likely workers were stopped)")
            
            # Always update status to cancelled, even if not found in worker pool
            # This handles the case where workers were stopped but job is still marked running
            job.status = TrainingStatus.CANCELLED.value
            job.completed_at = datetime.utcnow()
            job.worker_id = None  # Clear worker assignment
            
            self.db.commit()
            self.db.refresh(job)
            
            logger.info(f"Cancelled job {job_id} (status was: {job.status})")
            return job
        
        # If not found as TrainingJob, try as Project
        from app.models.project import Project, ProjectStatus
        from app.services.project_service import ProjectService
        
        project = self.db.query(Project).filter(Project.id == job_id).first()
        if project:
            # Cancel via ProjectService
            project_service = ProjectService(db=self.db)
            return project_service.cancel_project(job_id)
        
        raise ValueError(f"Job or Project with ID {job_id} not found")
    
    def delete_job(self, job_id: int) -> bool:
        """
        Delete a training job.
        
        Args:
            job_id: Training job ID.
            
        Returns:
            True if deleted, False if not found.
        """
        job = self.get_job(job_id)
        if not job:
            return False
        
        self.db.delete(job)
        self.db.commit()
        
        return True
    
    def get_config(self) -> TrainingConfig:
        """
        Get the training configuration.
        
        Creates default configuration if it doesn't exist.
        
        Returns:
            TrainingConfig object.
        """
        config = self.db.query(TrainingConfig).first()
        if not config:
            config = TrainingConfig()
            self.db.add(config)
            self.db.commit()
            self.db.refresh(config)
        return config
    
    def update_config(self, update_data: TrainingConfigUpdate) -> TrainingConfig:
        """
        Update the training configuration.
        
        Args:
            update_data: Configuration update data.
            
        Returns:
            Updated TrainingConfig object.
        """
        import json
        
        config = self.get_config()
        
        update_dict = update_data.model_dump(exclude_unset=True)
        
        # Handle selected_gpus: convert list to JSON string
        if "selected_gpus" in update_dict:
            gpus = update_dict.pop("selected_gpus")
            if gpus is not None:
                config.selected_gpus = json.dumps(gpus)
            else:
                config.selected_gpus = None
        
        # Handle other fields
        for field, value in update_dict.items():
            setattr(config, field, value)
        
        self.db.commit()
        self.db.refresh(config)
        
        # Update worker pool if max workers changed
        if "max_concurrent_workers" in update_dict:
            self.worker_pool.set_max_workers(config.max_concurrent_workers)
        
        return config
    
    def get_worker_pool_status(self) -> WorkerPoolStatus:
        """
        Get the current worker pool status.
        
        Returns:
            WorkerPoolStatus with pool information.
        """
        # Submit any pending or orphaned queued jobs to the pool if it's running
        if self.worker_pool.is_running:
            # Get all jobs currently in the pool (before submitting any)
            jobs_in_pool_before = self.worker_pool.get_all_jobs_in_pool()
            logger.debug(f"Jobs currently in pool: {jobs_in_pool_before}")
            
            # Submit pending jobs first
            pending_jobs = self.db.query(TrainingJob).filter(
                TrainingJob.status == TrainingStatus.PENDING.value
            ).all()
            if pending_jobs:
                logger.info(f"Found {len(pending_jobs)} pending jobs to submit")
            for job in pending_jobs:
                if self.worker_pool.submit_job(job.id):
                    job.status = TrainingStatus.QUEUED.value
            
            # Get updated jobs in pool after submitting pending jobs
            jobs_in_pool_after = self.worker_pool.get_all_jobs_in_pool()
            
            # Re-submit queued jobs that aren't actually in the pool (orphaned)
            queued_jobs = self.db.query(TrainingJob).filter(
                TrainingJob.status == TrainingStatus.QUEUED.value
            ).all()
            if queued_jobs:
                logger.info(f"Found {len(queued_jobs)} queued jobs in DB, checking for orphans...")
            orphaned_count = 0
            for job in queued_jobs:
                if job.id not in jobs_in_pool_after:
                    orphaned_count += 1
                    # Re-submit this orphaned queued job
                    logger.info(f"Found orphaned queued job {job.id}, re-submitting to pool")
                    if self.worker_pool.submit_job(job.id):
                        logger.info(f"Orphaned job {job.id} re-submitted successfully")
                    else:
                        logger.warning(f"Failed to re-submit orphaned job {job.id} - pool not running?")
            if orphaned_count > 0:
                logger.info(f"Re-submitted {orphaned_count} orphaned jobs")
            
            if pending_jobs or orphaned_count > 0:
                self.db.commit()
        else:
            logger.debug("Worker pool is not running, skipping orphaned job detection")
        
        pool_status = self.worker_pool.get_status()
        config = self.get_config()
        
        # Use the actual queue size from the worker pool
        # This represents jobs actually queued in the pool, not just pending in DB
        queue_size = pool_status.get("queue_size", 0)
        
        workers = [
            WorkerInfo(
                id=w["id"],
                status=WorkerStatus(w["status"]),
                current_job_id=w.get("current_job_id"),
                jobs_completed=w.get("jobs_completed", 0),
                started_at=w["started_at"],
                last_activity=w.get("last_activity"),
            )
            for w in pool_status.get("workers", [])
        ]
        
        return WorkerPoolStatus(
            total_workers=pool_status.get("total_workers", 0),
            active_workers=pool_status.get("active_workers", 0),
            idle_workers=pool_status.get("idle_workers", 0),
            busy_workers=pool_status.get("busy_workers", 0),
            max_workers=config.max_concurrent_workers,
            workers=workers,
            jobs_in_queue=queue_size,
        )
    
    def start_workers(self, count: int) -> None:
        """
        Start workers.
        
        Args:
            count: Number of workers to start.
            
        Raises:
            ValueError: If count exceeds maximum workers.
        """
        import json
        
        config = self.get_config()
        if count > config.max_concurrent_workers:
            raise ValueError(
                f"Cannot start {count} workers. "
                f"Maximum allowed: {config.max_concurrent_workers}"
            )
        
        # Get GPU configuration
        selected_gpus = None
        if not config.gpu_auto_detect and config.selected_gpus:
            try:
                selected_gpus = json.loads(config.selected_gpus)
            except (json.JSONDecodeError, TypeError):
                selected_gpus = None
        elif config.gpu_auto_detect:
            # Auto-detect: get all available GPUs
            from app.services.gpu_service import GPUService
            gpu_service = GPUService()
            gpus = gpu_service.detect_gpus()
            selected_gpus = [gpu["id"] for gpu in gpus] if gpus else None
        
        self.worker_pool.start_workers(count, selected_gpus=selected_gpus)
        
        # Submit any pending or orphaned queued jobs to the pool now that it's running
        # First, get current jobs in pool (before submitting new ones)
        jobs_in_pool_before = self.worker_pool.get_all_jobs_in_pool()
        
        # Submit pending jobs first
        pending_jobs = self.db.query(TrainingJob).filter(
            TrainingJob.status == TrainingStatus.PENDING.value
        ).all()
        submitted_job_ids = set()
        for job in pending_jobs:
            if self.worker_pool.submit_job(job.id):
                job.status = TrainingStatus.QUEUED.value
                submitted_job_ids.add(job.id)
        
        # Get updated jobs in pool after submitting pending jobs
        jobs_in_pool_after = self.worker_pool.get_all_jobs_in_pool()
        
        # Re-submit orphaned queued jobs (those in DB but not in pool)
        queued_jobs = self.db.query(TrainingJob).filter(
            TrainingJob.status == TrainingStatus.QUEUED.value
        ).all()
        orphaned_count = 0
        for job in queued_jobs:
            if job.id not in jobs_in_pool_after:
                orphaned_count += 1
                if self.worker_pool.submit_job(job.id):
                    logger.info(f"Re-submitted orphaned queued job {job.id}")
        
        if pending_jobs or orphaned_count > 0:
            self.db.commit()
            if pending_jobs:
                logger.info(f"Submitted {len(pending_jobs)} pending jobs to worker pool")
            if orphaned_count > 0:
                logger.info(f"Re-submitted {orphaned_count} orphaned queued jobs")
        
        # Update active workers count
        config.active_workers = self.worker_pool.active_worker_count
        self.db.commit()
    
    def stop_workers(self) -> None:
        """Stop all workers."""
        self.worker_pool.stop_all_workers()
        
        config = self.get_config()
        config.active_workers = 0
        self.db.commit()
    
    def restart_workers(self) -> None:
        """Restart all workers."""
        import json
        
        config = self.get_config()
        current_count = self.worker_pool.active_worker_count
        
        # Get GPU configuration
        selected_gpus = None
        if not config.gpu_auto_detect and config.selected_gpus:
            try:
                selected_gpus = json.loads(config.selected_gpus)
            except (json.JSONDecodeError, TypeError):
                selected_gpus = None
        elif config.gpu_auto_detect:
            # Auto-detect: get all available GPUs
            from app.services.gpu_service import GPUService
            gpu_service = GPUService()
            gpus = gpu_service.detect_gpus()
            selected_gpus = [gpu["id"] for gpu in gpus] if gpus else None
        
        self.worker_pool.stop_all_workers()
        self.worker_pool.start_workers(current_count or config.max_concurrent_workers, selected_gpus=selected_gpus)
        
        config.active_workers = self.worker_pool.active_worker_count
        self.db.commit()
    
    def get_model_download_path(self, job_id: int) -> tuple[Path, str] | None:
        """
        Get the download path for a completed training job's model.
        
        For directories with multiple files, creates a tar.gz archive.
        For single files, returns the file path directly.
        
        Args:
            job_id: Training job ID.
            
        Returns:
            Tuple of (file_path, filename) or None if not available.
            
        Raises:
            ValueError: If job not found, not completed, or has no model.
        """
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job with ID {job_id} not found")
        
        if job.status != TrainingStatus.COMPLETED.value:
            raise ValueError(f"Job {job_id} is not completed (status: {job.status})")
        
        if not job.model_path:
            raise ValueError(f"Job {job_id} has no model path")
        
        model_path = Path(job.model_path)
        if not model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
        
        # If it's a single file, return it directly
        if model_path.is_file():
            return model_path, model_path.name
        
        # It's a directory - check if we need to create an archive
        files = list(model_path.rglob("*"))
        file_count = sum(1 for f in files if f.is_file())
        
        if file_count == 0:
            raise ValueError(f"Model directory is empty: {model_path}")
        
        if file_count == 1:
            # Single file in directory, return it directly
            single_file = next(f for f in files if f.is_file())
            return single_file, single_file.name
        
        # Multiple files - create/get tar.gz archive
        archive_dir = settings.get_archive_path()
        
        archive_name = f"model_job_{job_id}.tar.gz"
        archive_path = archive_dir / archive_name
        
        # Check if archive already exists and is up to date
        if archive_path.exists():
            # Check if any model file is newer than the archive
            archive_mtime = archive_path.stat().st_mtime
            model_files_newer = any(
                f.stat().st_mtime > archive_mtime for f in files if f.is_file()
            )
            if not model_files_newer:
                # Archive is up to date
                return archive_path, archive_name
        
        # Create new archive
        logger.info(f"Creating archive for job {job_id}: {archive_path}")
        with tarfile.open(archive_path, "w:gz") as tar:
            # Add files with relative paths from model directory
            for file_path in files:
                if file_path.is_file():
                    arcname = file_path.relative_to(model_path.parent)
                    tar.add(file_path, arcname=str(arcname))
        
        logger.info(f"Archive created: {archive_path} ({archive_path.stat().st_size} bytes)")
        return archive_path, archive_name
    
    def get_model_file_info(self, job_id: int) -> dict[str, Any] | None:
        """
        Get information about a job's model files.
        
        Args:
            job_id: Training job ID.
            
        Returns:
            Dictionary with file info or None if not available.
        """
        job = self.get_job(job_id)
        if not job or not job.model_path:
            return None
        
        model_path = Path(job.model_path)
        if not model_path.exists():
            return None
        
        if model_path.is_file():
            return {
                "type": "file",
                "name": model_path.name,
                "size": model_path.stat().st_size,
                "path": str(model_path),
            }
        
        # Directory
        files = list(model_path.rglob("*"))
        file_list = [
            {"name": f.relative_to(model_path), "size": f.stat().st_size}
            for f in files if f.is_file()
        ]
        total_size = sum(f["size"] for f in file_list)
        
        return {
            "type": "directory",
            "name": model_path.name,
            "file_count": len(file_list),
            "total_size": total_size,
            "files": file_list,
            "path": str(model_path),
        }


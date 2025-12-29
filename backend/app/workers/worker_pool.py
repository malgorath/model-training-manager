"""
Worker pool manager.

Manages a pool of training workers for concurrent job processing.
"""

import logging
import threading
from datetime import datetime
from typing import Any, Callable

from sqlalchemy.orm import Session

from app.workers.training_worker import TrainingWorker

logger = logging.getLogger(__name__)


class WorkerPool:
    """
    Manager for a pool of training workers.
    
    Handles worker lifecycle, job distribution, and pool scaling.
    
    Attributes:
        max_workers: Maximum number of workers allowed.
        is_running: Whether the pool is running.
        active_worker_count: Number of currently active workers.
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        db_session_factory: Callable[[], Session] | None = None,
    ):
        """
        Initialize the worker pool.
        
        Args:
            max_workers: Maximum number of workers.
            db_session_factory: Factory function to create DB sessions.
        """
        self.max_workers = max_workers
        self._db_session_factory = db_session_factory
        self._workers: dict[str, TrainingWorker] = {}
        self._lock = threading.Lock()
        self._is_running = False
        self._job_queue: list[int] = []
        self._project_queue: list[int] = []
        self._queue_lock = threading.Lock()
    
    @property
    def is_running(self) -> bool:
        """Check if the pool is running."""
        return self._is_running
    
    @property
    def active_worker_count(self) -> int:
        """Get the number of active workers."""
        with self._lock:
            return len([w for w in self._workers.values() if w.status != "stopped"])
    
    def start_workers(self, count: int) -> None:
        """
        Start a specified number of workers.
        
        Args:
            count: Number of workers to start.
            
        Raises:
            ValueError: If count exceeds maximum workers.
        """
        with self._lock:
            current_count = len(self._workers)
            if current_count + count > self.max_workers:
                raise ValueError(
                    f"Cannot start {count} workers. "
                    f"Current: {current_count}, Max: {self.max_workers}"
                )
            
            for _ in range(count):
                worker = TrainingWorker(
                    db_session_factory=self._db_session_factory,
                )
                worker.start()
                self._workers[worker.id] = worker
                logger.info(f"WorkerPool: Started worker {worker.id}")
            
            self._is_running = True
    
    def stop_worker(self, worker_id: str) -> bool:
        """
        Stop a specific worker.
        
        Args:
            worker_id: ID of worker to stop.
            
        Returns:
            True if worker was stopped, False if not found.
        """
        with self._lock:
            worker = self._workers.get(worker_id)
            if not worker:
                return False
            
            worker.stop()
            del self._workers[worker_id]
            logger.info(f"WorkerPool: Stopped worker {worker_id}")
            
            if not self._workers:
                self._is_running = False
            
            return True
    
    def stop_all_workers(self) -> None:
        """Stop all workers in the pool."""
        with self._lock:
            for worker in list(self._workers.values()):
                worker.stop()
            
            self._workers.clear()
            self._is_running = False
            logger.info("WorkerPool: All workers stopped")
    
    def submit_job(self, job_id: int) -> bool:
        """
        Submit a job for processing.
        
        The job will be assigned to an idle worker or queued
        if all workers are busy.
        
        Args:
            job_id: Training job ID.
            
        Returns:
            True if job was submitted, False if pool not running.
        """
        if not self._is_running:
            return False
        
        with self._lock:
            # Find an idle worker
            for worker in self._workers.values():
                if worker.status == "idle":
                    worker.submit_job(job_id)
                    logger.info(f"WorkerPool: Job {job_id} assigned to {worker.id}")
                    return True
            
            # Queue the job if no idle workers
            with self._queue_lock:
                self._job_queue.append(job_id)
            logger.info(f"WorkerPool: Job {job_id} queued (no idle workers)")
        
        # Try to distribute queued jobs (in case workers become available)
        self.distribute_queued_jobs()
        return True
    
    def cancel_job(self, job_id: int) -> bool:
        """
        Cancel a job.
        
        Args:
            job_id: Training job ID to cancel.
            
        Returns:
            True if job was cancelled, False if not found.
        """
        # Check if job is in queue
        with self._queue_lock:
            if job_id in self._job_queue:
                self._job_queue.remove(job_id)
                logger.info(f"WorkerPool: Job {job_id} removed from queue")
                return True
        
        # Check if job is being processed
        with self._lock:
            for worker in self._workers.values():
                if worker.current_job_id == job_id:
                    worker.cancel_current_job()
                    logger.info(f"WorkerPool: Job {job_id} cancellation requested")
                    return True
        
        return False
    
    def set_max_workers(self, max_workers: int) -> None:
        """
        Update the maximum number of workers.
        
        Args:
            max_workers: New maximum worker count.
        """
        self.max_workers = max_workers
        logger.info(f"WorkerPool: Max workers set to {max_workers}")
    
    def get_status(self) -> dict[str, Any]:
        """
        Get the current pool status.
        
        Returns:
            Dictionary with pool status information.
        """
        # Distribute queued jobs before returning status
        # This ensures jobs get assigned when workers become idle
        self.distribute_queued_jobs()
        
        with self._lock:
            workers_info = [w.get_info() for w in self._workers.values()]
            
            idle_count = sum(1 for w in workers_info if w["status"] == "idle")
            busy_count = sum(1 for w in workers_info if w["status"] == "busy")
            
            with self._queue_lock:
                queue_size = len(self._job_queue)
            
            return {
                "total_workers": len(self._workers),
                "active_workers": len([w for w in workers_info if w["status"] != "stopped"]),
                "idle_workers": idle_count,
                "busy_workers": busy_count,
                "max_workers": self.max_workers,
                "is_running": self._is_running,
                "queue_size": queue_size,
                "workers": workers_info,
            }
    
    def get_worker(self, worker_id: str) -> TrainingWorker | None:
        """
        Get a specific worker by ID.
        
        Args:
            worker_id: Worker ID.
            
        Returns:
            TrainingWorker or None if not found.
        """
        with self._lock:
            return self._workers.get(worker_id)
    
    def distribute_queued_jobs(self) -> None:
        """
        Distribute queued jobs to idle workers.
        
        Called periodically to assign pending jobs.
        """
        with self._lock:
            for worker in self._workers.values():
                if worker.status == "idle":
                    with self._queue_lock:
                        if self._job_queue:
                            job_id = self._job_queue.pop(0)
                            worker.submit_job(job_id)
                            logger.info(
                                f"WorkerPool: Queued job {job_id} "
                                f"assigned to {worker.id}"
                            )
                        elif self._project_queue:
                            project_id = self._project_queue.pop(0)
                            if worker.add_project(project_id):
                                logger.info(
                                    f"WorkerPool: Queued project {project_id} "
                                    f"assigned to {worker.id}"
                                )
    
    def distribute_queued_projects(self) -> None:
        """Distribute queued projects to idle workers."""
        self.distribute_queued_jobs()  # Reuse same logic
    
    def add_workers(self, count: int) -> None:
        """
        Add additional workers to the pool.
        
        Args:
            count: Number of workers to add.
            
        Raises:
            ValueError: If adding would exceed max workers.
        """
        with self._lock:
            current_count = len(self._workers)
            if current_count + count > self.max_workers:
                raise ValueError(
                    f"Cannot add {count} workers. "
                    f"Would exceed max of {self.max_workers}"
                )
            
            for _ in range(count):
                worker = TrainingWorker(
                    db_session_factory=self._db_session_factory,
                )
                worker.start()
                self._workers[worker.id] = worker
                logger.info(f"WorkerPool: Added worker {worker.id}")
    
    def get_queued_jobs(self) -> list[int]:
        """
        Get list of queued job IDs.
        
        Returns:
            List of job IDs waiting in queue.
        """
        with self._queue_lock:
            return list(self._job_queue)
    
    def get_active_jobs(self) -> list[int]:
        """
        Get list of active job IDs being processed.
        
        Returns:
            List of job IDs currently being processed.
        """
        with self._lock:
            active_jobs = []
            for worker in self._workers.values():
                if worker.current_job_id is not None:
                    active_jobs.append(worker.current_job_id)
            return active_jobs
    
    def get_all_jobs_in_pool(self) -> set[int]:
        """
        Get set of all job IDs that are in the pool (queued or active).
        
        Includes jobs in pool queue, worker queues, and currently processing.
        
        Returns:
            Set of job IDs in the pool.
        """
        with self._lock:
            job_ids = set()
            # Jobs in pool queue
            with self._queue_lock:
                job_ids.update(self._job_queue)
            # Jobs in worker queues and currently processing
            for worker in self._workers.values():
                if worker.current_job_id is not None:
                    job_ids.add(worker.current_job_id)
                # Check worker's internal queue
                job_ids.update(worker.get_queued_jobs())
            return job_ids


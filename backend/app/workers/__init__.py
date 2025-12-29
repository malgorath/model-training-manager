"""
Training worker implementation.

This module exports the worker pool manager and training worker classes.
"""

from app.workers.training_worker import TrainingWorker
from app.workers.worker_pool import WorkerPool

__all__ = [
    "TrainingWorker",
    "WorkerPool",
]


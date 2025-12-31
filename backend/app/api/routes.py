"""
API route definitions.

This module defines all API endpoints for the application.
"""

from fastapi import APIRouter

from app.api.endpoints import datasets, training_jobs, config, workers, huggingface, projects, models

router = APIRouter()

# Include all endpoint routers
router.include_router(
    datasets.router,
    prefix="/datasets",
    tags=["Datasets"],
)

router.include_router(
    training_jobs.router,
    prefix="/jobs",
    tags=["Training Jobs"],
)

router.include_router(
    config.router,
    prefix="/config",
    tags=["Configuration"],
)

router.include_router(
    workers.router,
    prefix="/workers",
    tags=["Workers"],
)

router.include_router(
    huggingface.router,
    tags=["Hugging Face"],
)

router.include_router(
    projects.router,
    prefix="/projects",
    tags=["Projects"],
)
router.include_router(
    models.router,
    prefix="/models",
    tags=["Models"],
)
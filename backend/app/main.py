"""
FastAPI application entry point.

This module initializes and configures the FastAPI application.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.database import init_db
from app.api.routes import router

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    
    Handles startup and shutdown events for the application.
    """
    # Startup
    init_db()
    settings.get_upload_path()  # Ensure upload directory exists
    
    # Auto-start workers if configured
    from app.core.database import SessionLocal
    from app.services.training_service import TrainingService
    from app.models.training_config import TrainingConfig
    
    db = SessionLocal()
    try:
        config = db.query(TrainingConfig).first()
        if config and config.auto_start_workers:
            logger = logging.getLogger(__name__)
            logger.info("Auto-starting workers on startup (auto_start_workers=True)")
            service = TrainingService(db=db)
            # Start 1 worker by default (or use max_concurrent_workers if configured)
            worker_count = min(1, config.max_concurrent_workers) if config.max_concurrent_workers > 0 else 1
            try:
                service.start_workers(worker_count)
                logger.info(f"Auto-started {worker_count} worker(s) on startup")
            except Exception as e:
                logger.error(f"Failed to auto-start workers: {e}")
    finally:
        db.close()
    
    yield
    
    # Shutdown
    # Cleanup code here if needed


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured FastAPI application instance.
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="A professional model training management application with worker orchestration.",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API router
    app.include_router(router, prefix="/api/v1")
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "version": settings.app_version,
            "environment": settings.environment,
        }
    
    return app


# Create the application instance
app = create_app()


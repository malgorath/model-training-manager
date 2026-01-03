"""
Project Training Worker Extension.

Extends TrainingWorker with project-based training capabilities.
"""

import logging
from pathlib import Path
from typing import Any
from sqlalchemy.orm import Session

from app.models.project import Project, ProjectStatus
from app.models.training_config import TrainingConfig
from app.services.project_service import ProjectService
from app.services.model_resolution_service import ModelResolutionService
from app.services.model_validation_service import ModelValidationService
from app.workers.training_worker import TrainingWorker

logger = logging.getLogger(__name__)


def process_project(worker: TrainingWorker, project_id: int, db: Session) -> None:
    """
    Process a training project.
    
    Combines datasets from all traits based on percentages and trains a model.
    
    Args:
        worker: TrainingWorker instance.
        project_id: Project ID to process.
        db: Database session.
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        logger.error(f"Worker {worker.id}: Project {project_id} not found")
        return
    
    project_service = ProjectService(db=db)
    model_resolver = ModelResolutionService()
    validation_service = ModelValidationService()
    
    try:
        # Update project status
        project.status = ProjectStatus.RUNNING.value
        project.worker_id = worker.id
        project.started_at = datetime.utcnow()
        if project.log is None:
            project.log = ""
        worker._append_log(project, f"🎯 Project assigned to worker {worker.id}", db)
        db.commit()
        
        logger.info(f"Worker {worker.id}: Processing project {project_id} ({project.base_model})")
        
        # Validate model availability
        if not model_resolver.is_model_available(project.base_model):
            raise ModelNotFoundError(
                f"Model '{project.base_model}' not found. Please ensure it's downloaded or configure a local path override."
            )
        
        resolved_model_path = model_resolver.resolve_model_path(project.base_model)
        model_resolver.validate_model_format(resolved_model_path)
        worker._append_log(project, f"✅ Model validated: {resolved_model_path}", db)
        
        # Combine datasets from all traits
        combined_datasets = project_service.combine_datasets_for_training(
            project_id=project.id,
            max_rows=project.max_rows,
        )
        
        worker._append_log(project, f"📊 Combined {len(combined_datasets)} datasets for training", db)
        
        # Load and combine dataset data
        all_data = []
        for dataset_allocation in combined_datasets:
            dataset = dataset_allocation["dataset"]
            rows_to_use = dataset_allocation["rows"]
            
            # Load dataset
            data = worker._load_dataset(dataset)
            
            # Sample rows based on allocation
            if len(data) > rows_to_use:
                import random
                data = random.sample(data, rows_to_use)
            
            all_data.extend(data)
            worker._append_log(
                project,
                f"  - {dataset.name}: {len(data)} rows ({dataset_allocation['percentage']}%)",
                db,
            )
        
        worker._append_log(project, f"✅ Total training data: {len(all_data)} rows", db)
        
        # Determine output directory
        output_dir = Path(project.output_directory).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Train based on training type
        if project.training_type == "qlora":
            worker._train_qlora_for_project(project, all_data, output_dir, db, resolved_model_path)
        elif project.training_type == "unsloth":
            worker._train_unsloth_for_project(project, all_data, output_dir, db, resolved_model_path)
        elif project.training_type == "rag":
            worker._train_rag_for_project(project, all_data, output_dir, db, resolved_model_path)
        else:
            worker._train_standard_for_project(project, all_data, output_dir, db, resolved_model_path)
        
        # Validate trained model
        worker._append_log(project, "🔍 Validating trained model...", db)
        # Ensure model_path is committed to database before validation
        db.commit()  # Commit to ensure model_path is saved
        db.refresh(project)  # Refresh to get latest model_path
        
        # For RAG models, validate the model_path (rag_model subdirectory), not output_dir
        # For other models, model_path might be set to output_dir or a subdirectory
        if project.training_type == "rag":
            # For RAG, always use the rag_model subdirectory
            validation_path = str(output_dir / "rag_model")
        else:
            validation_path = project.model_path if project.model_path else str(output_dir)
        
        worker._append_log(project, f"🔍 Validating model at: {validation_path}", db)
        validation_result = validation_service.validate_model_complete(validation_path)
        
        if validation_result["valid"]:
            project.status = ProjectStatus.COMPLETED.value
            project.progress = 100.0
            # Don't overwrite model_path if it's already set (e.g., for RAG models)
            if not project.model_path:
                project.model_path = str(output_dir)
            worker._append_log(project, "✅ Model validation passed!", db)
            worker._append_log(project, "✅ Project training completed successfully!", db)
        else:
            project.status = ProjectStatus.FAILED.value
            project.error_message = "Model validation failed: " + "; ".join(validation_result.get("errors", []))
            worker._append_log(project, f"❌ Model validation failed: {project.error_message}", db)
        
        project.completed_at = datetime.utcnow()
        db.commit()
        
        logger.info(f"Worker {worker.id}: Project {project_id} completed")
        
    except Exception as e:
        logger.error(f"Worker {worker.id}: Project {project_id} failed - {str(e)}")
        if project:
            project.status = ProjectStatus.FAILED.value
            project.error_message = str(e)
            project.completed_at = datetime.utcnow()
            worker._append_log(project, f"❌ Training failed: {str(e)}", db)
            db.commit()


# Add project training methods to TrainingWorker
from datetime import datetime

def _train_qlora_for_project(
    worker: TrainingWorker,
    project: Project,
    data: list[dict],
    output_dir: Path,
    db: Session,
    model_path: str,
) -> None:
    """Train QLoRA for a project using combined dataset data."""
    # Use project's output directory and train with combined data
    # This is similar to _train_qlora_real but uses project data structure
    worker._append_log(project, "🚀 Starting QLoRA training...", db)
    
    # Create a temporary dataset-like object for compatibility
    class TempDataset:
        def __init__(self, data):
            self.data = data
    
    temp_dataset = TempDataset(data)
    
    # Reuse existing QLoRA training logic
    worker._train_qlora_real(
        job=project,  # Project can be used like a job for compatibility
        dataset=temp_dataset,
        data=data,
        output_dir=output_dir,
        db=db,
        model_path=model_path,
    )
    
    project.model_path = str(output_dir / "lora_model")


def _train_unsloth_for_project(
    worker: TrainingWorker,
    project: Project,
    data: list[dict],
    output_dir: Path,
    db: Session,
    model_path: str,
) -> None:
    """Train Unsloth for a project."""
    worker._append_log(project, "🚀 Starting Unsloth training...", db)
    
    class TempDataset:
        def __init__(self, data):
            self.data = data
    
    temp_dataset = TempDataset(data)
    
    worker._train_unsloth_real(
        job=project,
        dataset=temp_dataset,
        data=data,
        output_dir=output_dir,
        db=db,
        model_path=model_path,
    )
    
    project.model_path = str(output_dir / "lora_model")


def _train_rag_for_project(
    worker: TrainingWorker,
    project: Project,
    data: list[dict],
    output_dir: Path,
    db: Session,
    model_path: str,
) -> None:
    """Train RAG for a project."""
    worker._append_log(project, "🚀 Starting RAG training...", db)
    
    # Set model_path before calling _train_rag_real (it needs job.model_path)
    project.model_path = str(output_dir / "rag_model")
    db.flush()  # Ensure model_path is persisted before training
    
    worker._train_rag_real(
        job=project,
        data=data,
        output_dir=output_dir,
        db=db,
    )


def _train_standard_for_project(
    worker: TrainingWorker,
    project: Project,
    data: list[dict],
    output_dir: Path,
    db: Session,
    model_path: str,
) -> None:
    """Train standard fine-tuning for a project."""
    worker._append_log(project, "🚀 Starting standard training...", db)
    
    worker._train_standard_real(
        job=project,
        data=data,
        output_dir=output_dir,
        db=db,
        model_path=model_path,
    )
    
    project.model_path = str(output_dir / "standard_model")


# Monkey-patch methods onto TrainingWorker class
TrainingWorker.process_project = staticmethod(process_project)
TrainingWorker._train_qlora_for_project = _train_qlora_for_project
TrainingWorker._train_unsloth_for_project = _train_unsloth_for_project
TrainingWorker._train_rag_for_project = _train_rag_for_project
TrainingWorker._train_standard_for_project = _train_standard_for_project

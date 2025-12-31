"""
Clean all projects, training jobs, and logs from database.

Preserves:
- Settings (TrainingConfig)
- Models (DownloadedModel)
- Datasets (Dataset)
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.database import SessionLocal, init_db
from app.models.project import Project, ProjectTrait, ProjectTraitDataset
from app.models.training_job import TrainingJob


def clean_projects_jobs_logs():
    """Remove all projects, training jobs, and their logs."""
    init_db()
    db = SessionLocal()
    
    try:
        # Count before deletion
        project_count = db.query(Project).count()
        job_count = db.query(TrainingJob).count()
        
        print(f"Found {project_count} projects and {job_count} training jobs")
        
        # Delete ProjectTraitDataset (child of ProjectTrait)
        trait_dataset_count = db.query(ProjectTraitDataset).count()
        db.query(ProjectTraitDataset).delete()
        print(f"Deleted {trait_dataset_count} project trait datasets")
        
        # Delete ProjectTrait (child of Project)
        trait_count = db.query(ProjectTrait).count()
        db.query(ProjectTrait).delete()
        print(f"Deleted {trait_count} project traits")
        
        # Delete Projects (this will cascade to traits/datasets)
        db.query(Project).delete()
        print(f"Deleted {project_count} projects")
        
        # Clear logs from projects (already deleted, but just in case)
        # Projects are deleted, so logs are gone
        
        # Delete TrainingJobs (includes their logs)
        db.query(TrainingJob).delete()
        print(f"Deleted {job_count} training jobs")
        
        # Commit changes
        db.commit()
        
        # Verify
        remaining_projects = db.query(Project).count()
        remaining_jobs = db.query(TrainingJob).count()
        
        print(f"\n✅ Cleanup complete!")
        print(f"   Remaining projects: {remaining_projects}")
        print(f"   Remaining jobs: {remaining_jobs}")
        
        # Verify preserved data
        from app.models.dataset import Dataset
        from app.models.downloaded_model import DownloadedModel
        from app.models.training_config import TrainingConfig
        
        dataset_count = db.query(Dataset).count()
        model_count = db.query(DownloadedModel).count()
        config_count = db.query(TrainingConfig).count()
        
        print(f"\n✅ Preserved data:")
        print(f"   Datasets: {dataset_count}")
        print(f"   Models: {model_count}")
        print(f"   Settings: {config_count}")
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error during cleanup: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    clean_projects_jobs_logs()

"""
Migration script to move existing datasets from ./uploads to ./data/author/datasetname structure.

This script:
1. Scans the uploads directory for existing dataset files
2. Attempts to extract author and dataset name from filenames or database records
3. Moves files to the new structure: ./data/{author}/{datasetname}/
4. Updates database records with new paths

Usage:
    python -m scripts.migrate_datasets
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional, Tuple

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.core.database import Base
from app.models.dataset import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_author_and_name_from_filename(filename: str) -> Tuple[str, str]:
    """
    Extract author and dataset name from filename.
    
    Handles patterns like:
    - hf_author_datasetname_split.json -> (author, datasetname)
    - datasetname.json -> (user, datasetname)
    - uuid.json -> (user, unknown)
    
    Args:
        filename: Original filename.
        
    Returns:
        Tuple of (author, dataset_name).
    """
    # Remove extension
    name = Path(filename).stem
    
    # Check for HuggingFace pattern: hf_author_datasetname_split
    if name.startswith("hf_"):
        parts = name[3:].split("_")  # Remove "hf_" prefix
        if len(parts) >= 2:
            # Try to find where split name starts (train, test, validation)
            split_names = ["train", "test", "validation", "val"]
            split_idx = None
            for i, part in enumerate(parts):
                if part in split_names:
                    split_idx = i
                    break
            
            if split_idx and split_idx > 0:
                # author is first part, dataset name is everything before split
                author = parts[0]
                dataset_name = "_".join(parts[1:split_idx])
                return author, dataset_name
            elif len(parts) >= 2:
                # Assume first is author, rest is dataset name
                author = parts[0]
                dataset_name = "_".join(parts[1:])
                return author, dataset_name
    
    # Check if it's a UUID (hex string)
    if re.match(r'^[0-9a-f]{32}$', name, re.IGNORECASE):
        return "user", "unknown"
    
    # Default: use as dataset name with "user" as author
    return "user", name


def migrate_dataset_file(
    old_path: Path,
    author: str,
    dataset_name: str,
    new_base_dir: Path,
) -> Path:
    """
    Move a dataset file to the new structure.
    
    Args:
        old_path: Current file path.
        author: Author name.
        dataset_name: Dataset name.
        new_base_dir: New base directory (./data).
        
    Returns:
        New file path.
    """
    # Sanitize names
    sanitized_author = settings._sanitize_name(author)
    sanitized_dataset = settings._sanitize_name(dataset_name)
    
    # Create new directory structure
    new_dir = new_base_dir / sanitized_author / sanitized_dataset
    new_dir.mkdir(parents=True, exist_ok=True)
    
    # Move file
    new_path = new_dir / old_path.name
    if old_path.exists():
        old_path.rename(new_path)
        logger.info(f"Moved {old_path} -> {new_path}")
    else:
        logger.warning(f"File not found: {old_path}")
    
    return new_path


def migrate_datasets():
    """Main migration function."""
    # Setup database
    engine = create_engine(settings.database_url)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    
    try:
        # Get old uploads directory
        old_uploads_dir = Path("./uploads")
        if not old_uploads_dir.exists():
            logger.info("No uploads directory found. Nothing to migrate.")
            return
        
        # Get new data directory
        new_data_dir = settings.get_upload_path()
        
        # Find all dataset files in uploads
        dataset_files = list(old_uploads_dir.glob("*.json")) + list(old_uploads_dir.glob("*.csv"))
        
        if not dataset_files:
            logger.info("No dataset files found in uploads directory.")
            return
        
        logger.info(f"Found {len(dataset_files)} dataset files to migrate.")
        
        # Process each file
        for file_path in dataset_files:
            try:
                # Try to find corresponding database record
                dataset = db.query(Dataset).filter(
                    Dataset.file_path == str(file_path)
                ).first()
                
                if dataset:
                    # Use dataset name from database
                    author = "user"  # Default for user uploads
                    dataset_name = dataset.name
                    logger.info(f"Found database record for {file_path.name}: {dataset_name}")
                else:
                    # Extract from filename
                    author, dataset_name = extract_author_and_name_from_filename(file_path.name)
                    logger.info(f"No database record, extracted from filename: {author}/{dataset_name}")
                
                # Migrate file
                new_path = migrate_dataset_file(
                    file_path,
                    author,
                    dataset_name,
                    new_data_dir,
                )
                
                # Update database record if exists
                if dataset:
                    dataset.file_path = str(new_path)
                    db.commit()
                    logger.info(f"Updated database record for dataset {dataset.id}")
                
            except Exception as e:
                logger.error(f"Error migrating {file_path}: {e}")
                continue
        
        logger.info("Migration completed!")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    migrate_datasets()

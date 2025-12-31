"""
Downloaded Model Service.

Manages downloaded HuggingFace models stored locally.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Any, Dict
from datetime import datetime

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.downloaded_model import DownloadedModel
from app.models.training_config import TrainingConfig
from app.services.huggingface_service import HuggingFaceService

logger = logging.getLogger(__name__)


class DownloadedModelService:
    """
    Service for managing locally downloaded models.
    
    Handles downloading, tracking, and managing models stored locally.
    """
    
    def __init__(self, db: Session):
        """
        Initialize the downloaded model service.
        
        Args:
            db: Database session.
        """
        self.db = db
        self.model_dir = settings.get_model_path()
    
    def _ensure_model_directory(self, model_id: str) -> Path:
        """
        Ensure directory structure exists for a model.
        
        Args:
            model_id: HuggingFace model ID (e.g., "meta-llama/Llama-3.2-3B-Instruct").
            
        Returns:
            Path to the model directory.
        """
        # Create directory structure: ./data/models/{org}/{model_name}
        parts = model_id.split("/")
        if len(parts) == 2:
            org, model_name = parts
            model_dir = self.model_dir / org / model_name
        else:
            # Fallback for models without org
            model_dir = self.model_dir / model_id.replace("/", "_")
        
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir
    
    def download_model(
        self,
        model_id: str,
        hf_service: HuggingFaceService,
    ) -> DownloadedModel:
        """
        Download a model from HuggingFace and track it in the database.
        
        Args:
            model_id: HuggingFace model ID (e.g., "meta-llama/Llama-3.2-3B-Instruct").
            hf_service: HuggingFaceService instance with token configured.
            
        Returns:
            DownloadedModel instance.
            
        Raises:
            ValueError: If model download fails.
        """
        # Check if model already exists
        existing = self.db.query(DownloadedModel).filter(
            DownloadedModel.model_id == model_id
        ).first()
        
        if existing:
            logger.info(f"Model {model_id} already downloaded at {existing.local_path}")
            return existing
        
        # Ensure directory exists
        model_dir = self._ensure_model_directory(model_id)
        
        # Download model using HuggingFaceService
        try:
            downloaded_path = hf_service.download_model(model_id, model_dir)
        except Exception as e:
            logger.error(f"Failed to download model {model_id}: {e}")
            raise ValueError(f"Failed to download model: {str(e)}")
        
        # Extract basic info from model_id
        parts = model_id.split("/")
        name = parts[-1] if len(parts) > 1 else model_id
        author = parts[0] if len(parts) > 1 else ""
        
        # Calculate file size
        file_size = sum(
            f.stat().st_size
            for f in downloaded_path.rglob("*")
            if f.is_file()
        )
        
        # Create database record
        # Note: description, tags, model_type will be populated from API if available
        downloaded_model = DownloadedModel(
            model_id=model_id,
            name=name,
            author=author,
            description=None,
            local_path=str(downloaded_path),
            file_size=file_size,
            downloaded_at=datetime.utcnow(),
            tags=None,
            model_type=None,
            is_private=False,
        )
            
        try:
            self.db.add(downloaded_model)
            self.db.commit()
            self.db.refresh(downloaded_model)
            
            logger.info(f"Successfully downloaded and tracked model {model_id}")
            return downloaded_model
            
        except Exception as e:
            logger.error(f"Failed to track downloaded model {model_id}: {e}")
            self.db.rollback()
            # Clean up downloaded files if database operation fails
            import shutil
            if downloaded_path.exists():
                shutil.rmtree(downloaded_path, ignore_errors=True)
            raise
    
    def update_model_info(
        self,
        model_id: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        model_type: Optional[str] = None,
        is_private: Optional[bool] = None,
    ) -> DownloadedModel:
        """
        Update model metadata from HuggingFace API info.
        
        Args:
            model_id: HuggingFace model ID.
            description: Model description.
            tags: List of model tags.
            model_type: Model type.
            is_private: Whether model is private.
            
        Returns:
            Updated DownloadedModel instance.
        """
        model = self.get_model(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        if description is not None:
            model.description = description
        if tags is not None:
            model.tags = json.dumps(tags) if tags else None
        if model_type is not None:
            model.model_type = model_type
        if is_private is not None:
            model.is_private = is_private
        
        self.db.commit()
        self.db.refresh(model)
        return model
    
    def list_local_models(self) -> List[DownloadedModel]:
        """
        List all locally downloaded models.
        
        Returns:
            List of DownloadedModel instances.
        """
        return self.db.query(DownloadedModel).order_by(DownloadedModel.downloaded_at.desc()).all()
    
    def get_model(self, model_id: str) -> Optional[DownloadedModel]:
        """
        Get a downloaded model by ID.
        
        Args:
            model_id: HuggingFace model ID.
            
        Returns:
            DownloadedModel instance if found, None otherwise.
        """
        return self.db.query(DownloadedModel).filter(
            DownloadedModel.model_id == model_id
        ).first()
    
    def delete_model(self, model_id: str) -> None:
        """
        Delete a downloaded model and its files.
        
        Args:
            model_id: HuggingFace model ID.
            
        Raises:
            ValueError: If model not found.
        """
        model = self.get_model(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        # Delete files
        model_path = Path(model.local_path)
        if model_path.exists():
            import shutil
            shutil.rmtree(model_path, ignore_errors=True)
            logger.info(f"Deleted model files from {model_path}")
        
        # Delete database record
        self.db.delete(model)
        self.db.commit()
        
        logger.info(f"Deleted model {model_id} from database")
    
    def scan_models(self) -> dict[str, Any]:
        """
        Scan the model directory structure and auto-add valid models to the database.
        
        Scans ./data/models/{org}/{model_name}/ directories for valid model directories
        (containing config.json and tokenizer files), and adds them to the database
        if they don't already exist.
        
        Returns:
            Dictionary with:
                - scanned: Number of model directories scanned
                - added: Number of models added
                - skipped: Number of directories skipped (already in DB or invalid)
                - added_models: List of added model IDs
                - skipped_paths: List of skipped directory paths
        """
        if not self.model_dir.exists():
            return {
                "scanned": 0,
                "added": 0,
                "skipped": 0,
                "added_models": [],
                "skipped_paths": [],
            }
        
        scanned = 0
        added = 0
        skipped = 0
        added_models = []
        skipped_paths = []
        
        # Get all existing model paths from database
        existing_paths = {
            model.local_path
            for model in self.db.query(DownloadedModel).all()
        }
        
        # Scan directory structure: ./data/models/{org}/{model_name}/
        for org_dir in self.model_dir.iterdir():
            if not org_dir.is_dir():
                continue
            
            # Skip job directories (training outputs)
            if org_dir.name.startswith("job_"):
                continue
            
            org = org_dir.name
            
            # Scan model directories
            for model_dir in org_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                
                model_name = model_dir.name
                model_path_str = str(model_dir)
                scanned += 1
                
                # Skip if already in database
                if model_path_str in existing_paths:
                    skipped += 1
                    skipped_paths.append(model_path_str)
                    continue
                
                # Check if it's a valid model directory (has config.json)
                config_file = model_dir / "config.json"
                if not config_file.exists():
                    skipped += 1
                    skipped_paths.append(model_path_str)
                    continue
                
                # Check for at least one tokenizer file
                tokenizer_files = [
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "tokenizer.model",
                    "vocab.json",
                ]
                has_tokenizer = any((model_dir / f).exists() for f in tokenizer_files)
                
                if not has_tokenizer:
                    skipped += 1
                    skipped_paths.append(model_path_str)
                    continue
                
                # Try to extract model_id from directory structure
                model_id = f"{org}/{model_name}"
                
                # Calculate file size
                try:
                    file_size = sum(
                        f.stat().st_size
                        for f in model_dir.rglob("*")
                        if f.is_file()
                    )
                except Exception:
                    file_size = 0
                
                # Create database record
                try:
                    downloaded_model = DownloadedModel(
                        model_id=model_id,
                        name=model_name,
                        author=org,
                        description=f"Auto-discovered model from {model_path_str}",
                        local_path=model_path_str,
                        file_size=file_size,
                        downloaded_at=datetime.utcnow(),
                        tags=None,
                        model_type=None,
                        is_private=False,
                    )
                    
                    self.db.add(downloaded_model)
                    self.db.commit()
                    self.db.refresh(downloaded_model)
                    
                    added += 1
                    added_models.append(model_id)
                    existing_paths.add(model_path_str)  # Track newly added
                    
                    logger.info(f"Auto-discovered and added model: {model_id}")
                    
                except Exception as e:
                    logger.error(f"Error adding model {model_id}: {e}")
                    self.db.rollback()
                    skipped += 1
                    skipped_paths.append(model_path_str)
                    continue
        
        return {
            "scanned": scanned,
            "added": added,
            "skipped": skipped,
            "added_models": added_models,
            "skipped_paths": skipped_paths,
        }
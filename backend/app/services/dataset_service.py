"""
Dataset service for managing training datasets.

Handles dataset upload, validation, storage, and retrieval.
"""

import csv
import json
import uuid
from io import StringIO
from math import ceil
from pathlib import Path
from typing import Any

from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.dataset import Dataset
from app.schemas.dataset import DatasetCreate, DatasetUpdate


class DatasetService:
    """
    Service for managing datasets.
    
    Handles file upload, validation, storage, and CRUD operations
    for training datasets.
    
    Attributes:
        db: SQLAlchemy database session.
        upload_path: Path to the upload directory.
    """
    
    def __init__(self, db: Session):
        """
        Initialize the dataset service.
        
        Args:
            db: SQLAlchemy database session.
        """
        self.db = db
        self.upload_path = settings.get_upload_path()
    
    async def create_dataset(
        self,
        file: UploadFile,
        dataset_data: DatasetCreate,
    ) -> Dataset:
        """
        Create a new dataset from an uploaded file.
        
        Args:
            file: Uploaded file (CSV or JSON).
            dataset_data: Dataset metadata.
            
        Returns:
            Created Dataset object.
            
        Raises:
            ValueError: If file type is not supported or validation fails.
        """
        # Validate file extension
        filename = file.filename or "unknown"
        file_ext = Path(filename).suffix.lower()
        
        if file_ext not in settings.allowed_extensions:
            raise ValueError(
                f"Unsupported file type: {file_ext}. "
                f"Allowed types: {', '.join(settings.allowed_extensions)}"
            )
        
        # Read file content
        content = await file.read()
        file_size = len(content)
        
        if file_size > settings.max_upload_size:
            raise ValueError(
                f"File too large: {file_size} bytes. "
                f"Maximum allowed: {settings.max_upload_size} bytes"
            )
        
        # Parse and validate content
        try:
            if file_ext == ".csv":
                row_count, column_count, columns = self._parse_csv(content)
                file_type = "csv"
            else:  # .json
                row_count, column_count, columns = self._parse_json(content)
                file_type = "json"
        except Exception as e:
            raise ValueError(f"Failed to parse file: {str(e)}")
        
        # Generate unique filename and save
        unique_filename = f"{uuid.uuid4().hex}{file_ext}"
        file_path = self.upload_path / unique_filename
        file_path.write_bytes(content)
        
        # Create database record
        dataset = Dataset(
            name=dataset_data.name,
            description=dataset_data.description,
            filename=filename,
            file_path=str(file_path),
            file_type=file_type,
            file_size=file_size,
            row_count=row_count,
            column_count=column_count,
            columns=json.dumps(columns) if columns else None,
        )
        
        self.db.add(dataset)
        self.db.commit()
        self.db.refresh(dataset)
        
        return dataset
    
    def _parse_csv(self, content: bytes) -> tuple[int, int, list[str]]:
        """
        Parse CSV content and extract metadata.
        
        Args:
            content: Raw CSV file content.
            
        Returns:
            Tuple of (row_count, column_count, column_names).
        """
        text = content.decode("utf-8")
        reader = csv.reader(StringIO(text))
        
        # Get header row
        try:
            columns = next(reader)
        except StopIteration:
            raise ValueError("CSV file is empty")
        
        column_count = len(columns)
        
        # Count data rows
        row_count = sum(1 for _ in reader)
        
        return row_count, column_count, columns
    
    def _parse_json(self, content: bytes) -> tuple[int, int, list[str]]:
        """
        Parse JSON content and extract metadata.
        
        Args:
            content: Raw JSON file content.
            
        Returns:
            Tuple of (row_count, column_count, column_names).
        """
        text = content.decode("utf-8")
        data = json.loads(text)
        
        if not isinstance(data, list):
            raise ValueError("JSON file must contain an array of objects")
        
        if len(data) == 0:
            return 0, 0, []
        
        if not isinstance(data[0], dict):
            raise ValueError("JSON array must contain objects")
        
        # Get columns from first object
        columns = list(data[0].keys())
        column_count = len(columns)
        row_count = len(data)
        
        return row_count, column_count, columns
    
    def get_dataset(self, dataset_id: int) -> Dataset | None:
        """
        Get a dataset by ID.
        
        Args:
            dataset_id: Dataset ID.
            
        Returns:
            Dataset object or None if not found.
        """
        return self.db.query(Dataset).filter(Dataset.id == dataset_id).first()
    
    def list_datasets(
        self,
        page: int = 1,
        page_size: int = 10,
    ) -> dict[str, Any]:
        """
        List datasets with pagination.
        
        Args:
            page: Page number (1-indexed).
            page_size: Number of items per page.
            
        Returns:
            Dictionary with items, total, page, page_size, and pages.
        """
        query = self.db.query(Dataset).order_by(Dataset.created_at.desc())
        
        total = query.count()
        pages = ceil(total / page_size) if total > 0 else 1
        
        offset = (page - 1) * page_size
        items = query.offset(offset).limit(page_size).all()
        
        return {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": pages,
        }
    
    def update_dataset(
        self,
        dataset_id: int,
        update_data: DatasetUpdate,
    ) -> Dataset | None:
        """
        Update a dataset's metadata.
        
        Args:
            dataset_id: Dataset ID.
            update_data: Fields to update.
            
        Returns:
            Updated Dataset object or None if not found.
        """
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            return None
        
        update_dict = update_data.model_dump(exclude_unset=True)
        for field, value in update_dict.items():
            setattr(dataset, field, value)
        
        self.db.commit()
        self.db.refresh(dataset)
        
        return dataset
    
    def delete_dataset(self, dataset_id: int) -> bool:
        """
        Delete a dataset.
        
        Args:
            dataset_id: Dataset ID.
            
        Returns:
            True if deleted, False if not found.
        """
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            return False
        
        # Delete file from disk
        file_path = Path(dataset.file_path)
        if file_path.exists():
            file_path.unlink()
        
        # Delete database record
        self.db.delete(dataset)
        self.db.commit()
        
        return True


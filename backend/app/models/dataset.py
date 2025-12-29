"""
Dataset database model.

Represents uploaded training datasets (CSV or JSON files).
"""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import String, Integer, DateTime, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.training_job import TrainingJob


class Dataset(Base):
    """
    Dataset model representing an uploaded training dataset.
    
    Attributes:
        id: Unique identifier for the dataset.
        name: Human-readable name for the dataset.
        description: Optional description of the dataset.
        filename: Original filename of the uploaded file.
        file_path: Path where the file is stored on disk.
        file_type: Type of file (csv or json).
        file_size: Size of the file in bytes.
        row_count: Number of rows/records in the dataset.
        column_count: Number of columns/fields in the dataset.
        columns: JSON string of column names.
        created_at: Timestamp when the dataset was created.
        updated_at: Timestamp when the dataset was last updated.
        training_jobs: Related training jobs using this dataset.
    """
    
    __tablename__ = "datasets"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    file_type: Mapped[str] = mapped_column(String(10), nullable=False)  # 'csv' or 'json'
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    column_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    columns: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON string of column names
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    
    # Relationships
    training_jobs: Mapped[list["TrainingJob"]] = relationship(
        "TrainingJob",
        back_populates="dataset",
        cascade="all, delete-orphan",
    )
    
    def __repr__(self) -> str:
        return f"<Dataset(id={self.id}, name='{self.name}', type='{self.file_type}')>"


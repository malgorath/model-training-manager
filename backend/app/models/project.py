"""
Project database models.

Represents project-based training with traits and dataset allocations.
"""

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from sqlalchemy import String, Integer, Float, DateTime, Text, ForeignKey, CheckConstraint, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.dataset import Dataset


class ProjectStatus(str, Enum):
    """Project training status."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TraitType(str, Enum):
    """Types of traits a project can have."""
    
    REASONING = "reasoning"
    CODING = "coding"
    GENERAL_TOOLS = "general_tools"


class Project(Base):
    """
    Project model representing a model training project.
    
    A project defines a new model with specific traits (Reasoning, Coding, General/Tools),
    each using datasets with specified percentages of training data.
    
    Attributes:
        id: Unique identifier for the project.
        name: Human-readable name for the project.
        description: Optional description of the project.
        base_model: Base model identifier (HuggingFace model name).
        training_type: Type of training (qlora, unsloth, rag, standard).
        max_rows: Maximum number of rows to use for training (50K, 100K, 250K, 500K, 1M).
        output_directory: Directory where the trained model will be saved.
        status: Current status of the project.
        progress: Training progress (0-100).
        current_epoch: Current training epoch.
        current_loss: Current training loss.
        error_message: Error message if training failed.
        log: Training logs.
        model_path: Path to the trained model.
        worker_id: ID of the worker processing this project.
        started_at: Timestamp when training started.
        completed_at: Timestamp when training completed.
        created_at: Timestamp when the project was created.
        updated_at: Timestamp when the project was last updated.
        traits: Related project traits.
    """
    
    __tablename__ = "projects"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    base_model: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    training_type: Mapped[str] = mapped_column(String(20), nullable=False, default="qlora")
    max_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    output_directory: Mapped[str] = mapped_column(String(512), nullable=False)
    
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=ProjectStatus.PENDING.value,
        index=True,
    )
    
    # Progress tracking
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    current_epoch: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    current_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Training logs and output
    log: Mapped[str | None] = mapped_column(Text, nullable=True, default="")
    model_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    
    # Worker assignment
    worker_id: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    
    # Timestamps
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
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
    traits: Mapped[list["ProjectTrait"]] = relationship(
        "ProjectTrait",
        back_populates="project",
        cascade="all, delete-orphan",
    )
    
    def __repr__(self) -> str:
        return f"<Project(id={self.id}, name='{self.name}', status='{self.status}')>"


class ProjectTrait(Base):
    """
    Project trait model linking traits to projects.
    
    A project can have multiple traits: Reasoning, Coding, and General/Tools.
    Each trait uses one or more datasets with specified percentages.
    
    Attributes:
        id: Unique identifier for the trait.
        project_id: Foreign key to the project.
        project: Related project object.
        trait_type: Type of trait (reasoning, coding, general_tools).
        datasets: Related dataset allocations for this trait.
    """
    
    __tablename__ = "project_traits"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    trait_type: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    
    # Relationships
    project: Mapped["Project"] = relationship("Project", back_populates="traits")
    datasets: Mapped[list["ProjectTraitDataset"]] = relationship(
        "ProjectTraitDataset",
        back_populates="trait",
        cascade="all, delete-orphan",
    )
    
    def __repr__(self) -> str:
        return f"<ProjectTrait(id={self.id}, project_id={self.project_id}, type='{self.trait_type}')>"


class ProjectTraitDataset(Base):
    """
    Junction table linking project traits to datasets with percentages.
    
    Defines which datasets are used for each trait and what percentage
    of the overall training data each dataset represents.
    
    Attributes:
        id: Unique identifier for the allocation.
        project_trait_id: Foreign key to the project trait.
        trait: Related project trait object.
        dataset_id: Foreign key to the dataset.
        dataset: Related dataset object.
        percentage: Percentage of training data (0-100) this dataset represents.
    """
    
    __tablename__ = "project_trait_datasets"
    
    __table_args__ = (
        # Ensure percentage is between 0 and 100
        CheckConstraint('percentage >= 0.0 AND percentage <= 100.0', name='check_percentage_range'),
        # Ensure a dataset is not used twice in the same trait
        UniqueConstraint('project_trait_id', 'dataset_id', name='unique_trait_dataset'),
    )
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_trait_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("project_traits.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    dataset_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    percentage: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Relationships
    trait: Mapped["ProjectTrait"] = relationship("ProjectTrait", back_populates="datasets")
    dataset: Mapped["Dataset"] = relationship("Dataset")
    
    def __repr__(self) -> str:
        return f"<ProjectTraitDataset(id={self.id}, trait_id={self.project_trait_id}, dataset_id={self.dataset_id}, percentage={self.percentage})>"

"""
Training job database model.

Represents a model training job with its configuration and status.
"""

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from sqlalchemy import String, Integer, DateTime, Text, Float, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.dataset import Dataset


class TrainingStatus(str, Enum):
    """Training job status enumeration."""
    
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingType(str, Enum):
    """Training type enumeration."""
    
    QLORA = "qlora"
    UNSLOTH = "unsloth"
    RAG = "rag"
    STANDARD = "standard"


class TrainingJob(Base):
    """
    Training job model representing a model training task.
    
    Attributes:
        id: Unique identifier for the training job.
        name: Human-readable name for the job.
        description: Optional description of the job.
        status: Current status of the training job.
        training_type: Type of training (qlora, rag, standard).
        model_name: Name of the model being trained.
        
        dataset_id: Foreign key to the dataset used for training.
        dataset: Related dataset object.
        
        batch_size: Training batch size.
        learning_rate: Learning rate for training.
        epochs: Number of training epochs.
        lora_r: LoRA rank parameter.
        lora_alpha: LoRA alpha parameter.
        lora_dropout: LoRA dropout rate.
        
        progress: Training progress (0-100).
        current_epoch: Current training epoch.
        current_loss: Current training loss.
        error_message: Error message if training failed.
        
        worker_id: ID of the worker processing this job.
        
        started_at: Timestamp when training started.
        completed_at: Timestamp when training completed.
        created_at: Timestamp when the job was created.
        updated_at: Timestamp when the job was last updated.
    """
    
    __tablename__ = "training_jobs"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=TrainingStatus.PENDING.value,
        index=True,
    )
    training_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=TrainingType.QLORA.value,
    )
    model_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default="llama3.2:3b",
    )
    
    # Dataset relationship
    dataset_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="training_jobs")
    
    # Training parameters
    batch_size: Mapped[int] = mapped_column(Integer, nullable=False, default=4)
    learning_rate: Mapped[float] = mapped_column(Float, nullable=False, default=2e-4)
    epochs: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    lora_r: Mapped[int] = mapped_column(Integer, nullable=False, default=16)
    lora_alpha: Mapped[int] = mapped_column(Integer, nullable=False, default=32)
    lora_dropout: Mapped[float] = mapped_column(Float, nullable=False, default=0.05)
    
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
    
    def __repr__(self) -> str:
        return f"<TrainingJob(id={self.id}, name='{self.name}', status='{self.status}')>"


"""
Training configuration database model.

Stores global training configuration settings.
"""

from datetime import datetime

from sqlalchemy import String, Integer, DateTime, Float, Boolean, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class TrainingConfig(Base):
    """
    Training configuration model for storing global settings.
    
    This is a singleton-like table that stores the current training
    configuration. Only one row should exist at any time.
    
    Attributes:
        id: Unique identifier (should always be 1).
        max_concurrent_workers: Maximum number of concurrent training workers.
        active_workers: Current number of active workers.
        default_model: Default model to use for training.
        default_training_type: Default training type.
        
        default_batch_size: Default batch size for training.
        default_learning_rate: Default learning rate.
        default_epochs: Default number of epochs.
        default_lora_r: Default LoRA rank.
        default_lora_alpha: Default LoRA alpha.
        default_lora_dropout: Default LoRA dropout.
        
        auto_start_workers: Whether to auto-start workers on application startup.
        
        model_provider: Model provider type ("ollama" or "lm_studio").
        model_api_url: Base URL for the model API server.
        
        created_at: Timestamp when config was created.
        updated_at: Timestamp when config was last updated.
    """
    
    __tablename__ = "training_config"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    
    # Worker settings
    max_concurrent_workers: Mapped[int] = mapped_column(Integer, nullable=False, default=4)
    active_workers: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    
    # Model settings
    default_model: Mapped[str] = mapped_column(String(100), nullable=False, default="llama3.2:3b")
    default_training_type: Mapped[str] = mapped_column(String(20), nullable=False, default="qlora")
    
    # Training parameters
    default_batch_size: Mapped[int] = mapped_column(Integer, nullable=False, default=4)
    default_learning_rate: Mapped[float] = mapped_column(Float, nullable=False, default=2e-4)
    default_epochs: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    default_lora_r: Mapped[int] = mapped_column(Integer, nullable=False, default=16)
    default_lora_alpha: Mapped[int] = mapped_column(Integer, nullable=False, default=32)
    default_lora_dropout: Mapped[float] = mapped_column(Float, nullable=False, default=0.05)
    
    # Behavior settings
    auto_start_workers: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    
    # Model API settings
    model_provider: Mapped[str] = mapped_column(String(20), nullable=False, default="ollama")
    model_api_url: Mapped[str] = mapped_column(String(500), nullable=False, default="http://localhost:11434")
    
    # Directory settings
    output_directory_base: Mapped[str | None] = mapped_column(String(512), nullable=True, default=None)
    model_cache_path: Mapped[str | None] = mapped_column(String(512), nullable=True, default=None)
    
    # Timestamps
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
        return f"<TrainingConfig(id={self.id}, max_workers={self.max_concurrent_workers})>"


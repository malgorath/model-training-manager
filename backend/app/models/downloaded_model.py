"""
Downloaded model database model.

Tracks models downloaded from HuggingFace Hub and stored locally.
"""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import String, Integer, DateTime, BigInteger, Boolean, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base

if TYPE_CHECKING:
    pass


class DownloadedModel(Base):
    """
    Model for tracking downloaded HuggingFace models.
    
    Attributes:
        id: Unique identifier.
        model_id: HuggingFace model ID (e.g., "meta-llama/Llama-3.2-3B-Instruct").
        name: Display name (last part of model_id).
        author: Model author/organization (first part of model_id).
        description: Model description from HuggingFace.
        local_path: Path to downloaded model directory.
        file_size: Total size of downloaded model in bytes.
        downloaded_at: Timestamp when model was downloaded.
        tags: JSON array of model tags.
        model_type: Model type (e.g., "text-generation", "text2text-generation").
        is_private: Whether model requires authentication.
        created_at: Timestamp when record was created.
        updated_at: Timestamp when record was last updated.
    """
    
    __tablename__ = "downloaded_models"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    
    # Model identification
    model_id: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    author: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Model metadata
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON array as string
    model_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    is_private: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    
    # Storage information
    local_path: Mapped[str] = mapped_column(String(512), nullable=False)
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    downloaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    
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
        return f"<DownloadedModel(id={self.id}, model_id='{self.model_id}')>"

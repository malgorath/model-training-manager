"""
Application configuration module.

Manages all configuration settings using Pydantic settings management.
Configuration can be overridden via environment variables.
"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings.
    
    All settings can be overridden via environment variables.
    Environment variables should be prefixed with the appropriate prefix
    or match the field name exactly.
    
    Attributes:
        app_name: Name of the application.
        app_version: Current version of the application.
        debug: Enable debug mode.
        environment: Current environment (development, staging, production).
        
        database_url: SQLite database connection URL.
        
        ollama_base_url: Base URL for Ollama API server.
        ollama_model: Default model to use for training.
        ollama_timeout: Timeout in seconds for Ollama API requests.
        
        max_workers: Maximum number of concurrent training workers.
        default_workers: Default number of workers to start.
        
        upload_dir: Directory for storing uploaded datasets.
        max_upload_size: Maximum upload file size in bytes (default 100MB).
        allowed_extensions: Allowed file extensions for uploads.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Application settings
    app_name: str = "Model Training Manager"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: Literal["development", "staging", "production"] = "development"
    
    # Database settings
    database_url: str = "sqlite:///./trainers.db"
    
    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout: int = 300
    
    # Worker settings
    max_workers: int = 8
    default_workers: int = 2
    
    # File upload settings
    upload_dir: Path = Path("./data")
    max_upload_size: int = 100 * 1024 * 1024  # 100MB
    allowed_extensions: set[str] = {".csv", ".json"}
    
    # Model storage settings
    model_dir: Path = Path("./data/models")
    archive_dir: Path = Path("./data/archives")
    
    # Training settings
    default_batch_size: int = 4
    default_learning_rate: float = 2e-4
    default_epochs: int = 3
    default_lora_r: int = 16
    default_lora_alpha: int = 32
    default_lora_dropout: float = 0.05
    
    def get_upload_path(self) -> Path:
        """Get and ensure upload directory exists."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        return self.upload_dir
    
    @staticmethod
    def _sanitize_name(name: str) -> str:
        """
        Sanitize a name for filesystem compatibility.
        
        Removes or replaces characters that are not safe for filesystem paths.
        
        Args:
            name: Name to sanitize.
            
        Returns:
            Sanitized name safe for use in filesystem paths.
        """
        import re
        # Replace invalid filesystem characters with underscores
        # Invalid chars: < > : " / \ | ? *
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
        # Remove leading/trailing dots and spaces
        sanitized = sanitized.strip('. ')
        # Replace multiple consecutive underscores with single underscore
        sanitized = re.sub(r'_+', '_', sanitized)
        # Ensure it's not empty
        if not sanitized:
            sanitized = "unnamed"
        return sanitized
    
    def get_dataset_path(self, author: str, dataset_name: str) -> Path:
        """
        Get the path for a dataset in the structure: ./data/{author}/{datasetname}.
        
        Creates the directory structure if it doesn't exist.
        
        Args:
            author: Author/organization name (e.g., "user", "meta-llama").
            dataset_name: Name of the dataset.
            
        Returns:
            Path to the dataset directory.
        """
        sanitized_author = self._sanitize_name(author)
        sanitized_dataset = self._sanitize_name(dataset_name)
        dataset_path = self.upload_dir / sanitized_author / sanitized_dataset
        dataset_path.mkdir(parents=True, exist_ok=True)
        return dataset_path
    
    def get_model_path(self) -> Path:
        """Get and ensure model directory exists."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        return self.model_dir
    
    def get_archive_path(self) -> Path:
        """Get and ensure archive directory exists."""
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        return self.archive_dir


settings = Settings()


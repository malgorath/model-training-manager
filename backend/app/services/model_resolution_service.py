"""
Model Resolution Service.

Resolves model paths by checking HuggingFace cache, local overrides, and Ollama directories.
Validates model format and ensures models are available before training.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ModelResolutionError(Exception):
    """Base exception for model resolution errors."""
    pass


class ModelNotFoundError(ModelResolutionError):
    """Raised when a model cannot be found in any location."""
    pass


class ModelFormatError(ModelResolutionError):
    """Raised when a model format is invalid or incomplete."""
    pass


class ModelResolutionService:
    """
    Service for resolving and validating model paths.
    
    Checks multiple locations in order:
    1. Model path overrides (explicit user mappings)
    2. HuggingFace cache directory
    3. Ollama models directory (for reference, but GGUF format not suitable for training)
    
    Attributes:
        cache_base_path: Base path for HuggingFace model cache.
        ollama_models_path: Path to Ollama models directory.
        model_overrides: Dictionary mapping model names to local paths.
    """
    
    def __init__(
        self,
        cache_base_path: Optional[str] = None,
        ollama_models_path: Optional[str] = None,
    ):
        """
        Initialize the model resolution service.
        
        Args:
            cache_base_path: Base path for HuggingFace cache. Defaults to ~/.cache/huggingface/hub
            ollama_models_path: Path to Ollama models. Defaults to ~/.ollama/models
        """
        if cache_base_path is None:
            cache_base_path = Path.home() / ".cache" / "huggingface" / "hub"
        if ollama_models_path is None:
            ollama_models_path = Path.home() / ".ollama" / "models"
        
        self.cache_base_path = Path(cache_base_path)
        self.ollama_models_path = Path(ollama_models_path)
        self.model_overrides: dict[str, str] = {}
    
    def set_model_override(self, model_name: str, local_path: str) -> None:
        """
        Set a local path override for a specific model.
        
        Args:
            model_name: The model identifier (e.g., "meta-llama/Llama-3.2-3B-Instruct").
            local_path: Local filesystem path to the model.
        """
        self.model_overrides[model_name] = local_path
        logger.info(f"Set model override: {model_name} -> {local_path}")
    
    def resolve_model_path(self, model_name: str) -> str:
        """
        Resolve the filesystem path for a model.
        
        Checks in order:
        1. Model overrides
        2. HuggingFace cache
        
        Args:
            model_name: Model identifier (e.g., "meta-llama/Llama-3.2-3B-Instruct").
            
        Returns:
            Absolute path to the model directory.
            
        Raises:
            ModelNotFoundError: If model cannot be found in any location.
        """
        # Check overrides first
        if model_name in self.model_overrides:
            override_path = Path(self.model_overrides[model_name])
            if override_path.exists():
                logger.info(f"Using model override: {model_name} -> {override_path}")
                return str(override_path.absolute())
            else:
                logger.warning(f"Model override path does not exist: {override_path}")
        
        # Check HuggingFace cache
        hf_path = self._check_huggingface_cache(model_name)
        if hf_path:
            logger.info(f"Found model in HuggingFace cache: {model_name} -> {hf_path}")
            return str(hf_path)
        
        # Model not found
        raise ModelNotFoundError(
            f"Model '{model_name}' not found. "
            "Please ensure the model is downloaded to HuggingFace cache or configure a local path override."
        )
    
    def _check_huggingface_cache(self, model_name: str) -> Optional[Path]:
        """
        Check if model exists in HuggingFace cache.
        
        HuggingFace cache structure:
        ~/.cache/huggingface/hub/models--{org}--{model}/snapshots/{hash}/
        
        Args:
            model_name: Model identifier (e.g., "meta-llama/Llama-3.2-3B-Instruct").
            
        Returns:
            Path to model snapshot if found, None otherwise.
        """
        if not self.cache_base_path.exists():
            return None
        
        # Convert model name to cache directory format
        # "meta-llama/Llama-3.2-3B-Instruct" -> "models--meta-llama--Llama-3.2-3B-Instruct"
        cache_dir_name = f"models--{model_name.replace('/', '--')}"
        model_cache_dir = self.cache_base_path / cache_dir_name
        
        if not model_cache_dir.exists():
            return None
        
        # Check snapshots directory
        snapshots_dir = model_cache_dir / "snapshots"
        if not snapshots_dir.exists():
            return None
        
        # Find the first valid snapshot
        for snapshot_dir in snapshots_dir.iterdir():
            if snapshot_dir.is_dir():
                # Check if it has required files
                if self._is_valid_model_directory(snapshot_dir):
                    return snapshot_dir
        
        return None
    
    def _is_valid_model_directory(self, model_path: Path) -> bool:
        """
        Check if a directory contains a valid HuggingFace model.
        
        Args:
            model_path: Path to check.
            
        Returns:
            True if directory appears to contain a valid model.
        """
        # At minimum, should have config.json
        config_file = model_path / "config.json"
        if not config_file.exists():
            return False
        
        # Should have at least one tokenizer file
        tokenizer_files = [
            model_path / "tokenizer.json",
            model_path / "tokenizer_config.json",
            model_path / "tokenizer.model",
        ]
        if not any(f.exists() for f in tokenizer_files):
            return False
        
        return True
    
    def validate_model_format(self, model_path: str) -> None:
        """
        Validate that a model directory contains all required files.
        
        Required files:
        - config.json (model configuration)
        - tokenizer files (at least one of tokenizer.json, tokenizer_config.json, tokenizer.model)
        
        Args:
            model_path: Path to model directory.
            
        Raises:
            ModelFormatError: If model format is invalid.
        """
        path = Path(model_path)
        
        if not path.exists():
            raise ModelFormatError(f"Model path does not exist: {model_path}")
        
        if not path.is_dir():
            raise ModelFormatError(f"Model path is not a directory: {model_path}")
        
        # Check config.json
        config_file = path / "config.json"
        if not config_file.exists():
            raise ModelFormatError(
                f"Model format invalid: missing config.json in {model_path}. "
                "Expected HuggingFace format model."
            )
        
        # Validate JSON syntax
        try:
            with open(config_file, 'r') as f:
                json.load(f)
        except json.JSONDecodeError as e:
            raise ModelFormatError(
                f"Model format invalid: config.json contains invalid JSON: {e}"
            )
        
        # Check for tokenizer files
        tokenizer_files = [
            path / "tokenizer.json",
            path / "tokenizer_config.json",
            path / "tokenizer.model",
        ]
        if not any(f.exists() for f in tokenizer_files):
            raise ModelFormatError(
                f"Model format invalid: missing tokenizer files in {model_path}. "
                "Expected at least one of: tokenizer.json, tokenizer_config.json, tokenizer.model"
            )
        
        logger.info(f"Model format validated successfully: {model_path}")
    
    def is_model_available(self, model_name: str) -> bool:
        """
        Check if a model is available locally.
        
        Args:
            model_name: Model identifier.
            
        Returns:
            True if model is available, False otherwise.
        """
        try:
            resolved_path = self.resolve_model_path(model_name)
            self.validate_model_format(resolved_path)
            return True
        except (ModelNotFoundError, ModelFormatError):
            return False
    
    def list_available_models(self) -> list[str]:
        """
        List all models available in the HuggingFace cache.
        
        Returns:
            List of model identifiers found in cache.
        """
        available = []
        
        if not self.cache_base_path.exists():
            return available
        
        # Scan for model directories
        for item in self.cache_base_path.iterdir():
            if item.is_dir() and item.name.startswith("models--"):
                # Extract model name from directory name
                # "models--meta-llama--Llama-3.2-3B-Instruct" -> "meta-llama/Llama-3.2-3B-Instruct"
                model_name = item.name.replace("models--", "").replace("--", "/")
                
                # Check if it has valid snapshots
                snapshots_dir = item / "snapshots"
                if snapshots_dir.exists():
                    for snapshot in snapshots_dir.iterdir():
                        if snapshot.is_dir() and self._is_valid_model_directory(snapshot):
                            available.append(model_name)
                            break
        
        return available

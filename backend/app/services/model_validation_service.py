"""
Model Validation Service.

Validates trained models to ensure all files are present and working correctly.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class ModelValidationError(Exception):
    """Raised when model validation fails."""
    pass


class ModelValidationService:
    """
    Service for validating trained models.
    
    Performs comprehensive checks to ensure model files are complete,
    loadable, and functional.
    """
    
    REQUIRED_FILES = [
        "config.json",
        "tokenizer.json",  # At least one tokenizer file required
    ]
    
    OPTIONAL_TOKENIZER_FILES = [
        "tokenizer_config.json",
        "tokenizer.model",
        "vocab.json",
    ]
    
    def validate_model_files(self, model_path: str) -> Dict[str, Any]:
        """
        Validate that all required model files exist.
        
        Args:
            model_path: Path to the model directory.
            
        Returns:
            Dictionary with validation results:
            - valid: Boolean indicating if all files are present.
            - files_checked: List of files that were checked.
            - files_missing: List of missing required files.
            - errors: List of error messages.
        """
        errors: List[str] = []
        files_checked = []
        files_missing = []
        
        path = Path(model_path).expanduser()
        
        if not path.exists():
            errors.append(f"Model path does not exist: {model_path}")
            return {
                "valid": False,
                "files_checked": [],
                "files_missing": [],
                "errors": errors,
            }
        
        if not path.is_dir():
            errors.append(f"Model path is not a directory: {model_path}")
            return {
                "valid": False,
                "files_checked": [],
                "files_missing": [],
                "errors": errors,
            }
        
        # Check if this is a RAG model (has rag_config.json)
        rag_config_file = path / "rag_config.json"
        if rag_config_file.exists():
            # Validate RAG model files
            return self._validate_rag_model_files(path)
        
        # Standard transformers model validation
        # Check for config.json
        config_file = path / "config.json"
        files_checked.append("config.json")
        if not config_file.exists():
            files_missing.append("config.json")
            errors.append("Missing required file: config.json")
        else:
            # Validate JSON syntax
            try:
                with open(config_file, 'r') as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                errors.append(f"Invalid JSON in config.json: {str(e)}")
        
        # Check for at least one tokenizer file
        tokenizer_files_found = []
        for tokenizer_file in self.REQUIRED_FILES[1:] + self.OPTIONAL_TOKENIZER_FILES:
            tokenizer_path = path / tokenizer_file
            if tokenizer_path.exists():
                files_checked.append(tokenizer_file)
                tokenizer_files_found.append(tokenizer_file)
        
        if not tokenizer_files_found:
            errors.append(
                "Missing tokenizer files. Expected at least one of: "
                f"{', '.join(self.REQUIRED_FILES[1:] + self.OPTIONAL_TOKENIZER_FILES)}"
            )
            files_missing.extend(["tokenizer files"])
        
        # Check for model weights (optional but recommended)
        weight_files = list(path.glob("*.safetensors")) + list(path.glob("*.bin"))
        if not weight_files:
            logger.warning(f"No model weight files found in {model_path}")
        
        valid = len(errors) == 0
        
        return {
            "valid": valid,
            "files_checked": files_checked,
            "files_missing": files_missing,
            "errors": errors,
        }
    
    def _validate_rag_model_files(self, path: Path) -> Dict[str, Any]:
        """
        Validate RAG model files.
        
        RAG models require:
        - rag_config.json
        - vector_index.faiss
        - documents.json
        - embeddings_metadata.json (optional)
        
        Args:
            path: Path to the RAG model directory.
            
        Returns:
            Dictionary with validation results.
        """
        errors: List[str] = []
        files_checked = []
        files_missing = []
        
        # Required RAG files
        required_rag_files = {
            "rag_config.json": "RAG configuration file",
            "vector_index.faiss": "FAISS vector index",
            "documents.json": "Document corpus",
        }
        
        for filename, description in required_rag_files.items():
            file_path = path / filename
            files_checked.append(filename)
            if not file_path.exists():
                files_missing.append(filename)
                errors.append(f"Missing required RAG file: {filename} ({description})")
            else:
                # Validate JSON files
                if filename.endswith(".json"):
                    try:
                        with open(file_path, 'r') as f:
                            json.load(f)
                    except json.JSONDecodeError as e:
                        errors.append(f"Invalid JSON in {filename}: {str(e)}")
        
        # Optional files
        optional_files = ["embeddings_metadata.json"]
        for filename in optional_files:
            file_path = path / filename
            if file_path.exists():
                files_checked.append(filename)
                if filename.endswith(".json"):
                    try:
                        with open(file_path, 'r') as f:
                            json.load(f)
                    except json.JSONDecodeError as e:
                        errors.append(f"Invalid JSON in {filename}: {str(e)}")
        
        valid = len(errors) == 0
        
        return {
            "valid": valid,
            "files_checked": files_checked,
            "files_missing": files_missing,
            "errors": errors,
        }
    
    def validate_model_loading(self, model_path: str) -> Dict[str, Any]:
        """
        Validate that the model can be loaded successfully.
        
        Args:
            model_path: Path to the model directory.
            
        Returns:
            Dictionary with validation results:
            - valid: Boolean indicating if model loads successfully.
            - errors: List of error messages.
        """
        errors: List[str] = []
        
        path = Path(model_path).expanduser()
        
        # Check if this is a RAG model (has rag_config.json)
        rag_config_file = path / "rag_config.json"
        if rag_config_file.exists():
            # Validate RAG model loading (check FAISS index can be loaded)
            return self._validate_rag_model_loading(path)
        
        # Standard transformers model validation
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Try to load the model
            try:
                # Load config first to validate it
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(
                    str(path),
                    trust_remote_code=True,
                )
                # Ensure config has model_type attribute
                if not hasattr(config, 'model_type'):
                    # If config is a dict, convert it properly
                    if isinstance(config, dict):
                        # This shouldn't happen, but handle it gracefully
                        logger.warning(f"Config loaded as dict instead of Config object for {model_path}")
                        # Try to reload with proper config class
                        config = AutoConfig.from_pretrained(
                            str(path),
                            trust_remote_code=True,
                            _from_auto=True,
                        )
                
                # Now load the model
                model = AutoModelForCausalLM.from_pretrained(
                    str(path),
                    trust_remote_code=True,
                    config=config,
                )
                del model  # Free memory
                del config  # Free memory
                logger.info(f"Successfully loaded model from {model_path}")
            except Exception as e:
                error_msg = f"Failed to load model from {model_path}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
            
            # Try to load the tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    str(path),
                    trust_remote_code=True,
                )
                del tokenizer  # Free memory
                logger.info(f"Successfully loaded tokenizer from {model_path}")
            except Exception as e:
                error_msg = f"Failed to load tokenizer: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
                
        except ImportError:
            error_msg = "transformers library not available. Cannot validate model loading."
            errors.append(error_msg)
            logger.warning(error_msg)
        
        valid = len(errors) == 0
        
        return {
            "valid": valid,
            "errors": errors,
        }
    
    def _validate_rag_model_loading(self, path: Path) -> Dict[str, Any]:
        """
        Validate RAG model loading (check FAISS index can be loaded).
        
        Args:
            path: Path to the RAG model directory.
            
        Returns:
            Dictionary with validation results.
        """
        errors: List[str] = []
        
        # Try to load FAISS index
        try:
            import faiss
            index_path = path / "vector_index.faiss"
            if index_path.exists():
                index = faiss.read_index(str(index_path))
                logger.info(f"Successfully loaded FAISS index from {index_path} ({index.ntotal} vectors)")
                del index
            else:
                errors.append("FAISS index file not found: vector_index.faiss")
        except ImportError:
            errors.append("faiss library not available. Cannot validate FAISS index loading.")
            logger.warning("faiss library not available")
        except Exception as e:
            error_msg = f"Failed to load FAISS index: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
        
        # Validate documents.json can be loaded
        try:
            documents_path = path / "documents.json"
            if documents_path.exists():
                with open(documents_path, 'r') as f:
                    documents = json.load(f)
                logger.info(f"Successfully loaded documents.json ({len(documents)} documents)")
            else:
                errors.append("documents.json not found")
        except Exception as e:
            error_msg = f"Failed to load documents.json: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
        
        valid = len(errors) == 0
        
        return {
            "valid": valid,
            "errors": errors,
        }
    
    def validate_model_complete(self, model_path: str) -> Dict[str, Any]:
        """
        Perform complete model validation (files + loading).
        
        Args:
            model_path: Path to the model directory.
            
        Returns:
            Dictionary with complete validation results.
        """
        file_validation = self.validate_model_files(model_path)
        loading_validation = self.validate_model_loading(model_path)
        
        valid = file_validation["valid"] and loading_validation["valid"]
        all_errors = file_validation.get("errors", []) + loading_validation.get("errors", [])
        
        return {
            "valid": valid,
            "files": file_validation,
            "loading": loading_validation,
            "errors": all_errors,
        }

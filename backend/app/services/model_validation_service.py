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
        
        path = Path(model_path)
        
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
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Try to load the model
            try:
                # Load config first to validate it
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(
                    model_path,
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
                            model_path,
                            trust_remote_code=True,
                            _from_auto=True,
                        )
                
                # Now load the model
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
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
                    model_path,
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

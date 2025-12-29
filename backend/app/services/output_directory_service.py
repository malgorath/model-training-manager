"""
Output Directory Service.

Validates and manages output directories for trained models.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class DirectoryValidationError(Exception):
    """Raised when directory validation fails."""
    pass


class OutputDirectoryService:
    """
    Service for validating and managing output directories.
    
    Ensures directories exist, are writable, and have proper permissions
    before accepting them for model output.
    """
    
    def validate_directory(
        self,
        directory_path: str,
        create_if_missing: bool = False,
    ) -> Dict[str, Any]:
        """
        Validate that a directory exists and is writable.
        
        Args:
            directory_path: Path to the directory to validate.
            create_if_missing: If True, create the directory if it doesn't exist.
            
        Returns:
            Dictionary with validation results:
            - valid: Boolean indicating if directory is valid.
            - writable: Boolean indicating if directory is writable.
            - path: Resolved absolute path to the directory.
            - errors: List of error messages if validation failed.
            
        Raises:
            DirectoryValidationError: If directory cannot be validated or created.
        """
        errors = []
        path = Path(directory_path).expanduser().resolve()
        
        # Check if directory exists
        if not path.exists():
            if create_if_missing:
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created output directory: {path}")
                except OSError as e:
                    error_msg = f"Cannot create directory {path}: {str(e)}"
                    logger.error(error_msg)
                    raise DirectoryValidationError(error_msg) from e
            else:
                error_msg = f"Directory does not exist: {path}"
                errors.append(error_msg)
                raise DirectoryValidationError(error_msg)
        
        # Check if it's a directory
        if not path.is_dir():
            error_msg = f"Path exists but is not a directory: {path}"
            errors.append(error_msg)
            raise DirectoryValidationError(error_msg)
        
        # Check if directory is writable
        writable = os.access(path, os.W_OK)
        if not writable:
            error_msg = f"Directory is not writable: {path}. Please check permissions."
            errors.append(error_msg)
            raise DirectoryValidationError(error_msg)
        
        # Try to create a test file to confirm write access
        test_file = path / ".write_test"
        try:
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            error_msg = f"Cannot write to directory {path}: {str(e)}"
            errors.append(error_msg)
            raise DirectoryValidationError(error_msg) from e
        
        return {
            "valid": True,
            "writable": True,
            "path": str(path),
            "errors": [],
        }

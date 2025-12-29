"""
Tests for the Model Resolution Service.

Tests model path resolution, local cache checking, and HuggingFace model discovery.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import shutil

from app.services.model_resolution_service import (
    ModelResolutionService,
    ModelNotFoundError,
    ModelFormatError,
)


class TestModelResolutionService:
    """Tests for ModelResolutionService."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def service(self, temp_cache_dir):
        """Create a ModelResolutionService instance for testing."""
        with patch('app.services.model_resolution_service.Path.home', return_value=temp_cache_dir):
            service = ModelResolutionService(
                cache_base_path=str(temp_cache_dir / ".cache" / "huggingface" / "hub"),
                ollama_models_path=str(temp_cache_dir / ".ollama" / "models"),
            )
            return service
    
    def test_resolve_huggingface_model_in_cache(self, service, temp_cache_dir):
        """Test resolving a model that exists in HuggingFace cache."""
        # Create mock HuggingFace cache structure
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
        cache_path = temp_cache_dir / ".cache" / "huggingface" / "hub"
        model_cache_dir = cache_path / f"models--{model_name.replace('/', '--')}"
        snapshots_dir = model_cache_dir / "snapshots" / "abc123"
        snapshots_dir.mkdir(parents=True)
        
        # Create required model files
        (snapshots_dir / "config.json").write_text('{"model_type": "llama"}')
        (snapshots_dir / "tokenizer.json").write_text("{}")
        
        resolved_path = service.resolve_model_path(model_name)
        
        assert resolved_path == str(snapshots_dir)
        assert Path(resolved_path).exists()
    
    def test_resolve_model_not_found(self, service):
        """Test that ModelNotFoundError is raised when model doesn't exist."""
        model_name = "nonexistent/model"
        
        with pytest.raises(ModelNotFoundError) as exc_info:
            service.resolve_model_path(model_name)
        
        assert model_name in str(exc_info.value)
    
    def test_resolve_model_with_local_override(self, service, temp_cache_dir):
        """Test resolving a model using a local path override."""
        local_path = temp_cache_dir / "custom" / "models" / "my-model"
        local_path.mkdir(parents=True)
        (local_path / "config.json").write_text('{"model_type": "llama"}')
        
        service.set_model_override("meta-llama/Llama-3.2-3B-Instruct", str(local_path))
        
        resolved_path = service.resolve_model_path("meta-llama/Llama-3.2-3B-Instruct")
        
        assert resolved_path == str(local_path)
    
    def test_validate_model_format_valid(self, service, temp_cache_dir):
        """Test validating a valid HuggingFace model format."""
        model_path = temp_cache_dir / "valid-model"
        model_path.mkdir()
        
        # Create required files
        (model_path / "config.json").write_text('{"model_type": "llama"}')
        (model_path / "tokenizer.json").write_text("{}")
        (model_path / "tokenizer_config.json").write_text("{}")
        
        # Should not raise
        service.validate_model_format(str(model_path))
    
    def test_validate_model_format_missing_config(self, service, temp_cache_dir):
        """Test that ModelFormatError is raised when config.json is missing."""
        model_path = temp_cache_dir / "invalid-model"
        model_path.mkdir()
        
        with pytest.raises(ModelFormatError) as exc_info:
            service.validate_model_format(str(model_path))
        
        assert "config.json" in str(exc_info.value).lower()
    
    def test_validate_model_format_missing_tokenizer(self, service, temp_cache_dir):
        """Test that ModelFormatError is raised when tokenizer files are missing."""
        model_path = temp_cache_dir / "missing-tokenizer"
        model_path.mkdir()
        (model_path / "config.json").write_text('{"model_type": "llama"}')
        
        with pytest.raises(ModelFormatError) as exc_info:
            service.validate_model_format(str(model_path))
        
        assert "tokenizer" in str(exc_info.value).lower()
    
    def test_check_model_available_true(self, service, temp_cache_dir):
        """Test checking if a model is available locally."""
        model_name = "test/model"
        cache_path = temp_cache_dir / ".cache" / "huggingface" / "hub"
        model_cache_dir = cache_path / f"models--{model_name.replace('/', '--')}"
        snapshots_dir = model_cache_dir / "snapshots" / "abc123"
        snapshots_dir.mkdir(parents=True)
        (snapshots_dir / "config.json").write_text('{"model_type": "llama"}')
        (snapshots_dir / "tokenizer.json").write_text("{}")
        
        assert service.is_model_available(model_name) is True
    
    def test_check_model_available_false(self, service):
        """Test checking if a model is NOT available locally."""
        model_name = "nonexistent/model"
        
        assert service.is_model_available(model_name) is False
    
    def test_list_available_models(self, service, temp_cache_dir):
        """Test listing all available models in cache."""
        cache_path = temp_cache_dir / ".cache" / "huggingface" / "hub"
        
        # Create multiple model caches
        for model_name in ["model1/model", "model2/model", "model3/model"]:
            model_cache_dir = cache_path / f"models--{model_name.replace('/', '--')}"
            snapshots_dir = model_cache_dir / "snapshots" / "abc123"
            snapshots_dir.mkdir(parents=True)
            (snapshots_dir / "config.json").write_text('{"model_type": "llama"}')
            (snapshots_dir / "tokenizer.json").write_text("{}")
        
        available = service.list_available_models()
        
        assert len(available) == 3
        assert all("model" in m for m in available)
    
    def test_resolve_model_with_default_cache(self):
        """Test service uses default HuggingFace cache location."""
        with patch('app.services.model_resolution_service.Path.home', return_value=Path("/home/user")):
            service = ModelResolutionService()
            
            # Should use default paths
            assert "huggingface" in str(service.cache_base_path)
            assert "hub" in str(service.cache_base_path)
    
    def test_validate_model_format_invalid_json(self, service, temp_cache_dir):
        """Test that invalid JSON in config.json raises error."""
        model_path = temp_cache_dir / "bad-json-model"
        model_path.mkdir()
        (model_path / "config.json").write_text("invalid json content")
        
        with pytest.raises(ModelFormatError):
            service.validate_model_format(str(model_path))
    
    def test_resolve_model_prefers_override_over_cache(self, service, temp_cache_dir):
        """Test that model overrides take precedence over cache."""
        model_name = "test/model"
        
        # Create model in cache
        cache_path = temp_cache_dir / ".cache" / "huggingface" / "hub"
        model_cache_dir = cache_path / f"models--{model_name.replace('/', '--')}"
        snapshots_dir = model_cache_dir / "snapshots" / "cache-version"
        snapshots_dir.mkdir(parents=True)
        (snapshots_dir / "config.json").write_text('{"model_type": "llama"}')
        
        # Create override path
        override_path = temp_cache_dir / "override" / "model"
        override_path.mkdir(parents=True)
        (override_path / "config.json").write_text('{"model_type": "llama", "override": true}')
        
        service.set_model_override(model_name, str(override_path))
        
        resolved_path = service.resolve_model_path(model_name)
        
        assert resolved_path == str(override_path)
        assert "override" in str(resolved_path)

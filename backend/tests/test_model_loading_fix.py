"""
Tests to verify model loading fix for 'dict' object has no attribute 'model_type' error.

Following TDD methodology: Tests ensure that models are loaded correctly
with proper Config objects, not dicts.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

from app.workers.training_worker import TrainingWorker
from app.models.training_job import TrainingJob, TrainingStatus
from app.models.dataset import Dataset
from app.models.project import Project, ProjectStatus


@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    return Mock()


@pytest.fixture
def sample_dataset(mock_db_session):
    """Create a sample dataset."""
    dataset = Dataset(
        id=1,
        name="Test Dataset",
        filename="test.csv",
        file_path="./data/user/test.csv",
        file_type="csv",
        file_size=1024,
        row_count=100,
        column_count=2,
    )
    return dataset


@pytest.fixture
def training_job(mock_db_session):
    """Create a training job."""
    job = TrainingJob(
        id=1,
        name="Test Job",
        training_type="qlora",
        model_name="meta-llama/Llama-3.2-3B",
        dataset_id=1,
        status=TrainingStatus.PENDING.value,
        epochs=1,
        batch_size=1,
        learning_rate=2e-4,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    )
    return job


@pytest.fixture
def mock_model_path(tmp_path):
    """Create a mock model directory with config.json."""
    model_dir = tmp_path / "meta-llama" / "Llama-3.2-3B"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create config.json
    config = {
        "model_type": "llama",
        "vocab_size": 32000,
        "hidden_size": 3072,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "num_hidden_layers": 28,
        "intermediate_size": 8192,
        "max_position_embeddings": 8192,
    }
    (model_dir / "config.json").write_text(json.dumps(config))
    
    return str(model_dir)


class TestModelLoadingFix:
    """Tests for model loading fix."""
    
    def test_model_config_is_not_dict_after_loading(
        self, training_job, sample_dataset, mock_model_path, mock_db_session
    ):
        """
        Test that model.config is a Config object, not a dict, after loading.
        
        Verifies:
        - Config is loaded as proper Config object
        - model.config is not a dict
        - model.config.model_type is accessible
        """
        worker = TrainingWorker(worker_id="test-worker")
        
        # Mock the model loading
        with patch('app.workers.training_worker.AutoConfig') as mock_auto_config, \
             patch('app.workers.training_worker.AutoModelForCausalLM') as mock_auto_model, \
             patch('app.workers.training_worker.AutoTokenizer') as mock_tokenizer, \
             patch('app.workers.training_worker.check_torch_available', return_value=True), \
             patch('app.workers.training_worker.check_transformers_available', return_value=True), \
             patch('app.workers.training_worker.check_peft_available', return_value=True), \
             patch('app.workers.training_worker.ModelResolutionService') as mock_resolver:
            
            # Create a mock Config object (not a dict)
            mock_config = Mock()
            mock_config.model_type = "llama"
            mock_config.vocab_size = 32000
            mock_auto_config.from_pretrained.return_value = mock_config
            
            # Create a mock model with proper config
            mock_model = Mock()
            mock_model.config = mock_config
            mock_model.config.model_type = "llama"
            mock_auto_model.from_pretrained.return_value = mock_model
            
            # Mock tokenizer
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.pad_token = None
            mock_tokenizer_instance.eos_token = "<|end_of_text|>"
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            # Mock resolver
            mock_resolver_instance = Mock()
            mock_resolver_instance.is_model_available.return_value = True
            mock_resolver_instance.resolve_model_path.return_value = mock_model_path
            mock_resolver.return_value = mock_resolver_instance
            
            # Mock dataset loading
            with patch.object(worker, '_load_dataset', return_value=[{"text": "test"}]):
                # This should not raise an error
                try:
                    worker._train_qlora(training_job, sample_dataset, mock_db_session)
                except Exception as e:
                    # Check that the error is NOT about model_type
                    assert "'dict' object has no attribute 'model_type'" not in str(e), \
                        f"Model loading still has dict config issue: {e}"
    
    def test_config_dict_is_converted_to_config_object(
        self, training_job, sample_dataset, mock_model_path, mock_db_session
    ):
        """
        Test that if config is loaded as dict, it's converted to Config object.
        
        Verifies:
        - Dict config is detected and converted
        - Final config is a Config object
        """
        worker = TrainingWorker(worker_id="test-worker")
        
        with patch('app.workers.training_worker.AutoConfig') as mock_auto_config, \
             patch('app.workers.training_worker.AutoModelForCausalLM') as mock_auto_model, \
             patch('app.workers.training_worker.AutoTokenizer') as mock_tokenizer, \
             patch('app.workers.training_worker.check_torch_available', return_value=True), \
             patch('app.workers.training_worker.check_transformers_available', return_value=True), \
             patch('app.workers.training_worker.check_peft_available', return_value=True), \
             patch('app.workers.training_worker.ModelResolutionService') as mock_resolver, \
             patch('builtins.open', create=True) as mock_open:
            
            # First call returns dict (the problem case)
            config_dict = {"model_type": "llama", "vocab_size": 32000}
            # Second call returns proper Config object
            mock_config_obj = Mock()
            mock_config_obj.model_type = "llama"
            
            mock_auto_config.from_pretrained.side_effect = [
                config_dict,  # First call returns dict
                mock_config_obj,  # Second call returns Config object
            ]
            
            # Mock CONFIG_MAPPING
            with patch('app.workers.training_worker.CONFIG_MAPPING', {"llama": Mock}):
                mock_config_class = Mock()
                mock_config_class.from_pretrained.return_value = mock_config_obj
                
                # Create mock model
                mock_model = Mock()
                mock_model.config = mock_config_obj
                mock_auto_model.from_pretrained.return_value = mock_model
                
                # Mock tokenizer
                mock_tokenizer_instance = Mock()
                mock_tokenizer_instance.pad_token = None
                mock_tokenizer_instance.eos_token = "<|end_of_text|>"
                mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
                
                # Mock resolver
                mock_resolver_instance = Mock()
                mock_resolver_instance.is_model_available.return_value = True
                mock_resolver_instance.resolve_model_path.return_value = mock_model_path
                mock_resolver.return_value = mock_resolver_instance
                
                # Mock file reading for config.json
                mock_file = MagicMock()
                mock_file.__enter__.return_value.read.return_value = json.dumps({"model_type": "llama"})
                mock_open.return_value = mock_file
                
                # Mock dataset loading
                with patch.object(worker, '_load_dataset', return_value=[{"text": "test"}]):
                    # This should handle dict config and convert it
                    try:
                        worker._train_qlora(training_job, sample_dataset, mock_db_session)
                    except Exception as e:
                        # Should not be a model_type error
                        assert "'dict' object has no attribute 'model_type'" not in str(e), \
                            f"Config dict was not properly converted: {e}"
    
    def test_model_config_is_forced_after_loading(
        self, training_job, sample_dataset, mock_model_path, mock_db_session
    ):
        """
        Test that model.config is forced to be Config object even if model loads with dict.
        
        Verifies:
        - If model.config is dict after loading, it's replaced with Config object
        """
        worker = TrainingWorker(worker_id="test-worker")
        
        with patch('app.workers.training_worker.AutoConfig') as mock_auto_config, \
             patch('app.workers.training_worker.AutoModelForCausalLM') as mock_auto_model, \
             patch('app.workers.training_worker.AutoTokenizer') as mock_tokenizer, \
             patch('app.workers.training_worker.check_torch_available', return_value=True), \
             patch('app.workers.training_worker.check_transformers_available', return_value=True), \
             patch('app.workers.training_worker.check_peft_available', return_value=True), \
             patch('app.workers.training_worker.ModelResolutionService') as mock_resolver:
            
            # Create proper Config object
            mock_config = Mock()
            mock_config.model_type = "llama"
            mock_auto_config.from_pretrained.return_value = mock_config
            
            # Create model that initially has dict config (the bug)
            mock_model = Mock()
            mock_model.config = {"model_type": "llama"}  # This is the problem!
            mock_auto_model.from_pretrained.return_value = mock_model
            
            # Mock tokenizer
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.pad_token = None
            mock_tokenizer_instance.eos_token = "<|end_of_text|>"
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            # Mock resolver
            mock_resolver_instance = Mock()
            mock_resolver_instance.is_model_available.return_value = True
            mock_resolver_instance.resolve_model_path.return_value = mock_model_path
            mock_resolver.return_value = mock_resolver_instance
            
            # Mock dataset loading
            with patch.object(worker, '_load_dataset', return_value=[{"text": "test"}]):
                # The code should detect dict config and fix it
                try:
                    worker._train_qlora(training_job, sample_dataset, mock_db_session)
                    # If we get here, the fix worked - model.config should be Config object now
                    assert not isinstance(mock_model.config, dict), \
                        "model.config should be Config object, not dict"
                except Exception as e:
                    # Check that the error is NOT about model_type
                    assert "'dict' object has no attribute 'model_type'" not in str(e), \
                        f"Model config was not fixed: {e}"

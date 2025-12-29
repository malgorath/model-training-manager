"""
Tests for database models.

Tests the SQLAlchemy models for correct field definitions,
relationships, and constraints.
"""

import pytest
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.models.dataset import Dataset
from app.models.training_job import TrainingJob, TrainingStatus, TrainingType
from app.models.training_config import TrainingConfig


class TestDatasetModel:
    """Tests for the Dataset model."""
    
    def test_create_dataset(self, test_db_session: Session):
        """Test creating a new dataset."""
        dataset = Dataset(
            name="Test Dataset",
            description="A test dataset",
            filename="test.csv",
            file_path="/uploads/test.csv",
            file_type="csv",
            file_size=1024,
            row_count=100,
            column_count=2,
            columns='["input", "output"]',
        )
        test_db_session.add(dataset)
        test_db_session.commit()
        
        assert dataset.id is not None
        assert dataset.name == "Test Dataset"
        assert dataset.file_type == "csv"
        assert dataset.created_at is not None
        assert dataset.updated_at is not None
    
    def test_dataset_required_fields(self, test_db_session: Session):
        """Test that required fields are enforced."""
        dataset = Dataset(
            name="Test",
            filename="test.csv",
            file_path="/path",
            file_type="csv",
            file_size=100,
        )
        test_db_session.add(dataset)
        test_db_session.commit()
        
        assert dataset.id is not None
        assert dataset.description is None
        assert dataset.row_count == 0
        assert dataset.column_count == 0
    
    def test_dataset_repr(self, test_db_session: Session):
        """Test dataset string representation."""
        dataset = Dataset(
            name="Test Dataset",
            filename="test.csv",
            file_path="/path",
            file_type="csv",
            file_size=100,
        )
        test_db_session.add(dataset)
        test_db_session.commit()
        
        repr_str = repr(dataset)
        assert "Dataset" in repr_str
        assert "Test Dataset" in repr_str
        assert "csv" in repr_str


class TestTrainingJobModel:
    """Tests for the TrainingJob model."""
    
    @pytest.fixture
    def sample_dataset(self, test_db_session: Session) -> Dataset:
        """Create a sample dataset for testing."""
        dataset = Dataset(
            name="Training Data",
            filename="data.csv",
            file_path="/uploads/data.csv",
            file_type="csv",
            file_size=2048,
            row_count=500,
            column_count=2,
        )
        test_db_session.add(dataset)
        test_db_session.commit()
        return dataset
    
    def test_create_training_job(self, test_db_session: Session, sample_dataset: Dataset):
        """Test creating a new training job."""
        job = TrainingJob(
            name="Test Training Job",
            description="Testing model training",
            dataset_id=sample_dataset.id,
            training_type=TrainingType.QLORA.value,
            model_name="llama3.2:3b",
            batch_size=4,
            learning_rate=2e-4,
            epochs=3,
        )
        test_db_session.add(job)
        test_db_session.commit()
        
        assert job.id is not None
        assert job.status == TrainingStatus.PENDING.value
        assert job.progress == 0.0
        assert job.current_epoch == 0
        assert job.dataset_id == sample_dataset.id
    
    def test_training_job_default_values(self, test_db_session: Session, sample_dataset: Dataset):
        """Test training job default values."""
        job = TrainingJob(
            name="Default Job",
            dataset_id=sample_dataset.id,
        )
        test_db_session.add(job)
        test_db_session.commit()
        
        assert job.status == TrainingStatus.PENDING.value
        assert job.training_type == TrainingType.QLORA.value
        assert job.model_name == "llama3.2:3b"
        assert job.batch_size == 4
        assert job.learning_rate == 2e-4
        assert job.epochs == 3
        assert job.lora_r == 16
        assert job.lora_alpha == 32
        assert job.lora_dropout == 0.05
    
    def test_training_job_relationship(self, test_db_session: Session, sample_dataset: Dataset):
        """Test training job - dataset relationship."""
        job = TrainingJob(
            name="Relationship Test",
            dataset_id=sample_dataset.id,
        )
        test_db_session.add(job)
        test_db_session.commit()
        
        # Test forward relationship
        assert job.dataset is not None
        assert job.dataset.name == "Training Data"
        
        # Test backward relationship
        assert len(sample_dataset.training_jobs) == 1
        assert sample_dataset.training_jobs[0].name == "Relationship Test"
    
    def test_training_job_cascade_delete(self, test_db_session: Session, sample_dataset: Dataset):
        """Test that training jobs are deleted when dataset is deleted."""
        job = TrainingJob(
            name="Cascade Test",
            dataset_id=sample_dataset.id,
        )
        test_db_session.add(job)
        test_db_session.commit()
        job_id = job.id
        
        # Delete the dataset
        test_db_session.delete(sample_dataset)
        test_db_session.commit()
        
        # Job should be deleted too
        deleted_job = test_db_session.get(TrainingJob, job_id)
        assert deleted_job is None
    
    def test_training_status_enum(self):
        """Test TrainingStatus enumeration values."""
        assert TrainingStatus.PENDING.value == "pending"
        assert TrainingStatus.QUEUED.value == "queued"
        assert TrainingStatus.RUNNING.value == "running"
        assert TrainingStatus.COMPLETED.value == "completed"
        assert TrainingStatus.FAILED.value == "failed"
        assert TrainingStatus.CANCELLED.value == "cancelled"
    
    def test_training_type_enum(self):
        """Test TrainingType enumeration values."""
        assert TrainingType.QLORA.value == "qlora"
        assert TrainingType.UNSLOTH.value == "unsloth"
        assert TrainingType.RAG.value == "rag"
        assert TrainingType.STANDARD.value == "standard"
    
    def test_training_job_repr(self, test_db_session: Session, sample_dataset: Dataset):
        """Test training job string representation."""
        job = TrainingJob(
            name="Repr Test",
            dataset_id=sample_dataset.id,
        )
        test_db_session.add(job)
        test_db_session.commit()
        
        repr_str = repr(job)
        assert "TrainingJob" in repr_str
        assert "Repr Test" in repr_str
        assert "pending" in repr_str


class TestTrainingConfigModel:
    """Tests for the TrainingConfig model."""
    
    def test_create_training_config(self, test_db_session: Session):
        """Test creating training configuration."""
        config = TrainingConfig(
            max_concurrent_workers=4,
            default_model="llama3.2:3b",
            default_training_type="qlora",
        )
        test_db_session.add(config)
        test_db_session.commit()
        
        assert config.id is not None
        assert config.max_concurrent_workers == 4
        assert config.active_workers == 0
    
    def test_training_config_default_values(self, test_db_session: Session):
        """Test training config default values."""
        config = TrainingConfig()
        test_db_session.add(config)
        test_db_session.commit()
        
        assert config.max_concurrent_workers == 4
        assert config.active_workers == 0
        assert config.default_model == "llama3.2:3b"
        assert config.default_training_type == "qlora"
        assert config.default_batch_size == 4
        assert config.default_learning_rate == 2e-4
        assert config.default_epochs == 3
        assert config.auto_start_workers is False
    
    def test_training_config_update(self, test_db_session: Session):
        """Test updating training configuration."""
        config = TrainingConfig()
        test_db_session.add(config)
        test_db_session.commit()
        
        original_updated_at = config.updated_at
        
        config.max_concurrent_workers = 8
        config.auto_start_workers = True
        test_db_session.commit()
        
        assert config.max_concurrent_workers == 8
        assert config.auto_start_workers is True
    
    def test_training_config_repr(self, test_db_session: Session):
        """Test training config string representation."""
        config = TrainingConfig(max_concurrent_workers=6)
        test_db_session.add(config)
        test_db_session.commit()
        
        repr_str = repr(config)
        assert "TrainingConfig" in repr_str
        assert "6" in repr_str
    
    def test_training_config_model_provider_default(self, test_db_session: Session):
        """Test model_provider field default value."""
        config = TrainingConfig()
        test_db_session.add(config)
        test_db_session.commit()
        
        assert config.model_provider == "ollama"
    
    def test_training_config_model_provider_custom(self, test_db_session: Session):
        """Test setting custom model_provider value."""
        config = TrainingConfig(
            model_provider="lm_studio",
            model_api_url="http://localhost:1234",
        )
        test_db_session.add(config)
        test_db_session.commit()
        
        assert config.model_provider == "lm_studio"
        assert config.model_api_url == "http://localhost:1234"
    
    def test_training_config_model_api_url_default(self, test_db_session: Session):
        """Test model_api_url field default value."""
        config = TrainingConfig()
        test_db_session.add(config)
        test_db_session.commit()
        
        assert config.model_api_url == "http://localhost:11434"
    
    def test_training_config_model_api_url_custom(self, test_db_session: Session):
        """Test setting custom model_api_url value."""
        config = TrainingConfig(
            model_api_url="http://localhost:1234",
        )
        test_db_session.add(config)
        test_db_session.commit()
        
        assert config.model_api_url == "http://localhost:1234"
    
    def test_training_config_provider_ollama(self, test_db_session: Session):
        """Test configuring for Ollama provider."""
        config = TrainingConfig(
            model_provider="ollama",
            model_api_url="http://localhost:11434",
        )
        test_db_session.add(config)
        test_db_session.commit()
        
        assert config.model_provider == "ollama"
        assert config.model_api_url == "http://localhost:11434"
    
    def test_training_config_provider_lm_studio(self, test_db_session: Session):
        """Test configuring for LM Studio provider."""
        config = TrainingConfig(
            model_provider="lm_studio",
            model_api_url="http://localhost:1234",
        )
        test_db_session.add(config)
        test_db_session.commit()
        
        assert config.model_provider == "lm_studio"
        assert config.model_api_url == "http://localhost:1234"


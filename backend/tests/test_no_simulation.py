"""
Tests verifying training fails explicitly when models unavailable.

Tests that training methods no longer fall back to simulation and instead
fail explicitly with clear error messages when models are unavailable.
"""

import pytest
from unittest.mock import patch, MagicMock
from sqlalchemy.orm import Session

from app.models.training_job import TrainingJob, TrainingStatus, TrainingType
from app.models.dataset import Dataset
from app.workers.training_worker import TrainingWorker
from app.services.model_resolution_service import ModelNotFoundError, ModelFormatError


class TestNoSimulationFallbacks:
    """Tests that training fails explicitly without simulation fallbacks."""
    
    @pytest.fixture
    def sample_dataset(self, test_db_session: Session):
        """Create a sample dataset."""
        dataset = Dataset(
            name="Test Dataset",
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
        return dataset
    
    @pytest.fixture
    def sample_job(self, test_db_session: Session, sample_dataset):
        """Create a sample training job."""
        job = TrainingJob(
            name="Test Job",
            dataset_id=sample_dataset.id,
            training_type=TrainingType.QLORA.value,
            model_name="nonexistent-model",
            status=TrainingStatus.PENDING.value,
        )
        test_db_session.add(job)
        test_db_session.commit()
        return job
    
    @pytest.fixture
    def worker(self):
        """Create a training worker."""
        def db_factory():
            from tests.conftest import test_db_engine
            from sqlalchemy.orm import sessionmaker
            TestingSessionLocal = sessionmaker(bind=test_db_engine)
            return TestingSessionLocal()
        
        return TrainingWorker(worker_id="test-worker", db_session_factory=db_factory)
    
    def test_qlora_fails_on_model_not_found(self, worker, sample_job, test_db_session):
        """Test that QLoRA training fails when model not found."""
        with patch('app.workers.training_worker.ModelResolutionService') as MockResolver:
            mock_service = MagicMock()
            mock_service.is_model_available.return_value = False
            MockResolver.return_value = mock_service
            
            with pytest.raises(ModelNotFoundError) as exc_info:
                worker._process_job(sample_job.id)
            
            assert "not found" in str(exc_info.value).lower()
            mock_service.is_model_available.assert_called_once()
    
    def test_unsloth_fails_on_model_not_found(self, worker, sample_job, test_db_session):
        """Test that Unsloth training fails when model not found."""
        sample_job.training_type = TrainingType.UNSLOTH.value
        test_db_session.commit()
        
        with patch('app.workers.training_worker.ModelResolutionService') as MockResolver:
            mock_service = MagicMock()
            mock_service.is_model_available.return_value = False
            MockResolver.return_value = mock_service
            
            with pytest.raises(ModelNotFoundError):
                worker.process_job(sample_job.id)
    
    def test_standard_fails_on_model_not_found(self, worker, sample_job, test_db_session):
        """Test that standard training fails when model not found."""
        sample_job.training_type = TrainingType.STANDARD.value
        test_db_session.commit()
        
        with patch('app.workers.training_worker.ModelResolutionService') as MockResolver:
            mock_service = MagicMock()
            mock_service.is_model_available.return_value = False
            MockResolver.return_value = mock_service
            
            with pytest.raises(ModelNotFoundError):
                worker.process_job(sample_job.id)
    
    def test_training_fails_on_model_format_error(self, worker, sample_job, test_db_session):
        """Test that training fails when model format is invalid."""
        with patch('app.workers.training_worker.ModelResolutionService') as MockResolver:
            mock_service = MagicMock()
            mock_service.is_model_available.return_value = True
            mock_service.resolve_model_path.return_value = "/fake/path"
            mock_service.validate_model_format.side_effect = ModelFormatError("Invalid format")
            MockResolver.return_value = mock_service
            
            with pytest.raises(ModelFormatError):
                worker.process_job(sample_job.id)
    
    def test_training_fails_on_missing_libraries(self, worker, sample_job, test_db_session):
        """Test that training fails when required libraries are missing."""
        with patch('app.workers.training_worker.check_torch_available', return_value=False):
            with pytest.raises(RuntimeError) as exc_info:
                worker._process_job(sample_job.id)
            
            assert "libraries" in str(exc_info.value).lower() or "dependencies" in str(exc_info.value).lower()
    
    def test_job_status_set_to_failed_on_error(self, worker, sample_job, test_db_session):
        """Test that job status is set to FAILED when training fails."""
        with patch('app.workers.training_worker.ModelResolutionService') as MockResolver:
            mock_service = MagicMock()
            mock_service.is_model_available.return_value = False
            MockResolver.return_value = mock_service
            
            try:
                worker._process_job(sample_job.id)
            except Exception:
                pass  # Expected to fail
            
            test_db_session.refresh(sample_job)
            
            test_db_session.refresh(sample_job)
            assert sample_job.status == TrainingStatus.FAILED.value
            assert sample_job.error_message is not None
    
    def test_no_simulation_methods_called(self, worker, sample_job, test_db_session):
        """Test that simulation methods are never called."""
        with patch('app.workers.training_worker.ModelResolutionService') as MockResolver:
            mock_service = MagicMock()
            mock_service.is_model_available.return_value = False
            MockResolver.return_value = mock_service
            
            # Ensure simulation methods don't exist or aren't called
            with patch.object(worker, '_train_qlora_simulated', side_effect=AssertionError("Simulation should not be called")):
                with pytest.raises(ModelNotFoundError):
                    worker._process_job(sample_job.id)

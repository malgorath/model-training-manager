"""
Tests for the Training service.

Tests training job creation, management, and worker coordination.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from sqlalchemy.orm import Session

from app.models.dataset import Dataset
from app.models.training_job import TrainingJob, TrainingStatus, TrainingType
from app.models.training_config import TrainingConfig
from app.schemas.training_job import TrainingJobCreate, TrainingJobUpdate, TrainingJobProgress
from app.schemas.training_config import TrainingConfigUpdate
from app.services.training_service import TrainingService


class TestTrainingService:
    """Tests for TrainingService."""
    
    @pytest.fixture
    def mock_worker_pool(self):
        """Create a mock worker pool."""
        pool = MagicMock()
        pool.is_running = False
        pool.active_worker_count = 0
        pool.get_status.return_value = {
            "total_workers": 0,
            "active_workers": 0,
            "idle_workers": 0,
            "busy_workers": 0,
            "workers": [],
        }
        return pool
    
    @pytest.fixture
    def service(self, test_db_session: Session, mock_worker_pool) -> TrainingService:
        """Create a TrainingService instance for testing."""
        # Store original pool
        original_pool = TrainingService._worker_pool
        
        # Set mock pool on class
        TrainingService._worker_pool = mock_worker_pool
        service = TrainingService(test_db_session)
        
        yield service
        
        # Restore original pool
        TrainingService._worker_pool = original_pool
    
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
    
    def test_create_job(
        self,
        service: TrainingService,
        sample_dataset: Dataset,
    ):
        """Test creating a training job."""
        job_data = TrainingJobCreate(
            name="Test Training Job",
            description="Testing job creation",
            dataset_id=sample_dataset.id,
            training_type=TrainingType.QLORA,
            model_name="llama3.2:3b",
            batch_size=8,
            learning_rate=1e-4,
            epochs=5,
        )
        
        job = service.create_job(job_data)
        
        assert job.id is not None
        assert job.name == "Test Training Job"
        assert job.status == TrainingStatus.PENDING.value
        assert job.training_type == TrainingType.QLORA.value
        assert job.batch_size == 8
        assert job.learning_rate == 1e-4
        assert job.epochs == 5
    
    def test_create_job_with_unsloth(
        self,
        service: TrainingService,
        sample_dataset: Dataset,
    ):
        """Test creating a training job with Unsloth type."""
        job_data = TrainingJobCreate(
            name="Unsloth Training Job",
            description="Testing Unsloth job creation",
            dataset_id=sample_dataset.id,
            training_type=TrainingType.UNSLOTH,
            model_name="llama3.2:3b",
            batch_size=8,
            learning_rate=1e-4,
            epochs=5,
        )
        
        job = service.create_job(job_data)
        
        assert job.id is not None
        assert job.name == "Unsloth Training Job"
        assert job.status == TrainingStatus.PENDING.value
        assert job.training_type == TrainingType.UNSLOTH.value
        assert job.batch_size == 8
    
    def test_create_job_with_defaults(
        self,
        service: TrainingService,
        sample_dataset: Dataset,
    ):
        """Test creating a job with default parameters."""
        job_data = TrainingJobCreate(
            name="Default Job",
            dataset_id=sample_dataset.id,
        )
        
        job = service.create_job(job_data)
        
        # Should use config defaults
        config = service.get_config()
        assert job.batch_size == config.default_batch_size
        assert job.learning_rate == config.default_learning_rate
        assert job.epochs == config.default_epochs
    
    def test_create_job_dataset_not_found(
        self,
        service: TrainingService,
    ):
        """Test creating a job with non-existent dataset."""
        job_data = TrainingJobCreate(
            name="Bad Job",
            dataset_id=99999,
        )
        
        with pytest.raises(ValueError) as exc_info:
            service.create_job(job_data)
        
        assert "not found" in str(exc_info.value)
    
    def test_get_job(
        self,
        service: TrainingService,
        sample_dataset: Dataset,
        test_db_session: Session,
    ):
        """Test getting a job by ID."""
        job = TrainingJob(
            name="Test Job",
            dataset_id=sample_dataset.id,
        )
        test_db_session.add(job)
        test_db_session.commit()
        
        retrieved = service.get_job(job.id)
        
        assert retrieved is not None
        assert retrieved.id == job.id
        assert retrieved.name == "Test Job"
    
    def test_get_job_not_found(self, service: TrainingService):
        """Test getting a non-existent job."""
        result = service.get_job(99999)
        assert result is None
    
    def test_list_jobs(
        self,
        service: TrainingService,
        sample_dataset: Dataset,
        test_db_session: Session,
    ):
        """Test listing jobs with pagination."""
        for i in range(15):
            job = TrainingJob(
                name=f"Job {i}",
                dataset_id=sample_dataset.id,
                status=TrainingStatus.PENDING.value if i % 2 == 0 else TrainingStatus.COMPLETED.value,
            )
            test_db_session.add(job)
        test_db_session.commit()
        
        # Test pagination
        result = service.list_jobs(page=1, page_size=10)
        
        assert result["total"] == 15
        assert result["page"] == 1
        assert result["pages"] == 2
        assert len(result["items"]) == 10
    
    def test_list_jobs_with_status_filter(
        self,
        service: TrainingService,
        sample_dataset: Dataset,
        test_db_session: Session,
    ):
        """Test listing jobs filtered by status."""
        for i in range(10):
            job = TrainingJob(
                name=f"Job {i}",
                dataset_id=sample_dataset.id,
                status=TrainingStatus.PENDING.value if i < 6 else TrainingStatus.COMPLETED.value,
            )
            test_db_session.add(job)
        test_db_session.commit()
        
        # Filter by pending
        result = service.list_jobs(status=TrainingStatus.PENDING)
        
        assert result["total"] == 6
        assert all(j.status == TrainingStatus.PENDING.value for j in result["items"])
    
    def test_update_job(
        self,
        service: TrainingService,
        sample_dataset: Dataset,
        test_db_session: Session,
    ):
        """Test updating a job's metadata."""
        job = TrainingJob(
            name="Original Name",
            dataset_id=sample_dataset.id,
        )
        test_db_session.add(job)
        test_db_session.commit()
        
        update_data = TrainingJobUpdate(
            name="Updated Name",
            description="New description",
        )
        
        updated = service.update_job(job.id, update_data)
        
        assert updated is not None
        assert updated.name == "Updated Name"
        assert updated.description == "New description"
    
    def test_update_job_progress(
        self,
        service: TrainingService,
        sample_dataset: Dataset,
        test_db_session: Session,
    ):
        """Test updating a job's progress."""
        job = TrainingJob(
            name="Progress Test",
            dataset_id=sample_dataset.id,
            status=TrainingStatus.RUNNING.value,
        )
        test_db_session.add(job)
        test_db_session.commit()
        
        progress_data = TrainingJobProgress(
            progress=50.0,
            current_epoch=2,
            current_loss=0.5,
        )
        
        updated = service.update_job_progress(job.id, progress_data)
        
        assert updated is not None
        assert updated.progress == 50.0
        assert updated.current_epoch == 2
        assert updated.current_loss == 0.5
    
    def test_cancel_job(
        self,
        service: TrainingService,
        sample_dataset: Dataset,
        test_db_session: Session,
    ):
        """Test cancelling a job."""
        job = TrainingJob(
            name="Cancel Test",
            dataset_id=sample_dataset.id,
            status=TrainingStatus.PENDING.value,
        )
        test_db_session.add(job)
        test_db_session.commit()
        
        cancelled = service.cancel_job(job.id)
        
        assert cancelled is not None
        assert cancelled.status == TrainingStatus.CANCELLED.value
        assert cancelled.completed_at is not None
    
    def test_cancel_completed_job_fails(
        self,
        service: TrainingService,
        sample_dataset: Dataset,
        test_db_session: Session,
    ):
        """Test that completed jobs cannot be cancelled."""
        job = TrainingJob(
            name="Completed Job",
            dataset_id=sample_dataset.id,
            status=TrainingStatus.COMPLETED.value,
        )
        test_db_session.add(job)
        test_db_session.commit()
        
        with pytest.raises(ValueError) as exc_info:
            service.cancel_job(job.id)
        
        assert "Cannot cancel" in str(exc_info.value)
    
    def test_get_config(
        self,
        service: TrainingService,
        test_db_session: Session,
    ):
        """Test getting the training configuration."""
        config = service.get_config()
        
        assert config is not None
        assert config.max_concurrent_workers == 4
        assert config.default_model == "llama3.2:3b"
    
    def test_get_config_creates_default(
        self,
        service: TrainingService,
        test_db_session: Session,
    ):
        """Test that get_config creates default if not exists."""
        # Ensure no config exists
        test_db_session.query(TrainingConfig).delete()
        test_db_session.commit()
        
        config = service.get_config()
        
        assert config is not None
        assert config.id is not None
    
    def test_update_config(
        self,
        service: TrainingService,
        test_db_session: Session,
    ):
        """Test updating the training configuration."""
        update_data = TrainingConfigUpdate(
            max_concurrent_workers=8,
            default_batch_size=16,
            auto_start_workers=True,
        )
        
        config = service.update_config(update_data)
        
        assert config.max_concurrent_workers == 8
        assert config.default_batch_size == 16
        assert config.auto_start_workers is True
    
    def test_get_worker_pool_status(
        self,
        service: TrainingService,
    ):
        """Test getting worker pool status."""
        status = service.get_worker_pool_status()
        
        assert status is not None
        assert status.total_workers == 0
        assert status.max_workers == 4
        assert isinstance(status.workers, list)
    
    def test_update_job_not_found(
        self,
        service: TrainingService,
    ):
        """Test updating a non-existent job."""
        update_data = TrainingJobUpdate(name="New Name")
        result = service.update_job(99999, update_data)
        assert result is None
    
    def test_update_job_progress_not_found(
        self,
        service: TrainingService,
    ):
        """Test updating progress for non-existent job."""
        progress_data = TrainingJobProgress(progress=50.0, current_epoch=1)
        result = service.update_job_progress(99999, progress_data)
        assert result is None
    
    def test_cancel_job_not_found(
        self,
        service: TrainingService,
    ):
        """Test cancelling a non-existent job."""
        with pytest.raises(ValueError) as exc_info:
            service.cancel_job(99999)
        assert "not found" in str(exc_info.value).lower()
    
    def test_start_workers(
        self,
        service: TrainingService,
    ):
        """Test starting workers."""
        # The mock_worker_pool is set on the class
        service.start_workers(2)
        TrainingService._worker_pool.start_workers.assert_called_once_with(2)
    
    def test_stop_workers(
        self,
        service: TrainingService,
    ):
        """Test stopping workers."""
        service.stop_workers()
        TrainingService._worker_pool.stop_all_workers.assert_called_once()
    
    def test_delete_job(
        self,
        service: TrainingService,
        sample_dataset: Dataset,
        test_db_session: Session,
    ):
        """Test deleting a job."""
        job = TrainingJob(
            name="Delete Test",
            dataset_id=sample_dataset.id,
            status=TrainingStatus.PENDING.value,
        )
        test_db_session.add(job)
        test_db_session.commit()
        job_id = job.id
        
        result = service.delete_job(job_id)
        
        assert result is True
        assert service.get_job(job_id) is None
    
    def test_delete_job_not_found(
        self,
        service: TrainingService,
    ):
        """Test deleting a non-existent job."""
        result = service.delete_job(99999)
        assert result is False
    
    def test_cancel_running_job(
        self,
        service: TrainingService,
        sample_dataset: Dataset,
        test_db_session: Session,
    ):
        """Test cancelling a running job triggers worker pool cancel."""
        job = TrainingJob(
            name="Running Job",
            dataset_id=sample_dataset.id,
            status=TrainingStatus.RUNNING.value,
        )
        test_db_session.add(job)
        test_db_session.commit()
        
        cancelled = service.cancel_job(job.id)
        
        assert cancelled.status == TrainingStatus.CANCELLED.value
        TrainingService._worker_pool.cancel_job.assert_called_once_with(job.id)
    
    def test_submit_job_when_pool_running(
        self,
        service: TrainingService,
        sample_dataset: Dataset,
        test_db_session: Session,
    ):
        """Test job is auto-submitted when pool is running."""
        TrainingService._worker_pool.is_running = True
        TrainingService._worker_pool.submit_job.return_value = True
        
        job_data = TrainingJobCreate(
            name="Submit Test",
            dataset_id=sample_dataset.id,
        )
        
        job = service.create_job(job_data)
        
        # Job should be queued when pool is running
        TrainingService._worker_pool.submit_job.assert_called_once_with(job.id)
    
    def test_restart_workers(
        self,
        service: TrainingService,
    ):
        """Test restarting workers."""
        TrainingService._worker_pool.active_worker_count = 2
        
        service.restart_workers()
        
        TrainingService._worker_pool.stop_all_workers.assert_called_once()
        TrainingService._worker_pool.start_workers.assert_called()


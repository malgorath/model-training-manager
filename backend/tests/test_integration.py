"""
Integration tests for the Model Training Manager.

Tests complete workflows across multiple services and components.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.dataset import Dataset
from app.models.training_job import TrainingJob, TrainingStatus, TrainingType
from app.models.training_config import TrainingConfig


class TestDatasetToTrainingWorkflow:
    """Tests for the complete dataset-to-training workflow."""
    
    def test_upload_dataset_and_create_training_job(
        self,
        client: TestClient,
        test_db_session: Session,
        sample_csv_content: bytes,
        tmp_path: Path,
    ):
        """Test complete workflow: upload dataset then create training job."""
        with patch("app.services.dataset_service.settings") as mock_settings:
            mock_settings.allowed_extensions = {".csv", ".json"}
            mock_settings.max_upload_size = 100 * 1024 * 1024
            mock_settings.get_upload_path.return_value = tmp_path
            
            # Step 1: Upload dataset
            response = client.post(
                "/api/v1/datasets/",
                files={"file": ("training_data.csv", sample_csv_content, "text/csv")},
                data={"name": "Training Data", "description": "Test data for training"},
            )
            
            assert response.status_code == 201
            dataset = response.json()
            dataset_id = dataset["id"]
            assert dataset["name"] == "Training Data"
            assert dataset["file_type"] == "csv"
            assert dataset["row_count"] == 3
        
        # Step 2: Create training job
        with patch("app.services.training_service.TrainingService._worker_pool") as mock_pool:
            mock_pool.is_running = False
            
            response = client.post(
                "/api/v1/jobs/",
                json={
                    "name": "Fine-tuning Job",
                    "description": "Testing the training pipeline",
                    "dataset_id": dataset_id,
                    "training_type": "qlora",
                    "batch_size": 2,
                    "epochs": 3,
                },
            )
            
            assert response.status_code == 201
            job = response.json()
            assert job["name"] == "Fine-tuning Job"
            assert job["dataset_id"] == dataset_id
            assert job["status"] == "pending"
            assert job["batch_size"] == 2
            assert job["epochs"] == 3
    
    def test_full_training_lifecycle(
        self,
        client: TestClient,
        test_db_session: Session,
        tmp_path: Path,
    ):
        """Test complete training job lifecycle: create, run, complete."""
        # Create dataset directly
        dataset = Dataset(
            name="Lifecycle Test Data",
            filename="data.csv",
            file_path=str(tmp_path / "data.csv"),
            file_type="csv",
            file_size=1000,
            row_count=100,
            column_count=2,
        )
        test_db_session.add(dataset)
        test_db_session.commit()
        
        # Create training job
        with patch("app.services.training_service.TrainingService._worker_pool") as mock_pool:
            mock_pool.is_running = False
            
            response = client.post(
                "/api/v1/jobs/",
                json={
                    "name": "Lifecycle Test",
                    "dataset_id": dataset.id,
                },
            )
            
            assert response.status_code == 201
            job_id = response.json()["id"]
        
        # Verify job status
        response = client.get(f"/api/v1/jobs/{job_id}/status")
        assert response.status_code == 200
        status = response.json()
        assert status["status"] == "pending"
        assert status["progress"] == 0.0
        
        # Simulate job running (update in database)
        job = test_db_session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        job.status = TrainingStatus.RUNNING.value
        job.progress = 50.0
        job.current_epoch = 2
        job.current_loss = 0.5
        test_db_session.commit()
        
        # Verify updated status
        response = client.get(f"/api/v1/jobs/{job_id}/status")
        assert response.status_code == 200
        status = response.json()
        assert status["status"] == "running"
        assert status["progress"] == 50.0
        assert status["current_epoch"] == 2
        
        # Complete the job
        job.status = TrainingStatus.COMPLETED.value
        job.progress = 100.0
        job.current_epoch = 3
        test_db_session.commit()
        
        # Verify completion
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        job_data = response.json()
        assert job_data["status"] == "completed"
        assert job_data["progress"] == 100.0


class TestConfigurationWorkflow:
    """Tests for configuration update workflow."""
    
    def test_update_config_and_verify(
        self,
        client: TestClient,
        test_db_session: Session,
    ):
        """Test updating configuration and verifying changes."""
        with patch("app.services.training_service.TrainingService._worker_pool") as mock_pool:
            mock_pool.set_max_workers = MagicMock()
            mock_pool.get_status.return_value = {"workers": []}
            
            # Get initial config
            response = client.get("/api/v1/config/")
            assert response.status_code == 200
            initial_config = response.json()
            
            # Update configuration
            response = client.patch(
                "/api/v1/config/",
                json={
                    "max_concurrent_workers": 8,
                    "default_batch_size": 16,
                    "default_epochs": 5,
                    "auto_start_workers": True,
                },
            )
            
            assert response.status_code == 200
            updated_config = response.json()
            assert updated_config["max_concurrent_workers"] == 8
            assert updated_config["default_batch_size"] == 16
            assert updated_config["default_epochs"] == 5
            assert updated_config["auto_start_workers"] is True
            
            # Verify changes persisted
            response = client.get("/api/v1/config/")
            assert response.status_code == 200
            final_config = response.json()
            assert final_config["max_concurrent_workers"] == 8
            assert final_config["default_batch_size"] == 16


class TestWorkerManagementWorkflow:
    """Tests for worker management workflow."""
    
    def test_start_stop_restart_workers(
        self,
        client: TestClient,
        test_db_session: Session,
    ):
        """Test worker lifecycle: start, check status, stop."""
        with patch("app.services.training_service.TrainingService._worker_pool") as mock_pool:
            mock_pool.start_workers = MagicMock()
            mock_pool.stop_all_workers = MagicMock()
            mock_pool.active_worker_count = 0
            mock_pool.get_status.return_value = {
                "total_workers": 0,
                "active_workers": 0,
                "idle_workers": 0,
                "busy_workers": 0,
                "workers": [],
            }
            
            # Initially no workers
            response = client.get("/api/v1/workers/")
            assert response.status_code == 200
            status = response.json()
            assert status["total_workers"] == 0
            
            # Start workers
            mock_pool.active_worker_count = 2
            mock_pool.get_status.return_value = {
                "total_workers": 2,
                "active_workers": 2,
                "idle_workers": 2,
                "busy_workers": 0,
                "workers": [
                    {"id": "w1", "status": "idle", "jobs_completed": 0, "started_at": "2024-01-01T00:00:00Z"},
                    {"id": "w2", "status": "idle", "jobs_completed": 0, "started_at": "2024-01-01T00:00:00Z"},
                ],
            }
            
            response = client.post(
                "/api/v1/workers/",
                json={"action": "start", "worker_count": 2},
            )
            assert response.status_code == 200
            mock_pool.start_workers.assert_called_once_with(2)
            
            # Stop workers
            mock_pool.active_worker_count = 0
            mock_pool.get_status.return_value = {
                "total_workers": 0,
                "active_workers": 0,
                "idle_workers": 0,
                "busy_workers": 0,
                "workers": [],
            }
            
            response = client.post(
                "/api/v1/workers/",
                json={"action": "stop"},
            )
            assert response.status_code == 200
            mock_pool.stop_all_workers.assert_called_once()


class TestErrorHandling:
    """Tests for error handling across the application."""
    
    def test_create_job_with_invalid_dataset(
        self,
        client: TestClient,
    ):
        """Test error handling when creating job with non-existent dataset."""
        with patch("app.services.training_service.TrainingService._worker_pool") as mock_pool:
            mock_pool.is_running = False
            
            response = client.post(
                "/api/v1/jobs/",
                json={
                    "name": "Bad Job",
                    "dataset_id": 99999,
                },
            )
            
            assert response.status_code == 400
            assert "not found" in response.json()["detail"]
    
    def test_cancel_completed_job_fails(
        self,
        client: TestClient,
        test_db_session: Session,
    ):
        """Test that completed jobs cannot be cancelled."""
        # Create a completed job
        dataset = Dataset(
            name="Test Data",
            filename="data.csv",
            file_path="/path/data.csv",
            file_type="csv",
            file_size=1000,
        )
        test_db_session.add(dataset)
        test_db_session.commit()
        
        job = TrainingJob(
            name="Completed Job",
            dataset_id=dataset.id,
            status=TrainingStatus.COMPLETED.value,
        )
        test_db_session.add(job)
        test_db_session.commit()
        
        # Try to cancel
        with patch("app.services.training_service.TrainingService._worker_pool"):
            response = client.post(f"/api/v1/jobs/{job.id}/cancel")
            
            assert response.status_code == 400
            assert "Cannot cancel" in response.json()["detail"]
    
    def test_get_nonexistent_dataset(self, client: TestClient):
        """Test getting a non-existent dataset."""
        response = client.get("/api/v1/datasets/99999")
        assert response.status_code == 404
    
    def test_get_nonexistent_job(self, client: TestClient):
        """Test getting a non-existent job."""
        response = client.get("/api/v1/jobs/99999")
        assert response.status_code == 404


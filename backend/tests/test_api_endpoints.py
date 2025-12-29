"""
Tests for API endpoints.

Tests all API endpoints using the FastAPI test client.
"""

import json
import pytest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.dataset import Dataset
from app.models.training_job import TrainingJob, TrainingStatus, TrainingType
from app.models.training_config import TrainingConfig


class TestHealthEndpoint:
    """Tests for the health check endpoint."""
    
    def test_health_check(self, client: TestClient):
        """Test health check returns healthy status."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestDatasetEndpoints:
    """Tests for dataset API endpoints."""
    
    def test_upload_dataset_csv(
        self,
        client: TestClient,
        test_db_session: Session,
        sample_csv_content: bytes,
        tmp_path: Path,
    ):
        """Test uploading a CSV dataset."""
        with patch("app.services.dataset_service.settings") as mock_settings:
            mock_settings.allowed_extensions = {".csv", ".json"}
            mock_settings.max_upload_size = 100 * 1024 * 1024
            mock_settings.get_upload_path.return_value = tmp_path
            
            response = client.post(
                "/api/v1/datasets/",
                files={"file": ("test.csv", sample_csv_content, "text/csv")},
                data={"name": "Test Dataset", "description": "A test"},
            )
        
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Dataset"
        assert data["file_type"] == "csv"
        assert data["row_count"] == 3
    
    def test_upload_dataset_invalid_type(
        self,
        client: TestClient,
        tmp_path: Path,
    ):
        """Test uploading an unsupported file type."""
        with patch("app.services.dataset_service.settings") as mock_settings:
            mock_settings.allowed_extensions = {".csv", ".json"}
            mock_settings.max_upload_size = 100 * 1024 * 1024
            mock_settings.get_upload_path.return_value = tmp_path
            
            response = client.post(
                "/api/v1/datasets/",
                files={"file": ("test.txt", b"content", "text/plain")},
                data={"name": "Invalid"},
            )
        
        assert response.status_code == 400
        assert "Unsupported" in response.json()["detail"]
    
    def test_list_datasets(
        self,
        client: TestClient,
        test_db_session: Session,
    ):
        """Test listing datasets."""
        # Create test datasets
        for i in range(5):
            dataset = Dataset(
                name=f"Dataset {i}",
                filename=f"data{i}.csv",
                file_path=f"/path/data{i}.csv",
                file_type="csv",
                file_size=100,
            )
            test_db_session.add(dataset)
        test_db_session.commit()
        
        response = client.get("/api/v1/datasets/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 5
        assert len(data["items"]) == 5
    
    def test_list_datasets_pagination(
        self,
        client: TestClient,
        test_db_session: Session,
    ):
        """Test dataset pagination."""
        for i in range(15):
            dataset = Dataset(
                name=f"Dataset {i}",
                filename=f"data{i}.csv",
                file_path=f"/path/data{i}.csv",
                file_type="csv",
                file_size=100,
            )
            test_db_session.add(dataset)
        test_db_session.commit()
        
        response = client.get("/api/v1/datasets/?page=1&page_size=10")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 15
        assert len(data["items"]) == 10
        assert data["pages"] == 2
    
    def test_get_dataset(
        self,
        client: TestClient,
        test_db_session: Session,
    ):
        """Test getting a specific dataset."""
        dataset = Dataset(
            name="Test Dataset",
            filename="test.csv",
            file_path="/path/test.csv",
            file_type="csv",
            file_size=100,
        )
        test_db_session.add(dataset)
        test_db_session.commit()
        
        response = client.get(f"/api/v1/datasets/{dataset.id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Dataset"
    
    def test_get_dataset_not_found(self, client: TestClient):
        """Test getting a non-existent dataset."""
        response = client.get("/api/v1/datasets/99999")
        assert response.status_code == 404
    
    def test_update_dataset(
        self,
        client: TestClient,
        test_db_session: Session,
    ):
        """Test updating a dataset."""
        dataset = Dataset(
            name="Original",
            filename="test.csv",
            file_path="/path/test.csv",
            file_type="csv",
            file_size=100,
        )
        test_db_session.add(dataset)
        test_db_session.commit()
        
        response = client.patch(
            f"/api/v1/datasets/{dataset.id}",
            json={"name": "Updated", "description": "New description"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated"
        assert data["description"] == "New description"
    
    def test_delete_dataset(
        self,
        client: TestClient,
        test_db_session: Session,
        tmp_path: Path,
    ):
        """Test deleting a dataset."""
        file_path = tmp_path / "test.csv"
        file_path.write_text("input,output\na,b\n")
        
        dataset = Dataset(
            name="To Delete",
            filename="test.csv",
            file_path=str(file_path),
            file_type="csv",
            file_size=100,
        )
        test_db_session.add(dataset)
        test_db_session.commit()
        dataset_id = dataset.id
        
        response = client.delete(f"/api/v1/datasets/{dataset_id}")
        
        assert response.status_code == 204
        
        # Verify deletion
        response = client.get(f"/api/v1/datasets/{dataset_id}")
        assert response.status_code == 404


class TestTrainingJobEndpoints:
    """Tests for training job API endpoints."""
    
    @pytest.fixture
    def sample_dataset(self, test_db_session: Session) -> Dataset:
        """Create a sample dataset."""
        dataset = Dataset(
            name="Training Data",
            filename="data.csv",
            file_path="/path/data.csv",
            file_type="csv",
            file_size=1000,
            row_count=100,
            column_count=2,
        )
        test_db_session.add(dataset)
        test_db_session.commit()
        return dataset
    
    def test_create_training_job(
        self,
        client: TestClient,
        sample_dataset: Dataset,
    ):
        """Test creating a training job."""
        with patch("app.services.training_service.TrainingService._worker_pool") as mock_pool:
            mock_pool.is_running = False
            
            response = client.post(
                "/api/v1/jobs/",
                json={
                    "name": "Test Job",
                    "description": "Testing",
                    "dataset_id": sample_dataset.id,
                    "training_type": "qlora",
                    "batch_size": 8,
                    "epochs": 5,
                },
            )
        
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Job"
        assert data["status"] == "pending"
        assert data["batch_size"] == 8
    
    def test_create_training_job_with_unsloth(
        self,
        client: TestClient,
        sample_dataset: Dataset,
    ):
        """Test creating a training job with Unsloth type."""
        with patch("app.services.training_service.TrainingService._worker_pool") as mock_pool:
            mock_pool.is_running = False
            
            response = client.post(
                "/api/v1/jobs/",
                json={
                    "name": "Unsloth Test Job",
                    "description": "Testing Unsloth",
                    "dataset_id": sample_dataset.id,
                    "training_type": "unsloth",
                    "batch_size": 8,
                    "epochs": 5,
                },
            )
        
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Unsloth Test Job"
        assert data["status"] == "pending"
        assert data["training_type"] == "unsloth"
        assert data["batch_size"] == 8
    
    def test_create_training_job_invalid_dataset(self, client: TestClient):
        """Test creating a job with invalid dataset."""
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
    
    def test_list_training_jobs(
        self,
        client: TestClient,
        sample_dataset: Dataset,
        test_db_session: Session,
    ):
        """Test listing training jobs."""
        for i in range(5):
            job = TrainingJob(
                name=f"Job {i}",
                dataset_id=sample_dataset.id,
            )
            test_db_session.add(job)
        test_db_session.commit()
        
        response = client.get("/api/v1/jobs/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 5
    
    def test_list_training_jobs_filter_by_status(
        self,
        client: TestClient,
        sample_dataset: Dataset,
        test_db_session: Session,
    ):
        """Test filtering jobs by status."""
        for i in range(5):
            job = TrainingJob(
                name=f"Job {i}",
                dataset_id=sample_dataset.id,
                status=TrainingStatus.PENDING.value if i < 3 else TrainingStatus.COMPLETED.value,
            )
            test_db_session.add(job)
        test_db_session.commit()
        
        response = client.get("/api/v1/jobs/?status=pending")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
    
    def test_get_training_job(
        self,
        client: TestClient,
        sample_dataset: Dataset,
        test_db_session: Session,
    ):
        """Test getting a specific training job."""
        job = TrainingJob(
            name="Test Job",
            dataset_id=sample_dataset.id,
        )
        test_db_session.add(job)
        test_db_session.commit()
        
        response = client.get(f"/api/v1/jobs/{job.id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Job"
    
    def test_get_training_job_status(
        self,
        client: TestClient,
        sample_dataset: Dataset,
        test_db_session: Session,
    ):
        """Test getting job status."""
        job = TrainingJob(
            name="Status Test",
            dataset_id=sample_dataset.id,
            progress=50.0,
            current_epoch=2,
        )
        test_db_session.add(job)
        test_db_session.commit()
        
        response = client.get(f"/api/v1/jobs/{job.id}/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["progress"] == 50.0
        assert data["current_epoch"] == 2
    
    def test_cancel_training_job(
        self,
        client: TestClient,
        sample_dataset: Dataset,
        test_db_session: Session,
    ):
        """Test cancelling a training job."""
        job = TrainingJob(
            name="Cancel Test",
            dataset_id=sample_dataset.id,
            status=TrainingStatus.PENDING.value,
        )
        test_db_session.add(job)
        test_db_session.commit()
        
        with patch("app.services.training_service.TrainingService._worker_pool") as mock_pool:
            mock_pool.cancel_job = MagicMock()
            
            response = client.post(f"/api/v1/jobs/{job.id}/cancel")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cancelled"


class TestConfigEndpoints:
    """Tests for configuration API endpoints."""
    
    def test_get_config(
        self,
        client: TestClient,
        test_db_session: Session,
    ):
        """Test getting training configuration."""
        with patch("app.services.training_service.TrainingService._worker_pool") as mock_pool:
            mock_pool.get_status.return_value = {"workers": []}
            
            response = client.get("/api/v1/config/")
        
        assert response.status_code == 200
        data = response.json()
        assert "max_concurrent_workers" in data
        assert "default_model" in data
    
    def test_update_config(
        self,
        client: TestClient,
        test_db_session: Session,
    ):
        """Test updating training configuration."""
        with patch("app.services.training_service.TrainingService._worker_pool") as mock_pool:
            mock_pool.set_max_workers = MagicMock()
            
            response = client.patch(
                "/api/v1/config/",
                json={
                    "max_concurrent_workers": 8,
                    "default_batch_size": 16,
                },
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["max_concurrent_workers"] == 8
        assert data["default_batch_size"] == 16
    
    def test_get_config_includes_new_fields(
        self,
        client: TestClient,
        test_db_session: Session,
    ):
        """Test getting training configuration includes model_provider and model_api_url."""
        with patch("app.services.training_service.TrainingService._worker_pool") as mock_pool:
            mock_pool.get_status.return_value = {"workers": []}
            
            response = client.get("/api/v1/config/")
        
        assert response.status_code == 200
        data = response.json()
        assert "model_provider" in data
        assert "model_api_url" in data
        assert data["model_provider"] == "ollama"
        assert data["model_api_url"] == "http://localhost:11434"
    
    def test_update_config_model_provider(
        self,
        client: TestClient,
        test_db_session: Session,
    ):
        """Test updating model_provider field."""
        with patch("app.services.training_service.TrainingService._worker_pool") as mock_pool:
            mock_pool.set_max_workers = MagicMock()
            
            response = client.patch(
                "/api/v1/config/",
                json={
                    "model_provider": "lm_studio",
                },
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_provider"] == "lm_studio"
    
    def test_update_config_model_api_url(
        self,
        client: TestClient,
        test_db_session: Session,
    ):
        """Test updating model_api_url field."""
        with patch("app.services.training_service.TrainingService._worker_pool") as mock_pool:
            mock_pool.set_max_workers = MagicMock()
            
            response = client.patch(
                "/api/v1/config/",
                json={
                    "model_api_url": "http://localhost:1234",
                },
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_api_url"] == "http://localhost:1234"
    
    def test_update_config_both_provider_fields(
        self,
        client: TestClient,
        test_db_session: Session,
    ):
        """Test updating both model_provider and model_api_url together."""
        with patch("app.services.training_service.TrainingService._worker_pool") as mock_pool:
            mock_pool.set_max_workers = MagicMock()
            
            response = client.patch(
                "/api/v1/config/",
                json={
                    "model_provider": "lm_studio",
                    "model_api_url": "http://localhost:1234",
                },
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_provider"] == "lm_studio"
        assert data["model_api_url"] == "http://localhost:1234"
    
    def test_update_config_invalid_provider(
        self,
        client: TestClient,
        test_db_session: Session,
    ):
        """Test updating with invalid model_provider value."""
        with patch("app.services.training_service.TrainingService._worker_pool") as mock_pool:
            mock_pool.set_max_workers = MagicMock()
            
            response = client.patch(
                "/api/v1/config/",
                json={
                    "model_provider": "invalid_provider",
                },
            )
        
        assert response.status_code == 422  # Validation error
    
    def test_update_config_invalid_url(
        self,
        client: TestClient,
        test_db_session: Session,
    ):
        """Test updating with invalid model_api_url format."""
        with patch("app.services.training_service.TrainingService._worker_pool") as mock_pool:
            mock_pool.set_max_workers = MagicMock()
            
            response = client.patch(
                "/api/v1/config/",
                json={
                    "model_api_url": "not-a-valid-url",
                },
            )
        
        assert response.status_code == 422  # Validation error


class TestWorkerEndpoints:
    """Tests for worker management API endpoints."""
    
    def test_get_worker_status(
        self,
        client: TestClient,
        test_db_session: Session,
    ):
        """Test getting worker pool status."""
        with patch("app.services.training_service.TrainingService._worker_pool") as mock_pool:
            mock_pool.get_status.return_value = {
                "total_workers": 2,
                "active_workers": 2,
                "idle_workers": 1,
                "busy_workers": 1,
                "workers": [
                    {
                        "id": "worker-1",
                        "status": "idle",
                        "current_job_id": None,
                        "jobs_completed": 5,
                        "started_at": "2024-01-01T00:00:00",
                        "last_activity": None,
                    }
                ],
            }
            
            response = client.get("/api/v1/workers/")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_workers" in data
        assert "workers" in data
    
    def test_start_workers(
        self,
        client: TestClient,
        test_db_session: Session,
    ):
        """Test starting workers."""
        with patch("app.services.training_service.TrainingService._worker_pool") as mock_pool:
            mock_pool.start_workers = MagicMock()
            mock_pool.active_worker_count = 2
            mock_pool.get_status.return_value = {
                "total_workers": 2,
                "active_workers": 2,
                "idle_workers": 2,
                "busy_workers": 0,
                "workers": [],
            }
            
            response = client.post(
                "/api/v1/workers/",
                json={"action": "start", "worker_count": 2},
            )
        
        assert response.status_code == 200
        mock_pool.start_workers.assert_called_once_with(2)
    
    def test_stop_workers(
        self,
        client: TestClient,
        test_db_session: Session,
    ):
        """Test stopping workers."""
        with patch("app.services.training_service.TrainingService._worker_pool") as mock_pool:
            mock_pool.stop_all_workers = MagicMock()
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


class TestHuggingFaceEndpoints:
    """Tests for Hugging Face API endpoints."""
    
    def test_search_datasets(
        self,
        client: TestClient,
    ):
        """Test searching for datasets on Hugging Face."""
        mock_results = {
            "items": [
                {
                    "id": "squad",
                    "name": "squad",
                    "author": "rajpurkar",
                    "description": "Stanford QA Dataset",
                    "downloads": 100000,
                    "likes": 500,
                    "tags": ["nlp", "qa"],
                    "last_modified": "2024-01-01",
                    "private": False,
                }
            ],
            "query": "qa",
            "limit": 20,
            "offset": 0,
        }
        
        with patch("app.api.endpoints.huggingface.HuggingFaceService") as mock_service_class:
            mock_service = MagicMock()
            mock_service.search_datasets = AsyncMock(return_value=mock_results)
            mock_service_class.return_value = mock_service
            
            response = client.get("/api/v1/huggingface/search?query=qa")
        
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "qa"
        assert len(data["items"]) == 1
        assert data["items"][0]["id"] == "squad"
    
    def test_search_datasets_with_params(
        self,
        client: TestClient,
    ):
        """Test searching with limit and offset."""
        mock_results = {
            "items": [],
            "query": "test",
            "limit": 10,
            "offset": 20,
        }
        
        with patch("app.api.endpoints.huggingface.HuggingFaceService") as mock_service_class:
            mock_service = MagicMock()
            mock_service.search_datasets = AsyncMock(return_value=mock_results)
            mock_service_class.return_value = mock_service
            
            response = client.get("/api/v1/huggingface/search?query=test&limit=10&offset=20")
        
        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 10
        assert data["offset"] == 20
    
    def test_search_datasets_error(
        self,
        client: TestClient,
    ):
        """Test search with error handling."""
        with patch("app.api.endpoints.huggingface.HuggingFaceService") as mock_service_class:
            mock_service = MagicMock()
            mock_service.search_datasets = AsyncMock(side_effect=ValueError("Search failed"))
            mock_service_class.return_value = mock_service
            
            response = client.get("/api/v1/huggingface/search?query=test")
        
        assert response.status_code == 500
    
    def test_get_dataset_info(
        self,
        client: TestClient,
    ):
        """Test getting dataset information."""
        mock_info = {
            "id": "squad",
            "name": "squad",
            "author": "rajpurkar",
            "description": "Stanford QA Dataset",
            "downloads": 100000,
            "likes": 500,
            "tags": ["nlp"],
            "private": False,
        }
        
        with patch("app.api.endpoints.huggingface.HuggingFaceService") as mock_service_class:
            mock_service = MagicMock()
            mock_service.get_dataset_info = AsyncMock(return_value=mock_info)
            mock_service_class.return_value = mock_service
            
            response = client.get("/api/v1/huggingface/datasets/squad")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "squad"
        assert data["author"] == "rajpurkar"
    
    def test_get_dataset_info_not_found(
        self,
        client: TestClient,
    ):
        """Test getting non-existent dataset info."""
        with patch("app.api.endpoints.huggingface.HuggingFaceService") as mock_service_class:
            mock_service = MagicMock()
            mock_service.get_dataset_info = AsyncMock(side_effect=ValueError("Not found"))
            mock_service_class.return_value = mock_service
            
            response = client.get("/api/v1/huggingface/datasets/nonexistent")
        
        assert response.status_code == 404
    
    def test_get_dataset_configs(
        self,
        client: TestClient,
    ):
        """Test getting dataset configurations."""
        with patch("app.api.endpoints.huggingface.HuggingFaceService") as mock_service_class:
            mock_service = MagicMock()
            mock_service.list_dataset_configs = AsyncMock(return_value=["en", "fr", "de"])
            mock_service_class.return_value = mock_service
            
            response = client.get("/api/v1/huggingface/datasets/test/configs")
        
        assert response.status_code == 200
        data = response.json()
        assert "configs" in data
        assert "en" in data["configs"]
    
    def test_download_dataset(
        self,
        client: TestClient,
        test_db_session: Session,
        tmp_path: Path,
    ):
        """Test downloading a dataset from Hugging Face."""
        # Create mock JSON file
        json_file = tmp_path / "hf_test_dataset_train.json"
        json_file.write_text('[{"text": "sample", "label": 1}]')
        
        with patch("app.api.endpoints.huggingface.HuggingFaceService") as mock_service_class:
            mock_service = MagicMock()
            mock_service.download_dataset.return_value = (
                json_file,
                1,
                ["text", "label"],
            )
            mock_service_class.return_value = mock_service
            
            response = client.post(
                "/api/v1/huggingface/download",
                json={
                    "dataset_id": "test/dataset",
                    "name": "Test Dataset",
                    "split": "train",
                    "max_rows": 1000,
                },
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["dataset_id"] == "test/dataset"
        assert data["row_count"] == 1
        assert "message" in data
        
        # Verify dataset was created in database
        dataset = test_db_session.query(Dataset).filter_by(name="Test Dataset").first()
        assert dataset is not None
        assert dataset.file_type == "json"
    
    def test_download_dataset_with_config(
        self,
        client: TestClient,
        test_db_session: Session,
        tmp_path: Path,
    ):
        """Test downloading dataset with configuration."""
        json_file = tmp_path / "hf_test_dataset_en_train.json"
        json_file.write_text('[{"question": "q", "answer": "a"}]')
        
        with patch("app.api.endpoints.huggingface.HuggingFaceService") as mock_service_class:
            mock_service = MagicMock()
            mock_service.download_dataset.return_value = (
                json_file,
                1,
                ["question", "answer"],
            )
            mock_service_class.return_value = mock_service
            
            response = client.post(
                "/api/v1/huggingface/download",
                json={
                    "dataset_id": "test/dataset",
                    "split": "train",
                    "config": "en",
                },
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["dataset_id"] == "test/dataset"
    
    def test_download_dataset_error(
        self,
        client: TestClient,
        tmp_path: Path,
    ):
        """Test download with error handling."""
        with patch("app.api.endpoints.huggingface.HuggingFaceService") as mock_service_class:
            mock_service = MagicMock()
            mock_service.download_dataset.side_effect = ValueError("Download failed")
            mock_service_class.return_value = mock_service
            
            response = client.post(
                "/api/v1/huggingface/download",
                json={
                    "dataset_id": "nonexistent/dataset",
                    "split": "train",
                },
            )
        
        assert response.status_code == 400
        assert "Download failed" in response.json()["detail"]


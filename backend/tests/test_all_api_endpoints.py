"""
Comprehensive tests for ALL API endpoints.

Following TDD methodology - tests define expected behavior for every endpoint.
"""

import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.project import Project, ProjectStatus
from app.models.dataset import Dataset
from app.models.training_job import TrainingJob, TrainingStatus
from app.models.training_config import TrainingConfig
from app.models.downloaded_model import DownloadedModel


class TestProjectsAPIEndpoints:
    """Comprehensive tests for all project API endpoints."""
    
    @pytest.fixture
    def sample_datasets(self, test_db_session: Session):
        """Create sample datasets for testing."""
        datasets = []
        for i in range(3):
            dataset = Dataset(
                name=f"Dataset {i+1}",
                filename=f"test{i+1}.csv",
                file_path=f"/data/user/test{i+1}/test{i+1}.csv",
                file_type="csv",
                file_size=1024,
                row_count=100,
                column_count=2,
                columns='["input", "output"]',
            )
            test_db_session.add(dataset)
            datasets.append(dataset)
        test_db_session.commit()
        return datasets
    
    def test_create_project_endpoint(self, client: TestClient, sample_datasets):
        """Test POST /api/v1/projects/ - Create project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("app.services.output_directory_service.OutputDirectoryService.validate_directory") as mock_validate:
                mock_validate.return_value = {"valid": True, "writable": True}
                
                payload = {
                    "name": "Test Project",
                    "description": "Test description",
                    "base_model": "meta-llama/Llama-3.2-3B-Instruct",
                    "training_type": "qlora",
                    "output_directory": f"{tmpdir}/test-project",
                    "traits": [
                        {
                            "trait_type": "reasoning",
                            "datasets": [{"dataset_id": sample_datasets[0].id, "percentage": 50.0}]
                        }
                    ]
                }
                response = client.post("/api/v1/projects/", json=payload)
                
                assert response.status_code == 201
                data = response.json()
                assert data["name"] == "Test Project"
                assert data["status"] == "pending"
    
    def test_list_projects_endpoint(self, client: TestClient, test_db_session: Session):
        """Test GET /api/v1/projects/ - List projects."""
        # Create test projects
        for i in range(3):
            project = Project(
                name=f"Project {i}",
                base_model="llama3.2:3b",
                training_type="qlora",
                output_directory=f"/output/project{i}",
                status=ProjectStatus.PENDING.value,
            )
            test_db_session.add(project)
        test_db_session.commit()
        
        response = client.get("/api/v1/projects/")
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 3
        assert data["total"] == 3
    
    def test_get_project_endpoint(self, client: TestClient, test_db_session: Session):
        """Test GET /api/v1/projects/{project_id} - Get project."""
        project = Project(
            name="Test Project",
            base_model="llama3.2:3b",
            training_type="qlora",
            output_directory="/output/test",
            status=ProjectStatus.PENDING.value,
        )
        test_db_session.add(project)
        test_db_session.commit()
        
        response = client.get(f"/api/v1/projects/{project.id}")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Project"
    
    def test_update_project_endpoint(self, client: TestClient, test_db_session: Session):
        """Test PATCH /api/v1/projects/{project_id} - Update project."""
        project = Project(
            name="Original Name",
            base_model="llama3.2:3b",
            training_type="qlora",
            output_directory="/output/test",
            status=ProjectStatus.PENDING.value,
        )
        test_db_session.add(project)
        test_db_session.commit()
        
        response = client.patch(
            f"/api/v1/projects/{project.id}",
            json={"name": "Updated Name", "description": "Updated description"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Name"
        assert data["description"] == "Updated description"
    
    def test_delete_project_endpoint(self, client: TestClient, test_db_session: Session):
        """Test DELETE /api/v1/projects/{project_id} - Delete project."""
        project = Project(
            name="To Delete",
            base_model="llama3.2:3b",
            training_type="qlora",
            output_directory="/output/test",
            status=ProjectStatus.PENDING.value,
        )
        test_db_session.add(project)
        test_db_session.commit()
        project_id = project.id
        
        response = client.delete(f"/api/v1/projects/{project_id}")
        assert response.status_code == 204
        
        # Verify deleted
        response = client.get(f"/api/v1/projects/{project_id}")
        assert response.status_code == 404
    
    def test_start_project_endpoint(self, client: TestClient, test_db_session: Session):
        """Test POST /api/v1/projects/{project_id}/start - Start project training."""
        project = Project(
            name="Test Project",
            base_model="meta-llama/Llama-3.2-3B-Instruct",
            training_type="qlora",
            output_directory="/output/test",
            status=ProjectStatus.PENDING.value,
        )
        test_db_session.add(project)
        test_db_session.commit()
        
        with patch("app.services.model_resolution_service.ModelResolutionService.is_model_available") as mock_available:
            mock_available.return_value = True
            with patch("app.services.training_service.TrainingService.worker_pool") as mock_pool:
                mock_pool.is_running = True
                mock_pool.queue_project = MagicMock(return_value=True)
                
                response = client.post(f"/api/v1/projects/{project.id}/start")
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "pending"
                mock_pool.queue_project.assert_called_once_with(project.id)
    
    def test_validate_output_dir_endpoint(self, client: TestClient):
        """Test POST /api/v1/projects/validate-output-dir - Validate output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            response = client.post(
                "/api/v1/projects/validate-output-dir",
                json={"output_directory": tmpdir}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is True
            assert data["writable"] is True
    
    def test_validate_model_endpoint(self, client: TestClient):
        """Test POST /api/v1/projects/validate-model - Validate model."""
        with patch("app.services.model_resolution_service.ModelResolutionService.is_model_available") as mock_available:
            mock_available.return_value = True
            with patch("app.services.model_resolution_service.ModelResolutionService.resolve_model_path") as mock_path:
                mock_path.return_value = "/path/to/model"
                
                response = client.post(
                    "/api/v1/projects/validate-model",
                    json={"model_name": "meta-llama/Llama-3.2-3B-Instruct"}
                )
                assert response.status_code == 200
                data = response.json()
                assert data["available"] is True
    
    def test_list_available_models_endpoint(self, client: TestClient, test_db_session: Session):
        """Test GET /api/v1/projects/models/available - List available models."""
        from app.models.downloaded_model import DownloadedModel
        
        # Create test downloaded models
        model1 = DownloadedModel(
            model_id="model1",
            name="model1",
            author="test",
            local_path="./data/models/model1",
            file_size=1000000,
            downloaded_at=datetime.utcnow(),
        )
        model2 = DownloadedModel(
            model_id="model2",
            name="model2",
            author="test",
            local_path="./data/models/model2",
            file_size=2000000,
            downloaded_at=datetime.utcnow(),
        )
        test_db_session.add(model1)
        test_db_session.add(model2)
        test_db_session.commit()
        
        # Mock is_model_available to return True for both
        with patch("app.services.model_resolution_service.ModelResolutionService.is_model_available") as mock_available:
            mock_available.return_value = True
            
            response = client.get("/api/v1/projects/models/available")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 2
            assert "model1" in data
            assert "model2" in data


class TestDatasetsAPIEndpoints:
    """Comprehensive tests for all dataset API endpoints."""
    
    def test_upload_dataset_endpoint(self, client: TestClient, sample_csv_content: bytes):
        """Test POST /api/v1/datasets/ - Upload dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("app.services.dataset_service.settings") as mock_settings:
                mock_settings.allowed_extensions = {".csv", ".json"}
                mock_settings.max_upload_size = 100 * 1024 * 1024
                mock_settings.get_dataset_path.return_value = Path(tmpdir) / "user" / "test"
                
                response = client.post(
                    "/api/v1/datasets/",
                    files={"file": ("test.csv", sample_csv_content, "text/csv")},
                    data={"name": "Test Dataset", "description": "Test"}
                )
                assert response.status_code == 201
                data = response.json()
                assert data["name"] == "Test Dataset"
                assert data["file_type"] == "csv"
    
    def test_list_datasets_endpoint(self, client: TestClient, test_db_session: Session):
        """Test GET /api/v1/datasets/ - List datasets."""
        for i in range(5):
            dataset = Dataset(
                name=f"Dataset {i}",
                filename=f"data{i}.csv",
                file_path=f"/data/user/data{i}/data{i}.csv",
                file_type="csv",
                file_size=100,
            )
            test_db_session.add(dataset)
        test_db_session.commit()
        
        response = client.get("/api/v1/datasets/")
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 5
    
    def test_get_dataset_endpoint(self, client: TestClient, test_db_session: Session):
        """Test GET /api/v1/datasets/{dataset_id} - Get dataset."""
        dataset = Dataset(
            name="Test Dataset",
            filename="test.csv",
            file_path="/data/user/test/test.csv",
            file_type="csv",
            file_size=1024,
        )
        test_db_session.add(dataset)
        test_db_session.commit()
        
        response = client.get(f"/api/v1/datasets/{dataset.id}")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Dataset"
    
    def test_update_dataset_endpoint(self, client: TestClient, test_db_session: Session):
        """Test PATCH /api/v1/datasets/{dataset_id} - Update dataset."""
        dataset = Dataset(
            name="Original",
            filename="test.csv",
            file_path="/data/user/test/test.csv",
            file_type="csv",
            file_size=1024,
        )
        test_db_session.add(dataset)
        test_db_session.commit()
        
        response = client.patch(
            f"/api/v1/datasets/{dataset.id}",
            json={"name": "Updated", "description": "New description"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated"
    
    def test_delete_dataset_endpoint(self, client: TestClient, test_db_session: Session):
        """Test DELETE /api/v1/datasets/{dataset_id} - Delete dataset."""
        dataset = Dataset(
            name="To Delete",
            filename="test.csv",
            file_path="/data/user/test/test.csv",
            file_type="csv",
            file_size=1024,
        )
        test_db_session.add(dataset)
        test_db_session.commit()
        dataset_id = dataset.id
        
        response = client.delete(f"/api/v1/datasets/{dataset_id}")
        assert response.status_code == 204
        
        response = client.get(f"/api/v1/datasets/{dataset_id}")
        assert response.status_code == 404
    
    def test_scan_datasets_endpoint(self, client: TestClient):
        """Test POST /api/v1/datasets/scan - Scan for datasets."""
        response = client.post("/api/v1/datasets/scan")
        assert response.status_code == 200
        data = response.json()
        assert "scanned" in data
        assert "added" in data
        assert "skipped" in data


class TestTrainingJobsAPIEndpoints:
    """Comprehensive tests for all training job API endpoints."""
    
    @pytest.fixture
    def sample_dataset(self, test_db_session: Session):
        """Create sample dataset for testing."""
        dataset = Dataset(
            name="Test Dataset",
            filename="test.csv",
            file_path="/data/user/test/test.csv",
            file_type="csv",
            file_size=1024,
            row_count=100,
        )
        test_db_session.add(dataset)
        test_db_session.commit()
        return dataset
    
    def test_create_training_job_endpoint(self, client: TestClient, sample_dataset):
        """Test POST /api/v1/jobs/ - Create training job."""
        with patch("app.services.training_service.TrainingService.worker_pool") as mock_pool:
            mock_pool.is_running = False
            
            payload = {
                "name": "Test Job",
                "dataset_id": sample_dataset.id,
                "training_type": "qlora",
                "batch_size": 4,
                "epochs": 3,
            }
            response = client.post("/api/v1/jobs/", json=payload)
            assert response.status_code == 201
            data = response.json()
            assert data["name"] == "Test Job"
            assert data["status"] == "pending"
    
    def test_list_training_jobs_endpoint(self, client: TestClient, test_db_session: Session, sample_dataset):
        """Test GET /api/v1/jobs/ - List training jobs."""
        for i in range(3):
            job = TrainingJob(
                name=f"Job {i}",
                dataset_id=sample_dataset.id,
                training_type=TrainingType.QLORA.value,
                model_name="llama3.2:3b",
                status=TrainingStatus.PENDING.value,
            )
            test_db_session.add(job)
        test_db_session.commit()
        
        response = client.get("/api/v1/jobs/")
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 3
    
    def test_get_training_job_endpoint(self, client: TestClient, test_db_session: Session, sample_dataset):
        """Test GET /api/v1/jobs/{job_id} - Get training job."""
        job = TrainingJob(
            name="Test Job",
            dataset_id=sample_dataset.id,
            training_type=TrainingType.QLORA.value,
            model_name="llama3.2:3b",
            status=TrainingStatus.PENDING.value,
        )
        test_db_session.add(job)
        test_db_session.commit()
        
        response = client.get(f"/api/v1/jobs/{job.id}")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Job"
    
    def test_start_training_job_endpoint(self, client: TestClient, test_db_session: Session, sample_dataset):
        """Test POST /api/v1/jobs/{job_id}/start - Start training job."""
        job = TrainingJob(
            name="Test Job",
            dataset_id=sample_dataset.id,
            training_type=TrainingType.QLORA.value,
            model_name="llama3.2:3b",
            status=TrainingStatus.PENDING.value,
        )
        test_db_session.add(job)
        test_db_session.commit()
        
        with patch("app.services.training_service.TrainingService.worker_pool") as mock_pool:
            mock_pool.is_running = True
            mock_pool.submit_job = MagicMock(return_value=True)
            
            response = client.post(f"/api/v1/jobs/{job.id}/start")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "pending"
    
    def test_cancel_training_job_endpoint(self, client: TestClient, test_db_session: Session, sample_dataset):
        """Test POST /api/v1/jobs/{job_id}/cancel - Cancel training job."""
        job = TrainingJob(
            name="Test Job",
            dataset_id=sample_dataset.id,
            training_type=TrainingType.QLORA.value,
            model_name="llama3.2:3b",
            status=TrainingStatus.RUNNING.value,
        )
        test_db_session.add(job)
        test_db_session.commit()
        
        with patch("app.services.training_service.TrainingService.worker_pool") as mock_pool:
            mock_pool.cancel_job = MagicMock(return_value=True)
            
            response = client.post(f"/api/v1/jobs/{job.id}/cancel")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "cancelled"


class TestModelsAPIEndpoints:
    """Comprehensive tests for all models API endpoints."""
    
    def test_search_models_endpoint(self, client: TestClient):
        """Test GET /api/v1/models/search - Search models."""
        with patch("app.services.huggingface_service.HuggingFaceService.search_models") as mock_search:
            mock_search.return_value = {
                "items": [{"id": "model1", "name": "Model 1"}],
                "query": "test",
                "limit": 10,
                "offset": 0,
            }
            
            response = client.get("/api/v1/models/search?query=test")
            assert response.status_code == 200
            data = response.json()
            assert len(data["items"]) == 1
    
    def test_list_local_models_endpoint(self, client: TestClient, test_db_session: Session):
        """Test GET /api/v1/models/ - List local models."""
        model = DownloadedModel(
            model_id="test/model",
            name="Test Model",
            author="test",
            local_path="/path/to/model",
            file_size=1024,
        )
        test_db_session.add(model)
        test_db_session.commit()
        
        response = client.get("/api/v1/models/")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
    
    def test_scan_models_endpoint(self, client: TestClient):
        """Test POST /api/v1/models/scan - Scan for models."""
        response = client.post("/api/v1/models/scan")
        assert response.status_code == 200
        data = response.json()
        assert "scanned" in data
        assert "added" in data


class TestWorkersAPIEndpoints:
    """Comprehensive tests for all workers API endpoints."""
    
    def test_get_worker_status_endpoint(self, client: TestClient):
        """Test GET /api/v1/workers/ - Get worker status."""
        with patch("app.services.training_service.TrainingService.worker_pool") as mock_pool:
            mock_pool.get_status.return_value = {
                "total_workers": 2,
                "active_workers": 2,
                "idle_workers": 1,
                "busy_workers": 1,
                "max_workers": 8,
                "is_running": True,
                "queue_size": 0,
                "workers": [],
            }
            
            response = client.get("/api/v1/workers/")
            assert response.status_code == 200
            data = response.json()
            assert data["total_workers"] == 2
    
    def test_control_workers_start_endpoint(self, client: TestClient):
        """Test POST /api/v1/workers/ - Start workers."""
        with patch("app.services.training_service.TrainingService") as mock_service:
            mock_instance = MagicMock()
            mock_service.return_value = mock_instance
            mock_instance.start_workers = MagicMock()
            mock_instance.get_worker_pool_status.return_value = {
                "total_workers": 1,
                "active_workers": 1,
                "is_running": True,
            }
            
            response = client.post(
                "/api/v1/workers/",
                json={"action": "start", "worker_count": 1}
            )
            assert response.status_code == 200
    
    def test_control_workers_stop_endpoint(self, client: TestClient):
        """Test POST /api/v1/workers/ - Stop workers."""
        with patch("app.services.training_service.TrainingService") as mock_service:
            mock_instance = MagicMock()
            mock_service.return_value = mock_instance
            mock_instance.stop_workers = MagicMock()
            mock_instance.get_worker_pool_status.return_value = {
                "total_workers": 0,
                "active_workers": 0,
                "is_running": False,
            }
            
            response = client.post(
                "/api/v1/workers/",
                json={"action": "stop"}
            )
            assert response.status_code == 200


class TestConfigAPIEndpoints:
    """Comprehensive tests for all config API endpoints."""
    
    def test_get_config_endpoint(self, client: TestClient, test_db_session: Session):
        """Test GET /api/v1/config/ - Get config."""
        config = TrainingConfig()
        test_db_session.add(config)
        test_db_session.commit()
        
        response = client.get("/api/v1/config/")
        assert response.status_code == 200
        data = response.json()
        assert "max_concurrent_workers" in data
    
    def test_update_config_endpoint(self, client: TestClient, test_db_session: Session):
        """Test PATCH /api/v1/config/ - Update config."""
        config = TrainingConfig()
        test_db_session.add(config)
        test_db_session.commit()
        
        response = client.patch(
            "/api/v1/config/",
            json={"max_concurrent_workers": 8, "default_batch_size": 8}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["max_concurrent_workers"] == 8
    
    def test_get_gpus_endpoint(self, client: TestClient):
        """Test GET /api/v1/config/gpus - Get GPUs."""
        with patch("app.services.gpu_service.GPUService.get_available_gpus") as mock_gpus:
            mock_gpus.return_value = []
            
            response = client.get("/api/v1/config/gpus")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)

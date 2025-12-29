"""
Tests for project API endpoints.

Tests project CRUD operations, validation endpoints, and model availability checks.
"""

import pytest
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.project import Project, ProjectStatus
from app.models.dataset import Dataset
from app.models.training_config import TrainingConfig


class TestProjectAPI:
    """Tests for project API endpoints."""
    
    @pytest.fixture
    def sample_datasets(self, test_db_session: Session):
        """Create sample datasets for testing."""
        datasets = []
        for i in range(3):
            dataset = Dataset(
                name=f"Dataset {i+1}",
                filename=f"test{i+1}.csv",
                file_path=f"/uploads/test{i+1}.csv",
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
    
    def test_create_project(self, client: TestClient, sample_datasets):
        """Test creating a new project via API."""
        payload = {
            "name": "Test Project",
            "description": "A test project",
            "base_model": "meta-llama/Llama-3.2-3B-Instruct",
            "training_type": "qlora",
            "max_rows": 50000,
            "output_directory": "/tmp/test-project",
            "traits": [
                {
                    "trait_type": "reasoning",
                    "datasets": [
                        {"dataset_id": sample_datasets[0].id, "percentage": 100.0}
                    ],
                },
            ],
        }
        
        response = client.post("/api/v1/projects/", json=payload)
        
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Project"
        assert data["status"] == ProjectStatus.PENDING.value
        assert "id" in data
    
    def test_create_project_invalid_data(self, client: TestClient, sample_datasets):
        """Test creating project with invalid data."""
        payload = {
            "name": "Test Project",
            "base_model": "llama3.2:3b",
            "training_type": "qlora",
            "max_rows": 50000,
            "output_directory": "/tmp/test-project",
            "traits": [
                {
                    "trait_type": "reasoning",
                    "datasets": [
                        {"dataset_id": sample_datasets[0].id, "percentage": 50.0},  # Not 100%
                    ],
                },
            ],
        }
        
        response = client.post("/api/v1/projects/", json=payload)
        
        assert response.status_code == 400
        assert "100%" in response.json()["detail"].lower()
    
    def test_get_project(self, client: TestClient, test_db_session: Session, sample_datasets):
        """Test retrieving a project by ID."""
        # Create a project directly in DB for testing
        from app.services.project_service import ProjectService
        service = ProjectService(db=test_db_session)
        
        project = service.create_project({
            "name": "Test Project",
            "base_model": "llama3.2:3b",
            "training_type": "qlora",
            "max_rows": 50000,
            "output_directory": "/tmp/test-project",
            "traits": [
                {
                    "trait_type": "reasoning",
                    "datasets": [
                        {"dataset_id": sample_datasets[0].id, "percentage": 100.0}
                    ],
                },
            ],
        })
        
        response = client.get(f"/api/v1/projects/{project.id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == project.id
        assert data["name"] == "Test Project"
    
    def test_list_projects(self, client: TestClient, test_db_session: Session, sample_datasets):
        """Test listing all projects."""
        from app.services.project_service import ProjectService
        service = ProjectService(db=test_db_session)
        
        # Create multiple projects
        for i in range(3):
            service.create_project({
                "name": f"Project {i+1}",
                "base_model": "llama3.2:3b",
                "training_type": "qlora",
                "max_rows": 50000,
                "output_directory": f"/tmp/project{i+1}",
                "traits": [
                    {
                        "trait_type": "reasoning",
                        "datasets": [
                            {"dataset_id": sample_datasets[0].id, "percentage": 100.0}
                        ],
                    },
                ],
            })
        
        response = client.get("/api/v1/projects/")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) >= 3
    
    def test_update_project(self, client: TestClient, test_db_session: Session, sample_datasets):
        """Test updating a project."""
        from app.services.project_service import ProjectService
        service = ProjectService(db=test_db_session)
        
        project = service.create_project({
            "name": "Original Name",
            "base_model": "llama3.2:3b",
            "training_type": "qlora",
            "max_rows": 50000,
            "output_directory": "/tmp/test-project",
            "traits": [
                {
                    "trait_type": "reasoning",
                    "datasets": [
                        {"dataset_id": sample_datasets[0].id, "percentage": 100.0}
                    ],
                },
            ],
        })
        
        response = client.patch(
            f"/api/v1/projects/{project.id}",
            json={"name": "Updated Name", "description": "Updated description"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Name"
        assert data["description"] == "Updated description"
    
    def test_validate_output_directory(self, client: TestClient, test_db_session: Session):
        """Test validating output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            response = client.post(
                "/api/v1/projects/validate-output-dir",
                json={"output_directory": tmpdir},
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is True
            assert data["writable"] is True
    
    def test_validate_output_directory_invalid(self, client: TestClient):
        """Test validating invalid output directory."""
        response = client.post(
            "/api/v1/projects/validate-output-dir",
            json={"output_directory": "/nonexistent/path/that/does/not/exist"},
        )
        
        assert response.status_code == 400
    
    def test_validate_model_availability(self, client: TestClient):
        """Test checking model availability."""
        response = client.post(
            "/api/v1/projects/validate-model",
            json={"model_name": "meta-llama/Llama-3.2-3B-Instruct"},
        )
        
        # Should return 200 even if model not found (just reports availability)
        assert response.status_code == 200
        data = response.json()
        assert "available" in data
        assert isinstance(data["available"], bool)
    
    def test_start_project_training(self, client: TestClient, test_db_session: Session, sample_datasets):
        """Test starting project training."""
        from app.services.project_service import ProjectService
        service = ProjectService(db=test_db_session)
        
        project = service.create_project({
            "name": "Test Project",
            "base_model": "llama3.2:3b",
            "training_type": "qlora",
            "max_rows": 50000,
            "output_directory": "/tmp/test-project",
            "traits": [
                {
                    "trait_type": "reasoning",
                    "datasets": [
                        {"dataset_id": sample_datasets[0].id, "percentage": 100.0}
                    ],
                },
            ],
        })
        
        response = client.post(f"/api/v1/projects/{project.id}/start")
        
        # May succeed or fail based on model availability
        assert response.status_code in [200, 400]
    
    def test_validate_trained_model(self, client: TestClient, test_db_session: Session, sample_datasets):
        """Test validating a trained model."""
        from app.services.project_service import ProjectService
        service = ProjectService(db=test_db_session)
        
        project = service.create_project({
            "name": "Test Project",
            "base_model": "llama3.2:3b",
            "training_type": "qlora",
            "max_rows": 50000,
            "output_directory": "/tmp/test-project",
            "traits": [
                {
                    "trait_type": "reasoning",
                    "datasets": [
                        {"dataset_id": sample_datasets[0].id, "percentage": 100.0}
                    ],
                },
            ],
        })
        
        # Try to validate - may fail if model doesn't exist
        response = client.post(f"/api/v1/projects/{project.id}/validate")
        
        # Should return some response
        assert response.status_code in [200, 400, 404]
    
    def test_list_available_models(self, client: TestClient):
        """Test listing available local models."""
        response = client.get("/api/v1/projects/models/available")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)  # List of model names

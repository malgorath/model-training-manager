"""
Comprehensive tests for training jobs API including projects.

Following TDD methodology: Tests ensure the training jobs API correctly
includes both TrainingJob and Project records, handles conversion, and
maintains backward compatibility.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.training_job import TrainingJob, TrainingStatus
from app.models.project import Project, ProjectStatus
from app.models.dataset import Dataset
from app.models.training_config import TrainingConfig


@pytest.fixture
def sample_dataset(test_db_session: Session) -> Dataset:
    """Create a sample dataset for testing."""
    dataset = Dataset(
        name="Test Dataset",
        filename="test.csv",
        file_path="./data/user/test.csv",
        file_type="csv",
        file_size=1024,
        row_count=100,
        column_count=2,
        columns='["input", "output"]',
    )
    test_db_session.add(dataset)
    test_db_session.commit()
    test_db_session.refresh(dataset)
    return dataset


@pytest.fixture
def training_config(test_db_session: Session) -> TrainingConfig:
    """Create training config for testing."""
    config = TrainingConfig(
        default_batch_size=4,
        default_learning_rate=2e-4,
        default_epochs=3,
        default_lora_r=16,
        default_lora_alpha=32,
        default_lora_dropout=0.05,
    )
    test_db_session.add(config)
    test_db_session.commit()
    test_db_session.refresh(config)
    return config


@pytest.fixture
def training_job(test_db_session: Session, sample_dataset: Dataset) -> TrainingJob:
    """Create a training job for testing."""
    job = TrainingJob(
        name="Test Training Job",
        training_type="qlora",
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        dataset_id=sample_dataset.id,
        status=TrainingStatus.RUNNING.value,
        progress=50.0,
        current_epoch=2,
        epochs=4,
    )
    test_db_session.add(job)
    test_db_session.commit()
    test_db_session.refresh(job)
    return job


@pytest.fixture
def training_project(test_db_session: Session) -> Project:
    """Create a project in training status for testing."""
    project = Project(
        name="Test Project",
        base_model="meta-llama/Llama-3.2-3B-Instruct",
        training_type="qlora",
        output_directory="./output/test-project",
        status=ProjectStatus.RUNNING.value,
        progress=30.0,
        current_epoch=1,
    )
    test_db_session.add(project)
    test_db_session.commit()
    test_db_session.refresh(project)
    return project


class TestTrainingJobsAPIComprehensive:
    """Comprehensive tests for training jobs API."""
    
    def test_list_jobs_includes_training_jobs(
        self, client: TestClient, training_job: TrainingJob
    ):
        """
        Test that list_jobs returns TrainingJob records.
        
        Verifies:
        - TrainingJob records are returned
        - Response format is correct
        """
        response = client.get("/api/v1/jobs/", params={"page": 1, "page_size": 10})
        
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert "pages" in data
        assert data["total"] >= 1
        assert any(j["id"] == training_job.id for j in data["items"])
    
    def test_list_jobs_includes_projects(
        self, client: TestClient, training_project: Project, training_config: TrainingConfig
    ):
        """
        Test that list_jobs includes Project records that are training.
        
        Verifies:
        - Projects with training status are included
        - Projects are converted to TrainingJobResponse format
        - All required fields are present
        """
        response = client.get("/api/v1/jobs/", params={"page": 1, "page_size": 10})
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        
        # Find the project in the response
        project_item = next((j for j in data["items"] if j["name"] == training_project.name), None)
        assert project_item is not None, "Project should be included in jobs list"
        
        # Verify conversion format
        assert project_item["id"] == training_project.id
        assert project_item["status"] == training_project.status
        assert project_item["model_name"] == training_project.base_model
        assert project_item["training_type"] == training_project.training_type
        assert project_item["progress"] == training_project.progress
        assert project_item["current_epoch"] == training_project.current_epoch
        assert "batch_size" in project_item
        assert "learning_rate" in project_item
        assert "epochs" in project_item
    
    def test_list_jobs_shows_all_statuses_by_default(
        self, client: TestClient, training_job: TrainingJob, training_project: Project
    ):
        """
        Test that list_jobs shows jobs with all statuses by default.
        
        Verifies:
        - No status filter returns all jobs
        - Both TrainingJob and Project records are included
        """
        response = client.get("/api/v1/jobs/", params={"page": 1, "page_size": 10})
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 2  # Should include both job and project
        
        # Verify both are present
        job_ids = [j["id"] for j in data["items"]]
        assert training_job.id in job_ids or any(j["name"] == training_job.name for j in data["items"])
        assert training_project.id in job_ids or any(j["name"] == training_project.name for j in data["items"])
    
    def test_list_jobs_filters_by_status_running(
        self, client: TestClient, training_job: TrainingJob, training_project: Project
    ):
        """
        Test that status filter works for both TrainingJob and Project.
        
        Verifies:
        - Filtering by status='running' returns running jobs and projects
        - All returned items have status 'running'
        """
        response = client.get(
            "/api/v1/jobs/",
            params={"page": 1, "page_size": 10, "status": "running"}
        )
        
        assert response.status_code == 200
        data = response.json()
        # All returned items should have status 'running'
        for item in data["items"]:
            assert item["status"] == "running"
    
    def test_list_jobs_filters_by_status_failed(
        self, client: TestClient, test_db_session: Session, sample_dataset: Dataset
    ):
        """
        Test that status filter works for failed jobs and projects.
        
        Verifies:
        - Filtering by status='failed' returns failed jobs and projects
        """
        # Create a failed training job
        failed_job = TrainingJob(
            name="Failed Job",
            training_type="qlora",
            model_name="test-model",
            dataset_id=sample_dataset.id,
            status=TrainingStatus.FAILED.value,
            error_message="Test error",
        )
        test_db_session.add(failed_job)
        
        # Create a failed project
        failed_project = Project(
            name="Failed Project",
            base_model="test-model",
            training_type="qlora",
            output_directory="./output/failed",
            status=ProjectStatus.FAILED.value,
            error_message="Test error",
        )
        test_db_session.add(failed_project)
        test_db_session.commit()
        
        response = client.get(
            "/api/v1/jobs/",
            params={"page": 1, "page_size": 10, "status": "failed"}
        )
        
        assert response.status_code == 200
        data = response.json()
        # All returned items should have status 'failed'
        for item in data["items"]:
            assert item["status"] == "failed"
    
    def test_list_jobs_pagination_works(
        self, client: TestClient, training_job: TrainingJob, training_project: Project
    ):
        """
        Test that pagination works correctly with mixed TrainingJob and Project records.
        
        Verifies:
        - Pagination returns correct page
        - Total count includes both types
        """
        response = client.get("/api/v1/jobs/", params={"page": 1, "page_size": 1})
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) <= 1
        assert data["total"] >= 2  # Should include both job and project
        assert data["page"] == 1
        assert data["page_size"] == 1
    
    def test_list_jobs_project_conversion_has_all_fields(
        self, client: TestClient, training_project: Project, training_config: TrainingConfig
    ):
        """
        Test that Project to TrainingJobResponse conversion includes all required fields.
        
        Verifies:
        - All TrainingJobResponse fields are present
        - Default values from config are used
        """
        response = client.get("/api/v1/jobs/", params={"page": 1, "page_size": 10})
        
        assert response.status_code == 200
        data = response.json()
        
        project_item = next((j for j in data["items"] if j["name"] == training_project.name), None)
        assert project_item is not None
        
        # Verify all required fields are present
        required_fields = [
            "id", "name", "description", "status", "training_type", "model_name",
            "dataset_id", "batch_size", "learning_rate", "epochs",
            "lora_r", "lora_alpha", "lora_dropout",
            "progress", "current_epoch", "current_loss", "error_message",
            "log", "model_path", "worker_id",
            "started_at", "completed_at", "created_at", "updated_at",
        ]
        
        for field in required_fields:
            assert field in project_item, f"Field {field} should be present in converted project"
        
        # Verify default values from config are used
        assert project_item["batch_size"] == training_config.default_batch_size
        assert project_item["learning_rate"] == training_config.default_learning_rate
        assert project_item["epochs"] == training_config.default_epochs  # Should be from config, not project

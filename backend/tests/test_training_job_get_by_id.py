"""
Tests for getting training job by ID, including Project IDs.

Following TDD methodology: Tests ensure that get_training_job endpoint
correctly handles both TrainingJob IDs and Project IDs.
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


class TestGetTrainingJobById:
    """Tests for getting training job by ID."""
    
    def test_get_training_job_by_id_returns_job(
        self, client: TestClient, training_job: TrainingJob
    ):
        """
        Test that get_training_job returns a TrainingJob by ID.
        
        Verifies:
        - TrainingJob can be fetched by ID
        - Response format is correct
        """
        response = client.get(f"/api/v1/jobs/{training_job.id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == training_job.id
        assert data["name"] == training_job.name
        assert data["status"] == training_job.status
    
    def test_get_training_job_by_id_returns_404_for_nonexistent_job(
        self, client: TestClient
    ):
        """
        Test that get_training_job returns 404 for nonexistent TrainingJob ID.
        
        Verifies:
        - 404 is returned for invalid TrainingJob ID
        """
        response = client.get("/api/v1/jobs/99999")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    def test_get_training_job_by_project_id_returns_project(
        self, client: TestClient, training_project: Project, training_config: TrainingConfig
    ):
        """
        Test that get_training_job returns a Project converted to TrainingJobResponse format.
        
        Verifies:
        - Project can be fetched by ID using jobs endpoint
        - Project is converted to TrainingJobResponse format
        - All required fields are present
        """
        response = client.get(f"/api/v1/jobs/{training_project.id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == training_project.id
        assert data["name"] == training_project.name
        assert data["status"] == training_project.status
        assert data["model_name"] == training_project.base_model
        assert data["training_type"] == training_project.training_type
        assert data["progress"] == training_project.progress
        assert data["current_epoch"] == training_project.current_epoch
        
        # Verify default values from config are used
        assert data["batch_size"] == training_config.default_batch_size
        assert data["learning_rate"] == training_config.default_learning_rate
        assert data["epochs"] == training_config.default_epochs
    
    def test_get_training_job_handles_both_job_and_project_ids(
        self, client: TestClient, test_db_session: Session, sample_dataset: Dataset, training_config: TrainingConfig
    ):
        """
        Test that get_training_job can fetch both TrainingJob and Project by ID.
        
        Verifies:
        - Both TrainingJob and Project can be fetched
        - Correct data is returned for each
        - TrainingJob takes precedence if both exist with same ID
        """
        # Create TrainingJob
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
        
        # Create Project (may have same ID since different table)
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
        
        # Fetch TrainingJob - should return the job
        job_response = client.get(f"/api/v1/jobs/{job.id}")
        assert job_response.status_code == 200
        job_data = job_response.json()
        assert job_data["id"] == job.id
        assert job_data["name"] == job.name
        
        # If project has different ID, fetch it
        # If same ID, TrainingJob takes precedence (which is correct behavior)
        if project.id != job.id:
            project_response = client.get(f"/api/v1/jobs/{project.id}")
            assert project_response.status_code == 200
            project_data = project_response.json()
            assert project_data["id"] == project.id
            assert project_data["name"] == project.name
        else:
            # If same ID, verify TrainingJob is returned (takes precedence)
            project_response = client.get(f"/api/v1/jobs/{project.id}")
            assert project_response.status_code == 200
            project_data = project_response.json()
            # Should return TrainingJob since it's checked first
            assert project_data["id"] == job.id
            assert project_data["name"] == job.name

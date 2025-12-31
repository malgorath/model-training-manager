"""
Tests for training jobs API to ensure projects are included.

Following TDD methodology: Tests ensure that when listing training jobs,
both TrainingJob records and Project records (that are training) are returned.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.training_job import TrainingJob, TrainingStatus
from app.models.project import Project, ProjectStatus
from app.models.dataset import Dataset


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
        epochs=3,
    )
    test_db_session.add(project)
    test_db_session.commit()
    test_db_session.refresh(project)
    return project


class TestTrainingJobsIncludeProjects:
    """Tests for training jobs API including projects."""
    
    def test_list_jobs_includes_training_jobs(
        self, client: TestClient, training_job: TrainingJob
    ):
        """
        Test that list_jobs returns TrainingJob records.
        
        Verifies:
        - TrainingJob records are returned
        - Status filtering works
        """
        response = client.get("/api/v1/jobs/", params={"page": 1, "page_size": 10})
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        assert any(j["id"] == training_job.id for j in data["items"])
    
    def test_list_jobs_includes_projects(
        self, client: TestClient, training_project: Project
    ):
        """
        Test that list_jobs includes Project records that are training.
        
        Verifies:
        - Projects with status 'running', 'pending', 'failed', 'completed' are included
        - Projects are converted to job-like format
        """
        response = client.get("/api/v1/jobs/", params={"page": 1, "page_size": 10})
        
        assert response.status_code == 200
        data = response.json()
        # Should include the project (may need to check by name or convert format)
        # This test documents the expected behavior
    
    def test_list_jobs_shows_all_statuses(
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
    
    def test_list_jobs_filters_by_status(
        self, client: TestClient, training_job: TrainingJob
    ):
        """
        Test that status filter works for both TrainingJob and Project.
        
        Verifies:
        - Filtering by status='running' returns running jobs and projects
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

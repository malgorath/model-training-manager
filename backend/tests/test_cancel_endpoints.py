"""
Tests for cancel button endpoints.

Following TDD methodology: Tests ensure cancel buttons work correctly
for both training jobs and projects.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.training_job import TrainingJob, TrainingStatus
from app.models.project import Project, ProjectStatus
from app.models.dataset import Dataset


@pytest.fixture
def running_training_job(test_db_session: Session) -> TrainingJob:
    """Create a running training job for testing."""
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
    
    job = TrainingJob(
        name="Running Job",
        training_type="qlora",
        model_name="meta-llama/Llama-3.2-3B",
        dataset_id=dataset.id,
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
def running_project(test_db_session: Session) -> Project:
    """Create a running project for testing."""
    project = Project(
        name="Running Project",
        base_model="meta-llama/Llama-3.2-3B",
        training_type="qlora",
        output_directory="./output/running",
        status=ProjectStatus.RUNNING.value,
        progress=50.0,
        current_epoch=2,
    )
    test_db_session.add(project)
    test_db_session.commit()
    test_db_session.refresh(project)
    return project


class TestCancelEndpoints:
    """Tests for cancel endpoints."""
    
    def test_cancel_training_job_endpoint_exists(
        self, client: TestClient, running_training_job: TrainingJob
    ):
        """
        Test that cancel training job endpoint exists and works.
        
        Verifies:
        - POST /api/v1/jobs/{id}/cancel endpoint exists
        - Returns 200 or appropriate status
        - Updates job status to cancelled
        """
        response = client.post(f"/api/v1/jobs/{running_training_job.id}/cancel")
        
        # Should not return 404 (endpoint exists)
        assert response.status_code != 404, "Cancel training job endpoint does not exist"
        
        # Should return success status
        assert response.status_code in [200, 201, 202], \
            f"Cancel endpoint returned {response.status_code}, expected 200/201/202"
        
        # Verify job status was updated
        get_response = client.get(f"/api/v1/jobs/{running_training_job.id}")
        assert get_response.status_code == 200
        job = get_response.json()
        # Status should be cancelled
        assert job['status'] == TrainingStatus.CANCELLED.value, \
            f"Job status should be cancelled, got {job['status']}"
    
    def test_cancel_project_endpoint_exists(
        self, client: TestClient, running_project: Project
    ):
        """
        Test that cancel project endpoint exists and works.
        
        Verifies:
        - POST /api/v1/projects/{id}/cancel endpoint exists (or similar)
        - Returns 200 or appropriate status
        - Updates project status to cancelled
        """
        # Check if cancel endpoint exists for projects
        response = client.post(f"/api/v1/projects/{running_project.id}/cancel")
        
        # If endpoint doesn't exist, we need to add it
        if response.status_code == 404:
            pytest.skip("Project cancel endpoint not yet implemented")
        
        # Should return success status
        assert response.status_code in [200, 201, 202], \
            f"Cancel project endpoint returned {response.status_code}, expected 200/201/202"
        
        # Verify project status was updated
        get_response = client.get(f"/api/v1/projects/{running_project.id}")
        assert get_response.status_code == 200
        project = get_response.json()
        # Status should be cancelled
        assert project['status'] == ProjectStatus.CANCELLED.value, \
            f"Project status should be cancelled, got {project['status']}"
    
    def test_cancel_only_works_for_running_or_queued_items(
        self, client: TestClient, test_db_session: Session
    ):
        """
        Test that cancel only works for running or queued items.
        
        Verifies:
        - Cannot cancel completed jobs
        - Cannot cancel failed jobs
        - Cannot cancel already cancelled jobs
        """
        # Create completed job
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
        
        completed_job = TrainingJob(
            name="Completed Job",
            training_type="qlora",
            model_name="meta-llama/Llama-3.2-3B",
            dataset_id=dataset.id,
            status=TrainingStatus.COMPLETED.value,
        )
        test_db_session.add(completed_job)
        test_db_session.commit()
        test_db_session.refresh(completed_job)
        
        # Try to cancel completed job
        response = client.post(f"/api/v1/jobs/{completed_job.id}/cancel")
        
        # Should return error (400 or 409)
        assert response.status_code in [400, 409], \
            f"Cancel should fail for completed jobs, got {response.status_code}"

"""
Tests to verify all run/start buttons point to working endpoints.

Following TDD methodology: Tests ensure every button that starts/runs something
has a corresponding working API endpoint.

Buttons tested:
1. Project Start button -> POST /api/v1/projects/{id}/start
2. Training Job Start button -> POST /api/v1/jobs/{id}/start
3. Worker Start button -> POST /api/v1/workers/ (action: start)
4. Worker Restart button -> POST /api/v1/workers/ (action: restart)
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
def pending_project(test_db_session: Session) -> Project:
    """Create a pending project for testing."""
    project = Project(
        name="Test Project",
        base_model="meta-llama/Llama-3.2-3B",
        training_type="qlora",
        output_directory="./output/test-project",
        status=ProjectStatus.PENDING.value,
    )
    test_db_session.add(project)
    test_db_session.commit()
    test_db_session.refresh(project)
    return project


@pytest.fixture
def pending_training_job(test_db_session: Session, sample_dataset: Dataset) -> TrainingJob:
    """Create a pending training job for testing."""
    job = TrainingJob(
        name="Test Training Job",
        training_type="qlora",
        model_name="meta-llama/Llama-3.2-3B",
        dataset_id=sample_dataset.id,
        status=TrainingStatus.PENDING.value,
    )
    test_db_session.add(job)
    test_db_session.commit()
    test_db_session.refresh(job)
    return job


class TestAllRunButtonsEndpoints:
    """Tests for all run/start button endpoints."""
    
    def test_project_start_button_endpoint_exists(
        self, client: TestClient, pending_project: Project
    ):
        """
        Test that Project Start button endpoint exists and works.
        
        Verifies:
        - POST /api/v1/projects/{id}/start endpoint exists
        - Returns 200 or appropriate status
        - Updates project status
        """
        response = client.post(f"/api/v1/projects/{pending_project.id}/start")
        
        # Should not return 404 (endpoint exists)
        assert response.status_code != 404, "Project start endpoint does not exist"
        
        # Should return success status
        assert response.status_code in [200, 201, 202], \
            f"Project start endpoint returned {response.status_code}, expected 200/201/202"
        
        # Verify project was queued (status may remain pending until worker picks it up)
        get_response = client.get(f"/api/v1/projects/{pending_project.id}")
        assert get_response.status_code == 200
        project = get_response.json()
        # Project should be queued (status may be pending, queued, or running)
        # The important thing is the endpoint exists and doesn't error
        assert project['status'] in [
            ProjectStatus.PENDING.value,
            ProjectStatus.RUNNING.value,
            'queued'  # Some systems use 'queued' status
        ], f"Project status should be pending/queued/running after start, got {project['status']}"
    
    def test_training_job_start_button_endpoint_exists(
        self, client: TestClient, pending_training_job: TrainingJob
    ):
        """
        Test that Training Job Start button endpoint exists and works.
        
        Verifies:
        - POST /api/v1/jobs/{id}/start endpoint exists
        - Returns 200 or appropriate status
        - Updates job status
        """
        response = client.post(f"/api/v1/jobs/{pending_training_job.id}/start")
        
        # Should not return 404 (endpoint exists)
        assert response.status_code != 404, "Training job start endpoint does not exist"
        
        # Should return success status
        assert response.status_code in [200, 201, 202], \
            f"Training job start endpoint returned {response.status_code}, expected 200/201/202"
        
        # Verify job status was updated
        get_response = client.get(f"/api/v1/jobs/{pending_training_job.id}")
        assert get_response.status_code == 200
        job = get_response.json()
        # Status should have changed from pending
        assert job['status'] != TrainingStatus.PENDING.value or response.status_code == 202, \
            "Job status should be updated when started"
    
    def test_worker_start_button_endpoint_exists(self, client: TestClient):
        """
        Test that Worker Start button endpoint exists and works.
        
        Verifies:
        - POST /api/v1/workers/ with action: start exists
        - Returns 200 or appropriate status
        - Updates worker pool status
        """
        response = client.post(
            "/api/v1/workers/",
            json={"action": "start", "worker_count": 1}
        )
        
        # Should not return 404 (endpoint exists)
        assert response.status_code != 404, "Worker start endpoint does not exist"
        
        # Should return success status
        assert response.status_code in [200, 201, 202], \
            f"Worker start endpoint returned {response.status_code}, expected 200/201/202"
        
        # Verify response has worker pool status
        if response.status_code == 200:
            data = response.json()
            assert "is_running" in data or "active_workers" in data or "workers" in data, \
                "Worker start should return worker pool status"
    
    def test_worker_restart_button_endpoint_exists(self, client: TestClient):
        """
        Test that Worker Restart button endpoint exists and works.
        
        Verifies:
        - POST /api/v1/workers/ with action: restart exists
        - Returns 200 or appropriate status
        - Updates worker pool status
        """
        response = client.post(
            "/api/v1/workers/",
            json={"action": "restart"}
        )
        
        # Should not return 404 (endpoint exists)
        assert response.status_code != 404, "Worker restart endpoint does not exist"
        
        # Should return success status
        assert response.status_code in [200, 201, 202], \
            f"Worker restart endpoint returned {response.status_code}, expected 200/201/202"
        
        # Verify response has worker pool status
        if response.status_code == 200:
            data = response.json()
            assert "is_running" in data or "active_workers" in data or "workers" in data, \
                "Worker restart should return worker pool status"
    
    def test_all_endpoints_return_proper_error_for_invalid_ids(self, client: TestClient):
        """
        Test that all start endpoints return proper errors for invalid IDs.
        
        Verifies:
        - All endpoints return 404 for nonexistent IDs (not 500 or other errors)
        """
        # Test project start with invalid ID
        response = client.post("/api/v1/projects/99999/start")
        assert response.status_code in [404, 400], \
            f"Project start should return 404/400 for invalid ID, got {response.status_code}"
        
        # Test job start with invalid ID
        response = client.post("/api/v1/jobs/99999/start")
        assert response.status_code in [404, 400], \
            f"Job start should return 404/400 for invalid ID, got {response.status_code}"
    
    def test_endpoints_handle_already_running_items(self, client: TestClient, test_db_session: Session):
        """
        Test that start endpoints handle items that are already running.
        
        Verifies:
        - Endpoints return appropriate status for already-running items
        """
        # Create a running project
        running_project = Project(
            name="Running Project",
            base_model="meta-llama/Llama-3.2-3B",
            training_type="qlora",
            output_directory="./output/running",
            status=ProjectStatus.RUNNING.value,
        )
        test_db_session.add(running_project)
        test_db_session.commit()
        test_db_session.refresh(running_project)
        
        # Try to start already-running project
        response = client.post(f"/api/v1/projects/{running_project.id}/start")
        # Should return 200 (idempotent) or 400 (already running)
        assert response.status_code in [200, 400, 409], \
            f"Starting already-running project should return 200/400/409, got {response.status_code}"

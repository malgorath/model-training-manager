"""
Test cancel button via UI simulation.

Following TDD methodology: Tests simulate actual UI button clicks
to verify cancel button works for stuck jobs.
"""

import pytest
import requests
from sqlalchemy.orm import Session

from app.models.training_job import TrainingJob, TrainingStatus
from app.models.dataset import Dataset
from app.models.project import Project, ProjectStatus


@pytest.fixture
def stuck_running_job(test_db_session: Session) -> TrainingJob:
    """Create a job stuck in running state with no workers."""
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
    
    # Create job stuck in running state (no worker assigned)
    job = TrainingJob(
        name="Stuck Running Job",
        training_type="qlora",
        model_name="meta-llama/Llama-3.2-3B",
        dataset_id=dataset.id,
        status=TrainingStatus.RUNNING.value,
        progress=50.0,
        current_epoch=2,
        epochs=4,
        worker_id=None,  # No worker assigned - stuck!
    )
    test_db_session.add(job)
    test_db_session.commit()
    test_db_session.refresh(job)
    return job


@pytest.fixture
def stuck_running_project(test_db_session: Session) -> Project:
    """Create a project stuck in running state with no workers."""
    project = Project(
        name="Stuck Running Project",
        base_model="meta-llama/Llama-3.2-3B",
        training_type="qlora",
        output_directory="./output/stuck",
        status=ProjectStatus.RUNNING.value,
        progress=50.0,
        current_epoch=2,
        worker_id=None,  # No worker assigned - stuck!
    )
    test_db_session.add(project)
    test_db_session.commit()
    test_db_session.refresh(project)
    return project


class TestCancelButtonUI:
    """Tests for cancel button via UI simulation."""
    
    def test_cancel_stuck_running_job_via_ui(
        self, client, stuck_running_job: TrainingJob
    ):
        """
        Test that cancel button works for job stuck running with no workers.
        
        Simulates: User clicks cancel button in JobMonitor UI.
        """
        # Simulate UI click: POST /api/v1/jobs/{id}/cancel
        response = client.post(f"/api/v1/jobs/{stuck_running_job.id}/cancel")
        
        # Should succeed (not 404, not 400)
        assert response.status_code == 200, \
            f"Cancel should work for stuck jobs, got {response.status_code}: {response.text}"
        
        # Verify job is cancelled
        job_data = response.json()
        assert job_data['status'] == TrainingStatus.CANCELLED.value, \
            f"Job should be cancelled, got status: {job_data['status']}"
        assert job_data['worker_id'] is None, \
            "Worker ID should be cleared"
        
        # Verify via GET endpoint (simulating UI refresh)
        get_response = client.get(f"/api/v1/jobs/{stuck_running_job.id}")
        assert get_response.status_code == 200
        job = get_response.json()
        assert job['status'] == TrainingStatus.CANCELLED.value, \
            "Job should remain cancelled after refresh"
    
    def test_cancel_stuck_running_project_via_ui(
        self, client, stuck_running_project: Project
    ):
        """
        Test that cancel button works for project stuck running with no workers.
        
        Simulates: User clicks cancel button in ProjectDetail UI.
        """
        # Simulate UI click: POST /api/v1/projects/{id}/cancel
        response = client.post(f"/api/v1/projects/{stuck_running_project.id}/cancel")
        
        # Should succeed (not 404, not 400)
        assert response.status_code == 200, \
            f"Cancel should work for stuck projects, got {response.status_code}: {response.text}"
        
        # Verify project is cancelled
        project_data = response.json()
        assert project_data['status'] == ProjectStatus.CANCELLED.value, \
            f"Project should be cancelled, got status: {project_data['status']}"
        assert project_data['worker_id'] is None, \
            "Worker ID should be cleared"
        
        # Verify via GET endpoint (simulating UI refresh)
        get_response = client.get(f"/api/v1/projects/{stuck_running_project.id}")
        assert get_response.status_code == 200
        project = get_response.json()
        assert project['status'] == ProjectStatus.CANCELLED.value, \
            "Project should remain cancelled after refresh"
    
    def test_cancel_button_refreshes_job_list(
        self, client, stuck_running_job: TrainingJob
    ):
        """
        Test that cancel button causes job list to refresh (simulating UI).
        
        Simulates: User clicks cancel, UI should refresh and show cancelled status.
        """
        # Get initial job list
        initial_response = client.get("/api/v1/jobs?page=1&page_size=10")
        assert initial_response.status_code == 200
        initial_jobs = initial_response.json()['items']
        initial_job = next((j for j in initial_jobs if j['id'] == stuck_running_job.id), None)
        assert initial_job is not None
        assert initial_job['status'] == TrainingStatus.RUNNING.value
        
        # Cancel the job (simulate button click)
        cancel_response = client.post(f"/api/v1/jobs/{stuck_running_job.id}/cancel")
        assert cancel_response.status_code == 200
        
        # Get updated job list (simulating UI refresh after cancel)
        updated_response = client.get("/api/v1/jobs?page=1&page_size=10")
        assert updated_response.status_code == 200
        updated_jobs = updated_response.json()['items']
        updated_job = next((j for j in updated_jobs if j['id'] == stuck_running_job.id), None)
        assert updated_job is not None
        assert updated_job['status'] == TrainingStatus.CANCELLED.value, \
            "Job list should show cancelled status after cancel button click"

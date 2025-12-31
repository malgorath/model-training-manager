"""
Test that cancel button works for Projects that appear in jobs list.

RED PHASE: This test will fail until cancel_job() handles Projects.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.project import Project, ProjectStatus
from app.models.dataset import Dataset


@pytest.fixture
def running_project_in_jobs_list(test_db_session: Session) -> Project:
    """Create a Project that appears in jobs list (simulating test-unsloth-training)."""
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
    
    # Create Project that will appear in jobs list
    project = Project(
        name="test-unsloth-training",
        base_model="meta-llama/Llama-3.2-3B",
        training_type="unsloth",
        output_directory="./output/test-unsloth-training",
        status=ProjectStatus.RUNNING.value,
        progress=50.0,
        current_epoch=2,
        worker_id=None,  # Stuck with no worker
    )
    test_db_session.add(project)
    test_db_session.commit()
    test_db_session.refresh(project)
    return project


class TestCancelButtonHandlesProjects:
    """Test cancel button works for Projects in jobs list."""
    
    def test_cancel_project_via_jobs_endpoint_returns_200(
        self, client: TestClient, running_project_in_jobs_list: Project
    ):
        """
        RED PHASE: Test that cancel endpoint returns 200 for Project.
        
        This will fail because cancel_job() only checks TrainingJob table.
        """
        # Simulate clicking cancel button in UI
        response = client.post(f"/api/v1/jobs/{running_project_in_jobs_list.id}/cancel")
        
        # Should return 200, not 400 or 404
        assert response.status_code == 200, \
            f"Cancel should return 200 for Project, got {response.status_code}: {response.text}"
        
        # Verify project is cancelled
        project_data = response.json()
        assert project_data['status'] == 'cancelled', \
            f"Project should be cancelled, got status: {project_data['status']}"

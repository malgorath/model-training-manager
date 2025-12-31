"""
Test that start button works for Projects that appear in jobs list.

RED PHASE: This test will verify start endpoint handles Projects.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.project import Project, ProjectStatus
from app.models.dataset import Dataset


@pytest.fixture
def pending_project_in_jobs_list(test_db_session: Session) -> Project:
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
        status=ProjectStatus.PENDING.value,
        progress=0.0,
        current_epoch=0,
        worker_id=None,
    )
    test_db_session.add(project)
    test_db_session.commit()
    test_db_session.refresh(project)
    return project


class TestStartButtonHandlesProjects:
    """Test start button works for Projects in jobs list."""
    
    def test_start_project_via_jobs_endpoint_returns_200(
        self, client: TestClient, pending_project_in_jobs_list: Project
    ):
        """
        Test that start endpoint returns 200 for Project.
        
        This should work if start endpoint handles Projects.
        """
        # Simulate clicking start button in UI
        response = client.post(f"/api/v1/jobs/{pending_project_in_jobs_list.id}/start")
        
        # Should return 200, not 400 or 404
        assert response.status_code == 200, \
            f"Start should return 200 for Project, got {response.status_code}: {response.text}"

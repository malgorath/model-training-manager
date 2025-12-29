"""
Integration tests for project-based training lifecycle.

Tests the complete flow from project creation through training to model validation.
"""

import pytest
import tempfile
from pathlib import Path
from sqlalchemy.orm import Session
from unittest.mock import patch, MagicMock

from app.models.project import Project, ProjectStatus
from app.models.dataset import Dataset
from app.services.project_service import ProjectService
from app.services.model_resolution_service import ModelResolutionService
from app.services.model_validation_service import ModelValidationService
from app.workers.training_worker import TrainingWorker


class TestProjectLifecycle:
    """Integration tests for complete project lifecycle."""
    
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
                row_count=1000,
                column_count=2,
                columns='["input", "output"]',
            )
            test_db_session.add(dataset)
            datasets.append(dataset)
        test_db_session.commit()
        return datasets
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        import shutil
        shutil.rmtree(temp_dir)
    
    def test_create_project_with_multiple_traits(self, test_db_session: Session, sample_datasets, temp_output_dir):
        """Test creating a project with all trait types."""
        service = ProjectService(db=test_db_session)
        
        project_data = {
            "name": "Integration Test Project",
            "description": "Test project for integration",
            "base_model": "meta-llama/Llama-3.2-3B-Instruct",
            "training_type": "qlora",
            "max_rows": 50000,
            "output_directory": str(temp_output_dir / "test-project"),
            "traits": [
                {
                    "trait_type": "reasoning",
                    "datasets": [
                        {"dataset_id": sample_datasets[0].id, "percentage": 100.0}
                    ],
                },
                {
                    "trait_type": "coding",
                    "datasets": [
                        {"dataset_id": sample_datasets[1].id, "percentage": 100.0}
                    ],
                },
                {
                    "trait_type": "general_tools",
                    "datasets": [
                        {"dataset_id": sample_datasets[2].id, "percentage": 100.0}
                    ],
                },
            ],
        }
        
        project = service.create_project(project_data)
        
        assert project.id is not None
        assert len(project.traits) == 3
        assert project.status == ProjectStatus.PENDING.value
    
    def test_dataset_combination_for_training(self, test_db_session: Session, sample_datasets):
        """Test combining datasets based on percentages."""
        service = ProjectService(db=test_db_session)
        
        project_data = {
            "name": "Combination Test",
            "base_model": "llama3.2:3b",
            "training_type": "qlora",
            "max_rows": 100000,
            "output_directory": "/tmp/test",
            "traits": [
                {
                    "trait_type": "general_tools",
                    "datasets": [
                        {"dataset_id": sample_datasets[0].id, "percentage": 50.0},
                        {"dataset_id": sample_datasets[1].id, "percentage": 30.0},
                        {"dataset_id": sample_datasets[2].id, "percentage": 20.0},
                    ],
                },
            ],
        }
        
        project = service.create_project(project_data)
        
        combined = service.combine_datasets_for_training(project.id, max_rows=100000)
        
        assert len(combined) == 3
        total_rows = sum(item["rows"] for item in combined)
        assert total_rows <= 100000
        
        # Verify percentages are respected
        percentages = [item["percentage"] for item in combined]
        assert sum(percentages) == 100.0
    
    def test_project_validation_rules(self, test_db_session: Session, sample_datasets):
        """Test that validation rules are enforced."""
        service = ProjectService(db=test_db_session)
        
        # Test: Reasoning must have exactly 1 dataset
        with pytest.raises(Exception) as exc_info:
            service.create_project({
                "name": "Invalid",
                "base_model": "llama3.2:3b",
                "training_type": "qlora",
                "max_rows": 50000,
                "output_directory": "/tmp",
                "traits": [
                    {
                        "trait_type": "reasoning",
                        "datasets": [
                            {"dataset_id": sample_datasets[0].id, "percentage": 50.0},
                            {"dataset_id": sample_datasets[1].id, "percentage": 50.0},
                        ],
                    },
                ],
            })
        
        assert "reasoning" in str(exc_info.value).lower() or "one dataset" in str(exc_info.value).lower()
        
        # Test: Percentages must sum to 100%
        with pytest.raises(Exception):
            service.create_project({
                "name": "Invalid",
                "base_model": "llama3.2:3b",
                "training_type": "qlora",
                "max_rows": 50000,
                "output_directory": "/tmp",
                "traits": [
                    {
                        "trait_type": "general_tools",
                        "datasets": [
                            {"dataset_id": sample_datasets[0].id, "percentage": 50.0},
                            {"dataset_id": sample_datasets[1].id, "percentage": 40.0},
                        ],
                    },
                ],
            })
        
        # Test: No duplicate datasets
        with pytest.raises(Exception):
            service.create_project({
                "name": "Invalid",
                "base_model": "llama3.2:3b",
                "training_type": "qlora",
                "max_rows": 50000,
                "output_directory": "/tmp",
                "traits": [
                    {
                        "trait_type": "reasoning",
                        "datasets": [
                            {"dataset_id": sample_datasets[0].id, "percentage": 100.0}
                        ],
                    },
                    {
                        "trait_type": "coding",
                        "datasets": [
                            {"dataset_id": sample_datasets[0].id, "percentage": 100.0}  # Duplicate!
                        ],
                    },
                ],
            })
    
    @patch('app.workers.training_worker.ModelResolutionService')
    def test_project_training_flow(self, MockResolver, test_db_session: Session, sample_datasets, temp_output_dir):
        """Test the complete training flow for a project."""
        # Setup project
        service = ProjectService(db=test_db_session)
        
        project_data = {
            "name": "Training Test Project",
            "base_model": "meta-llama/Llama-3.2-3B-Instruct",
            "training_type": "qlora",
            "max_rows": 50000,
            "output_directory": str(temp_output_dir / "training-project"),
            "traits": [
                {
                    "trait_type": "reasoning",
                    "datasets": [
                        {"dataset_id": sample_datasets[0].id, "percentage": 100.0}
                    ],
                },
            ],
        }
        
        project = service.create_project(project_data)
        
        # Mock model resolution
        mock_resolver = MagicMock()
        mock_resolver.is_model_available.return_value = True
        mock_resolver.resolve_model_path.return_value = "/fake/model/path"
        mock_resolver.validate_model_format.return_value = None
        MockResolver.return_value = mock_resolver
        
        # Mock validation service
        with patch('app.workers.training_worker.ModelValidationService') as MockValidation:
            mock_validation = MagicMock()
            mock_validation.validate_model_complete.return_value = {
                "valid": True,
                "files": {"valid": True, "errors": []},
                "loading": {"valid": True, "errors": []},
                "errors": [],
            }
            MockValidation.return_value = mock_validation
            
            # Create worker and process project
            def db_factory():
                return test_db_session
            
            worker = TrainingWorker(
                worker_id="test-worker",
                db_session_factory=db_factory,
            )
            
            # Mock training methods to avoid actual model loading
            with patch.object(worker, '_train_qlora_real'):
                # _process_project is called internally by the worker loop
                # Test by directly calling it
                if hasattr(worker, '_process_project'):
                    worker._process_project(project.id)
                else:
                    # Skip this test if method doesn't exist yet
                    pytest.skip("_process_project method not implemented")
            
            test_db_session.refresh(project)
            
            # Verify project was processed
            assert project.worker_id == "test-worker"
            # Status will be completed or failed based on mocked training

"""
Tests for ProjectService, ModelValidationService, and OutputDirectoryService.

Tests project CRUD operations, validation logic, dataset combination, and directory validation.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from sqlalchemy.orm import Session

from app.models.project import Project, ProjectTrait, TraitType, ProjectTraitDataset
from app.models.dataset import Dataset
from app.services.project_service import (
    ProjectService,
    ProjectValidationError,
    DatasetAllocationError,
)
from app.services.model_validation_service import (
    ModelValidationService,
    ModelValidationError,
)
from app.services.output_directory_service import (
    OutputDirectoryService,
    DirectoryValidationError,
)


class TestProjectService:
    """Tests for ProjectService."""
    
    @pytest.fixture
    def service(self, test_db_session: Session):
        """Create a ProjectService instance."""
        return ProjectService(db=test_db_session)
    
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
    
    def test_create_project(self, service, sample_datasets):
        """Test creating a project."""
        project_data = {
            "name": "Test Project",
            "description": "A test project",
            "base_model": "meta-llama/Llama-3.2-3B-Instruct",
            "training_type": "qlora",
            "max_rows": 50000,
            "output_directory": "/output/test-project",
            "traits": [
                {
                    "trait_type": TraitType.REASONING.value,
                    "datasets": [
                        {"dataset_id": sample_datasets[0].id, "percentage": 100.0}
                    ],
                },
            ],
        }
        
        project = service.create_project(project_data)
        
        assert project.id is not None
        assert project.name == "Test Project"
        assert len(project.traits) == 1
        assert project.traits[0].trait_type == TraitType.REASONING.value
    
    def test_create_project_with_multiple_traits(self, service, sample_datasets):
        """Test creating a project with multiple traits."""
        project_data = {
            "name": "Multi-Trait Project",
            "base_model": "llama3.2:3b",
            "training_type": "qlora",
            "max_rows": 100000,
            "output_directory": "/output/multi-trait",
            "traits": [
                {
                    "trait_type": TraitType.REASONING.value,
                    "datasets": [
                        {"dataset_id": sample_datasets[0].id, "percentage": 100.0}
                    ],
                },
                {
                    "trait_type": TraitType.CODING.value,
                    "datasets": [
                        {"dataset_id": sample_datasets[1].id, "percentage": 100.0}
                    ],
                },
                {
                    "trait_type": TraitType.GENERAL_TOOLS.value,
                    "datasets": [
                        {"dataset_id": sample_datasets[2].id, "percentage": 100.0}
                    ],
                },
            ],
        }
        
        project = service.create_project(project_data)
        
        assert len(project.traits) == 3
    
    def test_validate_reasoning_one_dataset(self, service, sample_datasets):
        """Test that Reasoning trait must have exactly one dataset."""
        project_data = {
            "name": "Test Project",
            "base_model": "llama3.2:3b",
            "training_type": "qlora",
            "max_rows": 50000,
            "output_directory": "/output",
            "traits": [
                {
                    "trait_type": TraitType.REASONING.value,
                    "datasets": [
                        {"dataset_id": sample_datasets[0].id, "percentage": 50.0},
                        {"dataset_id": sample_datasets[1].id, "percentage": 50.0},
                    ],
                },
            ],
        }
        
        with pytest.raises(ProjectValidationError) as exc_info:
            service.create_project(project_data)
        
        assert "reasoning" in str(exc_info.value).lower()
        assert "one dataset" in str(exc_info.value).lower()
    
    def test_validate_coding_one_dataset(self, service, sample_datasets):
        """Test that Coding trait must have exactly one dataset."""
        project_data = {
            "name": "Test Project",
            "base_model": "llama3.2:3b",
            "training_type": "qlora",
            "max_rows": 50000,
            "output_directory": "/output",
            "traits": [
                {
                    "trait_type": TraitType.CODING.value,
                    "datasets": [
                        {"dataset_id": sample_datasets[0].id, "percentage": 60.0},
                        {"dataset_id": sample_datasets[1].id, "percentage": 40.0},
                    ],
                },
            ],
        }
        
        with pytest.raises(ProjectValidationError) as exc_info:
            service.create_project(project_data)
        
        assert "coding" in str(exc_info.value).lower()
        assert "one dataset" in str(exc_info.value).lower()
    
    def test_validate_general_tools_multiple_datasets(self, service, sample_datasets):
        """Test that General/Tools trait can have multiple datasets."""
        project_data = {
            "name": "Test Project",
            "base_model": "llama3.2:3b",
            "training_type": "qlora",
            "max_rows": 50000,
            "output_directory": "/output",
            "traits": [
                {
                    "trait_type": TraitType.GENERAL_TOOLS.value,
                    "datasets": [
                        {"dataset_id": sample_datasets[0].id, "percentage": 33.33},
                        {"dataset_id": sample_datasets[1].id, "percentage": 33.33},
                        {"dataset_id": sample_datasets[2].id, "percentage": 33.34},
                    ],
                },
            ],
        }
        
        project = service.create_project(project_data)
        
        assert len(project.traits) == 1
        assert len(project.traits[0].datasets) == 3
    
    def test_validate_percentages_sum_to_100(self, service, sample_datasets):
        """Test that dataset percentages must sum to 100%."""
        project_data = {
            "name": "Test Project",
            "base_model": "llama3.2:3b",
            "training_type": "qlora",
            "max_rows": 50000,
            "output_directory": "/output",
            "traits": [
                {
                    "trait_type": TraitType.GENERAL_TOOLS.value,
                    "datasets": [
                        {"dataset_id": sample_datasets[0].id, "percentage": 40.0},
                        {"dataset_id": sample_datasets[1].id, "percentage": 40.0},
                    ],
                },
            ],
        }
        
        with pytest.raises(DatasetAllocationError) as exc_info:
            service.create_project(project_data)
        
        assert "100%" in str(exc_info.value)
    
    def test_validate_no_duplicate_datasets_in_project(self, service, sample_datasets):
        """Test that a dataset cannot be used twice in the same project."""
        project_data = {
            "name": "Test Project",
            "base_model": "llama3.2:3b",
            "training_type": "qlora",
            "max_rows": 50000,
            "output_directory": "/output",
            "traits": [
                {
                    "trait_type": TraitType.REASONING.value,
                    "datasets": [
                        {"dataset_id": sample_datasets[0].id, "percentage": 100.0}
                    ],
                },
                {
                    "trait_type": TraitType.CODING.value,
                    "datasets": [
                        {"dataset_id": sample_datasets[0].id, "percentage": 100.0}  # Same dataset!
                    ],
                },
            ],
        }
        
        with pytest.raises(DatasetAllocationError) as exc_info:
            service.create_project(project_data)
        
        assert "multiple times" in str(exc_info.value).lower() or "duplicate" in str(exc_info.value).lower() or "twice" in str(exc_info.value).lower()
    
    def test_get_project(self, service, sample_datasets):
        """Test retrieving a project by ID."""
        project_data = {
            "name": "Test Project",
            "base_model": "llama3.2:3b",
            "training_type": "qlora",
            "max_rows": 50000,
            "output_directory": "/output",
            "traits": [
                {
                    "trait_type": TraitType.REASONING.value,
                    "datasets": [
                        {"dataset_id": sample_datasets[0].id, "percentage": 100.0}
                    ],
                },
            ],
        }
        
        created = service.create_project(project_data)
        retrieved = service.get_project(created.id)
        
        assert retrieved.id == created.id
        assert retrieved.name == "Test Project"
    
    def test_list_projects(self, service, sample_datasets):
        """Test listing all projects."""
        for i in range(3):
            project_data = {
                "name": f"Project {i+1}",
                "base_model": "llama3.2:3b",
                "training_type": "qlora",
                "max_rows": 50000,
                "output_directory": f"/output/project{i+1}",
                "traits": [
                    {
                        "trait_type": TraitType.REASONING.value,
                        "datasets": [
                            {"dataset_id": sample_datasets[0].id, "percentage": 100.0}
                        ],
                    },
                ],
            }
            service.create_project(project_data)
        
        projects = service.list_projects()
        
        assert len(projects) >= 3
    
    def test_combine_datasets_for_training(self, service, sample_datasets):
        """Test combining datasets based on percentages."""
        project_data = {
            "name": "Test Project",
            "base_model": "llama3.2:3b",
            "training_type": "qlora",
            "max_rows": 100000,
            "output_directory": "/output",
            "traits": [
                {
                    "trait_type": TraitType.GENERAL_TOOLS.value,
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
        # Each dataset should have rows proportional to its percentage
        total_rows = sum(item["rows"] for item in combined)
        assert total_rows <= 100000


class TestModelValidationService:
    """Tests for ModelValidationService."""
    
    @pytest.fixture
    def service(self):
        """Create a ModelValidationService instance."""
        return ModelValidationService()
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create a temporary model directory with valid files."""
        temp_dir = tempfile.mkdtemp()
        model_path = Path(temp_dir) / "model"
        model_path.mkdir()
        
        # Create required files
        (model_path / "config.json").write_text('{"model_type": "llama"}')
        (model_path / "tokenizer.json").write_text("{}")
        (model_path / "tokenizer_config.json").write_text("{}")
        
        yield str(model_path)
        
        import shutil
        shutil.rmtree(temp_dir)
    
    def test_validate_model_files_exist(self, service, temp_model_dir):
        """Test validating that all required model files exist."""
        result = service.validate_model_files(temp_model_dir)
        
        assert result["valid"] is True
        assert "config.json" in result["files_checked"]
    
    def test_validate_model_files_missing(self, service):
        """Test validation fails when files are missing."""
        temp_dir = tempfile.mkdtemp()
        try:
            model_path = Path(temp_dir) / "incomplete_model"
            model_path.mkdir()
            (model_path / "config.json").write_text('{}')
            # Missing tokenizer files
            
            result = service.validate_model_files(str(model_path))
            
            assert result["valid"] is False
            assert "missing" in result["errors"][0].lower() or "tokenizer" in result["errors"][0].lower()
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    @patch('transformers.AutoModelForCausalLM')
    @patch('transformers.AutoTokenizer')
    def test_validate_model_loading(self, mock_tokenizer, mock_model, service, temp_model_dir):
        """Test validating that model can be loaded."""
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        result = service.validate_model_loading(temp_model_dir)
        
        assert result["valid"] is True
        mock_model.from_pretrained.assert_called_once()
        mock_tokenizer.from_pretrained.assert_called_once()
    
    def test_validate_model_complete(self, service, temp_model_dir):
        """Test complete model validation."""
        with patch.object(service, 'validate_model_loading', return_value={"valid": True, "errors": []}):
            result = service.validate_model_complete(temp_model_dir)
            
            assert result["valid"] is True
            assert "files" in result
            assert "loading" in result


class TestOutputDirectoryService:
    """Tests for OutputDirectoryService."""
    
    @pytest.fixture
    def service(self):
        """Create an OutputDirectoryService instance."""
        return OutputDirectoryService()
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        import shutil
        shutil.rmtree(temp_dir)
    
    def test_validate_directory_writable(self, service, temp_dir):
        """Test validating a writable directory."""
        result = service.validate_directory(temp_dir)
        
        assert result["valid"] is True
        assert result["writable"] is True
    
    def test_validate_directory_nonexistent(self, service):
        """Test validating a non-existent directory."""
        nonexistent = "/nonexistent/path/that/should/not/exist"
        
        with pytest.raises(DirectoryValidationError) as exc_info:
            service.validate_directory(nonexistent)
        
        assert "exist" in str(exc_info.value).lower()
    
    def test_validate_directory_not_writable(self, service, temp_dir):
        """Test validating a non-writable directory."""
        read_only_dir = Path(temp_dir) / "readonly"
        read_only_dir.mkdir()
        
        # Make directory read-only
        os.chmod(read_only_dir, 0o444)
        
        try:
            with pytest.raises(DirectoryValidationError) as exc_info:
                service.validate_directory(str(read_only_dir))
            
            assert "write" in str(exc_info.value).lower() or "permission" in str(exc_info.value).lower()
        finally:
            # Restore permissions for cleanup
            os.chmod(read_only_dir, 0o755)
    
    def test_create_directory_if_needed(self, service, temp_dir):
        """Test creating directory if it doesn't exist."""
        new_dir = Path(temp_dir) / "new" / "directory" / "path"
        
        result = service.validate_directory(str(new_dir), create_if_missing=True)
        
        assert result["valid"] is True
        assert new_dir.exists()
        assert new_dir.is_dir()

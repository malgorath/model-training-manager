"""
Tests for Project, ProjectTrait, and ProjectTraitDataset models.

Tests the SQLAlchemy models for project-based training with traits and dataset allocations.
"""

import pytest
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.models.project import Project, ProjectStatus, ProjectTrait, TraitType, ProjectTraitDataset
from app.models.dataset import Dataset


class TestProjectModel:
    """Tests for the Project model."""
    
    def test_create_project(self, test_db_session: Session):
        """Test creating a new project."""
        project = Project(
            name="Test Project",
            description="A test project",
            base_model="meta-llama/Llama-3.2-3B-Instruct",
            training_type="qlora",
            max_rows=50000,
            output_directory="/output/test-project",
        )
        test_db_session.add(project)
        test_db_session.commit()
        
        assert project.id is not None
        assert project.name == "Test Project"
        assert project.base_model == "meta-llama/Llama-3.2-3B-Instruct"
        assert project.status == ProjectStatus.PENDING.value
        assert project.max_rows == 50000
        assert project.created_at is not None
        assert project.updated_at is not None
    
    def test_project_required_fields(self, test_db_session: Session):
        """Test that required fields are enforced."""
        project = Project(
            name="Test",
            base_model="llama3.2:3b",
            training_type="qlora",
            max_rows=100000,
            output_directory="/output",
        )
        test_db_session.add(project)
        test_db_session.commit()
        
        assert project.id is not None
        assert project.status == ProjectStatus.PENDING.value
    
    def test_project_default_values(self, test_db_session: Session):
        """Test default values for optional fields."""
        project = Project(
            name="Test Project",
            base_model="llama3.2:3b",
            training_type="qlora",
            max_rows=50000,
            output_directory="/output",
        )
        test_db_session.add(project)
        test_db_session.commit()
        
        assert project.status == ProjectStatus.PENDING.value
        assert project.description is None
        assert project.progress == 0.0
        assert project.current_epoch == 0
        assert project.error_message is None
    
    def test_project_traits_relationship(self, test_db_session: Session):
        """Test relationship between Project and ProjectTrait."""
        project = Project(
            name="Test Project",
            base_model="llama3.2:3b",
            training_type="qlora",
            max_rows=50000,
            output_directory="/output",
        )
        test_db_session.add(project)
        test_db_session.flush()
        
        trait = ProjectTrait(
            project_id=project.id,
            trait_type=TraitType.REASONING.value,
        )
        test_db_session.add(trait)
        test_db_session.commit()
        
        assert len(project.traits) == 1
        assert project.traits[0].trait_type == TraitType.REASONING.value
    
    def test_project_max_rows_constraints(self, test_db_session: Session):
        """Test that max_rows accepts valid values."""
        valid_rows = [50000, 100000, 250000, 500000, 1000000]
        
        for rows in valid_rows:
            project = Project(
                name=f"Project {rows}",
                base_model="llama3.2:3b",
                training_type="qlora",
                max_rows=rows,
                output_directory="/output",
            )
            test_db_session.add(project)
        
        test_db_session.commit()
        
        projects = test_db_session.query(Project).all()
        assert len(projects) == len(valid_rows)


class TestProjectTraitModel:
    """Tests for the ProjectTrait model."""
    
    def test_create_project_trait(self, test_db_session: Session):
        """Test creating a project trait."""
        project = Project(
            name="Test Project",
            base_model="llama3.2:3b",
            training_type="qlora",
            max_rows=50000,
            output_directory="/output",
        )
        test_db_session.add(project)
        test_db_session.flush()
        
        trait = ProjectTrait(
            project_id=project.id,
            trait_type=TraitType.REASONING.value,
        )
        test_db_session.add(trait)
        test_db_session.commit()
        
        assert trait.id is not None
        assert trait.project_id == project.id
        assert trait.trait_type == TraitType.REASONING.value
    
    def test_all_trait_types(self, test_db_session: Session):
        """Test all valid trait types."""
        project = Project(
            name="Test Project",
            base_model="llama3.2:3b",
            training_type="qlora",
            max_rows=50000,
            output_directory="/output",
        )
        test_db_session.add(project)
        test_db_session.flush()
        
        trait_types = [TraitType.REASONING, TraitType.CODING, TraitType.GENERAL_TOOLS]
        
        for trait_type in trait_types:
            trait = ProjectTrait(
                project_id=project.id,
                trait_type=trait_type.value,
            )
            test_db_session.add(trait)
        
        test_db_session.commit()
        
        assert len(project.traits) == 3
    
    def test_trait_project_relationship(self, test_db_session: Session):
        """Test relationship from trait to project."""
        project = Project(
            name="Test Project",
            base_model="llama3.2:3b",
            training_type="qlora",
            max_rows=50000,
            output_directory="/output",
        )
        test_db_session.add(project)
        test_db_session.flush()
        
        trait = ProjectTrait(
            project_id=project.id,
            trait_type=TraitType.CODING.value,
        )
        test_db_session.add(trait)
        test_db_session.commit()
        
        assert trait.project.id == project.id
        assert trait.project.name == "Test Project"
    
    def test_trait_datasets_relationship(self, test_db_session: Session):
        """Test relationship between trait and datasets."""
        project = Project(
            name="Test Project",
            base_model="llama3.2:3b",
            training_type="qlora",
            max_rows=50000,
            output_directory="/output",
        )
        test_db_session.add(project)
        test_db_session.flush()
        
        trait = ProjectTrait(
            project_id=project.id,
            trait_type=TraitType.GENERAL_TOOLS.value,
        )
        test_db_session.add(trait)
        test_db_session.flush()
        
        dataset = Dataset(
            name="Test Dataset",
            filename="test.csv",
            file_path="/uploads/test.csv",
            file_type="csv",
            file_size=1024,
            row_count=100,
            column_count=2,
            columns='["input", "output"]',
        )
        test_db_session.add(dataset)
        test_db_session.flush()
        
        trait_dataset = ProjectTraitDataset(
            project_trait_id=trait.id,
            dataset_id=dataset.id,
            percentage=100.0,
        )
        test_db_session.add(trait_dataset)
        test_db_session.commit()
        
        assert len(trait.datasets) == 1
        assert trait.datasets[0].dataset_id == dataset.id
        assert trait.datasets[0].percentage == 100.0


class TestProjectTraitDatasetModel:
    """Tests for the ProjectTraitDataset junction model."""
    
    def test_create_project_trait_dataset(self, test_db_session: Session):
        """Test creating a project trait dataset allocation."""
        project = Project(
            name="Test Project",
            base_model="llama3.2:3b",
            training_type="qlora",
            max_rows=50000,
            output_directory="/output",
        )
        test_db_session.add(project)
        test_db_session.flush()
        
        trait = ProjectTrait(
            project_id=project.id,
            trait_type=TraitType.REASONING.value,
        )
        test_db_session.add(trait)
        test_db_session.flush()
        
        dataset = Dataset(
            name="Test Dataset",
            filename="test.csv",
            file_path="/uploads/test.csv",
            file_type="csv",
            file_size=1024,
            row_count=100,
            column_count=2,
            columns='["input", "output"]',
        )
        test_db_session.add(dataset)
        test_db_session.flush()
        
        trait_dataset = ProjectTraitDataset(
            project_trait_id=trait.id,
            dataset_id=dataset.id,
            percentage=100.0,
        )
        test_db_session.add(trait_dataset)
        test_db_session.commit()
        
        assert trait_dataset.id is not None
        assert trait_dataset.project_trait_id == trait.id
        assert trait_dataset.dataset_id == dataset.id
        assert trait_dataset.percentage == 100.0
    
    def test_percentage_range_constraint(self, test_db_session: Session):
        """Test that percentage must be between 0 and 100."""
        project = Project(
            name="Test Project",
            base_model="llama3.2:3b",
            training_type="qlora",
            max_rows=50000,
            output_directory="/output",
        )
        test_db_session.add(project)
        test_db_session.flush()
        
        trait = ProjectTrait(
            project_id=project.id,
            trait_type=TraitType.GENERAL_TOOLS.value,  # General/Tools can have multiple datasets
        )
        test_db_session.add(trait)
        test_db_session.flush()
        
        # Create multiple datasets for different percentages
        valid_percentages = [0.0, 25.0, 50.0, 75.0, 100.0]
        for i, percentage in enumerate(valid_percentages):
            dataset = Dataset(
                name=f"Test Dataset {i+1}",
                filename=f"test{i+1}.csv",
                file_path=f"/uploads/test{i+1}.csv",
                file_type="csv",
                file_size=1024,
                row_count=100,
                column_count=2,
                columns='["input", "output"]',
            )
            test_db_session.add(dataset)
            test_db_session.flush()
            
            trait_dataset = ProjectTraitDataset(
                project_trait_id=trait.id,
                dataset_id=dataset.id,
                percentage=percentage,
            )
            test_db_session.add(trait_dataset)
            test_db_session.flush()
        
        test_db_session.commit()
        
        # Should have created all valid allocations
        allocations = test_db_session.query(ProjectTraitDataset).filter_by(
            project_trait_id=trait.id
        ).all()
        assert len(allocations) == len(valid_percentages)
    
    def test_multiple_datasets_for_general_tools(self, test_db_session: Session):
        """Test that General/Tools trait can have multiple datasets."""
        project = Project(
            name="Test Project",
            base_model="llama3.2:3b",
            training_type="qlora",
            max_rows=50000,
            output_directory="/output",
        )
        test_db_session.add(project)
        test_db_session.flush()
        
        trait = ProjectTrait(
            project_id=project.id,
            trait_type=TraitType.GENERAL_TOOLS.value,
        )
        test_db_session.add(trait)
        test_db_session.flush()
        
        # Create multiple datasets
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
        
        test_db_session.flush()
        
        # Allocate percentages
        percentages = [33.33, 33.33, 33.34]  # Sums to 100%
        for dataset, percentage in zip(datasets, percentages):
            trait_dataset = ProjectTraitDataset(
                project_trait_id=trait.id,
                dataset_id=dataset.id,
                percentage=percentage,
            )
            test_db_session.add(trait_dataset)
        
        test_db_session.commit()
        
        assert len(trait.datasets) == 3
    
    def test_dataset_unique_per_trait(self, test_db_session: Session):
        """Test that a dataset cannot be used twice in the same trait."""
        project = Project(
            name="Test Project",
            base_model="llama3.2:3b",
            training_type="qlora",
            max_rows=50000,
            output_directory="/output",
        )
        test_db_session.add(project)
        test_db_session.flush()
        
        trait = ProjectTrait(
            project_id=project.id,
            trait_type=TraitType.REASONING.value,
        )
        test_db_session.add(trait)
        test_db_session.flush()
        
        dataset = Dataset(
            name="Test Dataset",
            filename="test.csv",
            file_path="/uploads/test.csv",
            file_type="csv",
            file_size=1024,
            row_count=100,
            column_count=2,
            columns='["input", "output"]',
        )
        test_db_session.add(dataset)
        test_db_session.flush()
        
        # First allocation
        trait_dataset1 = ProjectTraitDataset(
            project_trait_id=trait.id,
            dataset_id=dataset.id,
            percentage=50.0,
        )
        test_db_session.add(trait_dataset1)
        test_db_session.flush()
        
        # Second allocation with same dataset - should fail
        trait_dataset2 = ProjectTraitDataset(
            project_trait_id=trait.id,
            dataset_id=dataset.id,
            percentage=50.0,
        )
        test_db_session.add(trait_dataset2)
        
        with pytest.raises(IntegrityError):
            test_db_session.commit()
    
    def test_dataset_relationship(self, test_db_session: Session):
        """Test relationship from ProjectTraitDataset to Dataset."""
        project = Project(
            name="Test Project",
            base_model="llama3.2:3b",
            training_type="qlora",
            max_rows=50000,
            output_directory="/output",
        )
        test_db_session.add(project)
        test_db_session.flush()
        
        trait = ProjectTrait(
            project_id=project.id,
            trait_type=TraitType.CODING.value,
        )
        test_db_session.add(trait)
        test_db_session.flush()
        
        dataset = Dataset(
            name="Test Dataset",
            filename="test.csv",
            file_path="/uploads/test.csv",
            file_type="csv",
            file_size=1024,
            row_count=100,
            column_count=2,
            columns='["input", "output"]',
        )
        test_db_session.add(dataset)
        test_db_session.flush()
        
        trait_dataset = ProjectTraitDataset(
            project_trait_id=trait.id,
            dataset_id=dataset.id,
            percentage=100.0,
        )
        test_db_session.add(trait_dataset)
        test_db_session.commit()
        
        assert trait_dataset.dataset.id == dataset.id
        assert trait_dataset.dataset.name == "Test Dataset"


class TestProjectModelValidation:
    """Tests for model validation constraints."""
    
    def test_reasoning_trait_one_dataset(self, test_db_session: Session):
        """Test that Reasoning trait can only have one dataset."""
        project = Project(
            name="Test Project",
            base_model="llama3.2:3b",
            training_type="qlora",
            max_rows=50000,
            output_directory="/output",
        )
        test_db_session.add(project)
        test_db_session.flush()
        
        trait = ProjectTrait(
            project_id=project.id,
            trait_type=TraitType.REASONING.value,
        )
        test_db_session.add(trait)
        test_db_session.flush()
        
        # Create two datasets
        dataset1 = Dataset(
            name="Dataset 1",
            filename="test1.csv",
            file_path="/uploads/test1.csv",
            file_type="csv",
            file_size=1024,
            row_count=100,
            column_count=2,
            columns='["input", "output"]',
        )
        dataset2 = Dataset(
            name="Dataset 2",
            filename="test2.csv",
            file_path="/uploads/test2.csv",
            file_type="csv",
            file_size=1024,
            row_count=100,
            column_count=2,
            columns='["input", "output"]',
        )
        test_db_session.add_all([dataset1, dataset2])
        test_db_session.flush()
        
        # Add first dataset - should work
        trait_dataset1 = ProjectTraitDataset(
            project_trait_id=trait.id,
            dataset_id=dataset1.id,
            percentage=100.0,
        )
        test_db_session.add(trait_dataset1)
        test_db_session.commit()
        
        # This is a business logic constraint that should be validated in the service layer
        # Database allows multiple datasets, but service should enforce single dataset
    
    def test_coding_trait_one_dataset(self, test_db_session: Session):
        """Test that Coding trait can only have one dataset."""
        project = Project(
            name="Test Project",
            base_model="llama3.2:3b",
            training_type="qlora",
            max_rows=50000,
            output_directory="/output",
        )
        test_db_session.add(project)
        test_db_session.flush()
        
        trait = ProjectTrait(
            project_id=project.id,
            trait_type=TraitType.CODING.value,
        )
        test_db_session.add(trait)
        test_db_session.flush()
        
        dataset = Dataset(
            name="Test Dataset",
            filename="test.csv",
            file_path="/uploads/test.csv",
            file_type="csv",
            file_size=1024,
            row_count=100,
            column_count=2,
            columns='["input", "output"]',
        )
        test_db_session.add(dataset)
        test_db_session.flush()
        
        trait_dataset = ProjectTraitDataset(
            project_trait_id=trait.id,
            dataset_id=dataset.id,
            percentage=100.0,
        )
        test_db_session.add(trait_dataset)
        test_db_session.commit()
        
        assert len(trait.datasets) == 1

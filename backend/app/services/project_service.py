"""
Project Service.

Manages project-based training with traits and dataset allocations.
"""

import logging
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models.project import Project, ProjectTrait, TraitType, ProjectTraitDataset, ProjectStatus
from app.models.dataset import Dataset

logger = logging.getLogger(__name__)


class ProjectValidationError(Exception):
    """Raised when project validation fails."""
    pass


class DatasetAllocationError(Exception):
    """Raised when dataset allocation validation fails."""
    pass


class ProjectService:
    """
    Service for managing training projects.
    
    Handles project creation, validation, and dataset combination
    based on trait configurations and percentages.
    """
    
    def __init__(self, db: Session):
        """
        Initialize the project service.
        
        Args:
            db: Database session.
        """
        self.db = db
    
    def create_project(self, project_data: Dict[str, Any]) -> Project:
        """
        Create a new training project with traits and dataset allocations.
        
        Args:
            project_data: Dictionary containing:
                - name: Project name.
                - description: Optional description.
                - base_model: Base model identifier.
                - training_type: Training type (qlora, unsloth, rag, standard).
                - max_rows: Maximum rows for training (50000, 100000, 250000, 500000, 1000000).
                - output_directory: Output directory path.
                - traits: List of trait configurations, each with:
                    - trait_type: Type of trait (reasoning, coding, general_tools).
                    - datasets: List of {dataset_id, percentage} dictionaries.
        
        Returns:
            Created Project instance.
            
        Raises:
            ProjectValidationError: If project validation fails.
            DatasetAllocationError: If dataset allocation validation fails.
        """
        # Validate project data
        self._validate_project_data(project_data)
        
        # Create project
        project = Project(
            name=project_data["name"],
            description=project_data.get("description"),
            base_model=project_data["base_model"],
            training_type=project_data["training_type"],
            max_rows=project_data["max_rows"],
            output_directory=project_data["output_directory"],
            status=ProjectStatus.PENDING.value,
        )
        self.db.add(project)
        self.db.flush()
        
        # Create traits and dataset allocations
        used_dataset_ids = set()
        
        for trait_data in project_data.get("traits", []):
            trait_type = trait_data["trait_type"]
            datasets = trait_data.get("datasets", [])
            
            # Validate trait
            self._validate_trait(trait_type, datasets)
            
            # Check for duplicate datasets across project
            for dataset_item in datasets:
                dataset_id = dataset_item["dataset_id"]
                if dataset_id in used_dataset_ids:
                    raise DatasetAllocationError(
                        f"Dataset {dataset_id} is used multiple times in the project. "
                        "Each dataset can only be used once per project."
                    )
                used_dataset_ids.add(dataset_id)
            
            # Create trait
            trait = ProjectTrait(
                project_id=project.id,
                trait_type=trait_type,
            )
            self.db.add(trait)
            self.db.flush()
            
            # Create dataset allocations
            total_percentage = 0.0
            for dataset_item in datasets:
                percentage = dataset_item["percentage"]
                total_percentage += percentage
                
                # Verify dataset exists
                dataset = self.db.query(Dataset).filter_by(id=dataset_item["dataset_id"]).first()
                if not dataset:
                    raise ProjectValidationError(f"Dataset {dataset_item['dataset_id']} not found")
                
                trait_dataset = ProjectTraitDataset(
                    project_trait_id=trait.id,
                    dataset_id=dataset_item["dataset_id"],
                    percentage=percentage,
                )
                self.db.add(trait_dataset)
            
            # Validate percentages sum to 100%
            if abs(total_percentage - 100.0) > 0.01:  # Allow small floating point errors
                raise DatasetAllocationError(
                    f"Dataset percentages for trait '{trait_type}' sum to {total_percentage}%, "
                    "but must sum to exactly 100%"
                )
        
        self.db.commit()
        self.db.refresh(project)
        
        logger.info(f"Created project: {project.id} - {project.name}")
        return project
    
    def get_project(self, project_id: int) -> Optional[Project]:
        """
        Get a project by ID.
        
        Args:
            project_id: Project ID.
            
        Returns:
            Project instance or None if not found.
        """
        return self.db.query(Project).filter_by(id=project_id).first()
    
    def list_projects(self, skip: int = 0, limit: int = 100) -> List[Project]:
        """
        List all projects.
        
        Args:
            skip: Number of projects to skip.
            limit: Maximum number of projects to return.
            
        Returns:
            List of Project instances.
        """
        return self.db.query(Project).offset(skip).limit(limit).all()
    
    def combine_datasets_for_training(
        self,
        project_id: int,
        max_rows: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Combine datasets from all traits based on percentages.
        
        Args:
            project_id: Project ID.
            max_rows: Maximum rows to use (defaults to project max_rows).
            
        Returns:
            List of dictionaries containing:
                - dataset_id: Dataset ID.
                - rows: Number of rows to use from this dataset.
                - percentage: Percentage allocation.
        """
        project = self.get_project(project_id)
        if not project:
            raise ProjectValidationError(f"Project {project_id} not found")
        
        if max_rows is None:
            max_rows = project.max_rows
        
        # Collect all datasets with their percentages
        dataset_allocations = []
        
        for trait in project.traits:
            for trait_dataset in trait.datasets:
                dataset_allocations.append({
                    "dataset_id": trait_dataset.dataset_id,
                    "percentage": trait_dataset.percentage,
                    "dataset": trait_dataset.dataset,
                })
        
        # Calculate rows per dataset
        combined = []
        total_allocated = 0
        
        for allocation in dataset_allocations:
            dataset = allocation["dataset"]
            percentage = allocation["percentage"]
            
            # Calculate rows based on percentage
            rows = int((max_rows * percentage) / 100.0)
            
            # Don't exceed dataset size
            rows = min(rows, dataset.row_count)
            
            combined.append({
                "dataset_id": allocation["dataset_id"],
                "rows": rows,
                "percentage": percentage,
                "dataset": dataset,
            })
            
            total_allocated += rows
        
        # If we have leftover capacity, distribute proportionally
        if total_allocated < max_rows:
            remaining = max_rows - total_allocated
            # Distribute remaining rows proportionally to datasets with available data
            for item in combined:
                available = item["dataset"].row_count - item["rows"]
                if available > 0 and remaining > 0:
                    add_rows = min(available, remaining // len(combined))
                    item["rows"] += add_rows
                    remaining -= add_rows
        
        return combined
    
    def _validate_project_data(self, project_data: Dict[str, Any]) -> None:
        """
        Validate project data structure.
        
        Args:
            project_data: Project data dictionary.
            
        Raises:
            ProjectValidationError: If validation fails.
        """
        required_fields = ["name", "base_model", "training_type", "max_rows", "output_directory"]
        for field in required_fields:
            if field not in project_data:
                raise ProjectValidationError(f"Missing required field: {field}")
        
        # Validate max_rows is one of the allowed values
        allowed_rows = [50000, 100000, 250000, 500000, 1000000]
        if project_data["max_rows"] not in allowed_rows:
            raise ProjectValidationError(
                f"max_rows must be one of {allowed_rows}, got {project_data['max_rows']}"
            )
    
    def _validate_trait(self, trait_type: str, datasets: List[Dict[str, Any]]) -> None:
        """
        Validate trait configuration.
        
        Args:
            trait_type: Type of trait.
            datasets: List of dataset allocations.
            
        Raises:
            ProjectValidationError: If validation fails.
        """
        if trait_type == TraitType.REASONING.value:
            if len(datasets) != 1:
                raise ProjectValidationError(
                    f"Reasoning trait must have exactly one dataset, got {len(datasets)}"
                )
        elif trait_type == TraitType.CODING.value:
            if len(datasets) != 1:
                raise ProjectValidationError(
                    f"Coding trait must have exactly one dataset, got {len(datasets)}"
                )
        elif trait_type == TraitType.GENERAL_TOOLS.value:
            if len(datasets) < 1:
                raise ProjectValidationError(
                    "General/Tools trait must have at least one dataset"
                )
        else:
            raise ProjectValidationError(f"Unknown trait type: {trait_type}")

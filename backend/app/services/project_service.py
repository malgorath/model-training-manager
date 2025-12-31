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
            model_type=project_data.get("model_type"),
            training_type=project_data["training_type"],
            max_rows=project_data.get("max_rows"),
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
            for dataset_item in datasets:
                percentage = dataset_item["percentage"]
                
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
        
        # Note: max_rows is now a cap, not a requirement
        # Total rows can be less than max_rows based on percentages
        
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

    def cancel_project(self, project_id: int) -> Project:
        """
        Cancel a running or pending project.
        
        Handles projects stuck in running state even when no workers are active.
        
        Args:
            project_id: Project ID to cancel.
            
        Returns:
            Cancelled Project instance.
            
        Raises:
            ProjectValidationError: If project not found or cannot be cancelled.
        """
        project = self.get_project(project_id)
        if not project:
            raise ProjectValidationError(f"Project {project_id} not found")
        
        if project.status in (ProjectStatus.COMPLETED.value, ProjectStatus.CANCELLED.value, ProjectStatus.FAILED.value):
            raise ProjectValidationError(f"Cannot cancel project with status: {project.status}")
        
        # Cancel in worker pool if running (even if no workers, this will clean up queue)
        if project.status == ProjectStatus.RUNNING.value:
            from app.services.training_service import TrainingService
            training_service = TrainingService(db=self.db)
            cancelled_in_pool = training_service.worker_pool.cancel_job(project.id)
            # If project wasn't in pool (e.g., workers stopped but project still marked running),
            # we still cancel it - this handles stuck projects
            if not cancelled_in_pool:
                logger.warning(f"Project {project_id} was running but not found in worker pool - cancelling anyway (likely workers were stopped)")
        
        # Always update status to cancelled, even if not found in worker pool
        # This handles the case where workers were stopped but project is still marked running
        project.status = ProjectStatus.CANCELLED.value
        project.worker_id = None  # Clear worker assignment
        
        self.db.commit()
        self.db.refresh(project)
        
        logger.info(f"Cancelled project: {project_id} - {project.name} (status was: {project.status})")
        return project
    
    def delete_project(self, project_id: int) -> bool:
        """
        Delete a project.
        
        Projects that are currently running cannot be deleted.
        Cascade delete will automatically remove related traits and dataset allocations.
        
        Args:
            project_id: Project ID to delete.
            
        Returns:
            True if project was deleted, False if not found.
            
        Raises:
            ProjectValidationError: If project is in a non-deletable state (e.g., RUNNING).
        """
        project = self.get_project(project_id)
        if not project:
            return False
        
        # Check if project can be deleted (not running)
        if project.status == ProjectStatus.RUNNING.value:
            raise ProjectValidationError(
                f"Cannot delete project {project_id} with status '{project.status}'. "
                "Please cancel or wait for the project to complete before deleting."
            )
        
        # Delete project (cascade will handle traits and dataset allocations)
        self.db.delete(project)
        self.db.commit()
        
        logger.info(f"Deleted project: {project_id} - {project.name}")
        return True

    def get_dataset_row_counts(self, project: Project) -> Dict[int, int]:
        """
        Calculate the actual number of rows to use from each dataset.
        
        Each dataset uses: dataset.row_count * percentage / 100
        
        Args:
            project: Project instance.
            
        Returns:
            Dictionary mapping dataset_id to number of rows to use.
        """
        row_counts = {}
        
        for trait in project.traits:
            for trait_dataset in trait.datasets:
                dataset = trait_dataset.dataset
                percentage = trait_dataset.percentage
                
                # Calculate rows: percentage of the dataset file
                rows = int((dataset.row_count * percentage) / 100.0)
                
                # Sum if dataset used in multiple traits (shouldn't happen, but handle it)
                if trait_dataset.dataset_id in row_counts:
                    row_counts[trait_dataset.dataset_id] += rows
                else:
                    row_counts[trait_dataset.dataset_id] = rows
        
        return row_counts
    
    def combine_datasets_for_training(
        self,
        project_id: int,
        max_rows: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Combine datasets from all traits based on percentages.
        
        Each dataset uses: dataset.row_count * percentage / 100 rows.
        
        Args:
            project_id: Project ID.
            max_rows: Not used - kept for API compatibility.
            
        Returns:
            List of dictionaries containing:
                - dataset_id: Dataset ID.
                - rows: Number of rows to use from this dataset (calculated as dataset.row_count * percentage / 100).
                - percentage: Percentage of dataset file to use.
        """
        project = self.get_project(project_id)
        if not project:
            raise ProjectValidationError(f"Project {project_id} not found")
        
        # Collect all datasets with their percentages
        dataset_allocations = []
        
        for trait in project.traits:
            for trait_dataset in trait.datasets:
                dataset_allocations.append({
                    "dataset_id": trait_dataset.dataset_id,
                    "percentage": trait_dataset.percentage,
                    "dataset": trait_dataset.dataset,
                })
        
        # Calculate rows per dataset: percentage of the dataset file
        combined = []
        
        for allocation in dataset_allocations:
            dataset = allocation["dataset"]
            percentage = allocation["percentage"]
            
            # Calculate rows: percentage of the dataset file
            rows = int((dataset.row_count * percentage) / 100.0)
            
            combined.append({
                "dataset_id": allocation["dataset_id"],
                "rows": rows,
                "percentage": percentage,
                "dataset": dataset,
            })
        
        return combined
    
    def _validate_project_data(self, project_data: Dict[str, Any]) -> None:
        """
        Validate project data structure.
        
        Args:
            project_data: Project data dictionary.
            
        Raises:
            ProjectValidationError: If validation fails.
        """
        required_fields = ["name", "base_model", "training_type", "output_directory"]
        for field in required_fields:
            if field not in project_data:
                raise ProjectValidationError(f"Missing required field: {field}")
        
    
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
        
        # Percentages are per-file usage amounts, no validation needed
        # Each percentage indicates how much of that specific dataset file to use
    
    def _validate_dataset_totals(self, project: Project) -> None:
        """
        Validate dataset totals (deprecated - max_rows is now a cap, not a requirement).
        
        This method is kept for backwards compatibility but no longer raises errors.
        max_rows is now treated as a maximum cap, and the actual total can be less.
        
        Args:
            project: Project instance to validate.
        """
        # No validation needed - max_rows is a cap, not a requirement
        pass
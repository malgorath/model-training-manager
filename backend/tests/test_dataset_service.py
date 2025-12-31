"""
Tests for the Dataset service.

Tests dataset upload, validation, storage, and CRUD operations.
"""

import json
import pytest
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.models.dataset import Dataset
from app.schemas.dataset import DatasetCreate, DatasetUpdate
from app.services.dataset_service import DatasetService


class TestDatasetService:
    """Tests for DatasetService."""
    
    @pytest.fixture
    def service(self, test_db_session: Session, temp_upload_dir: Path) -> DatasetService:
        """Create a DatasetService instance for testing."""
        with patch("app.services.dataset_service.settings") as mock_settings:
            mock_settings.allowed_extensions = {".csv", ".json"}
            mock_settings.max_upload_size = 100 * 1024 * 1024
            mock_settings.get_dataset_path.return_value = temp_upload_dir / "user" / "test_dataset"
            
            service = DatasetService(test_db_session)
            return service
    
    @pytest.fixture
    def mock_csv_upload(self, sample_csv_content: bytes) -> UploadFile:
        """Create a mock CSV upload file."""
        file = MagicMock(spec=UploadFile)
        file.filename = "test_data.csv"
        file.read = AsyncMock(return_value=sample_csv_content)
        return file
    
    @pytest.fixture
    def mock_json_upload(self, sample_json_content: bytes) -> UploadFile:
        """Create a mock JSON upload file."""
        file = MagicMock(spec=UploadFile)
        file.filename = "test_data.json"
        file.read = AsyncMock(return_value=sample_json_content)
        return file
    
    @pytest.mark.asyncio
    async def test_create_dataset_csv(
        self,
        service: DatasetService,
        mock_csv_upload: UploadFile,
    ):
        """Test creating a dataset from a CSV file."""
        dataset_data = DatasetCreate(
            name="Test CSV Dataset",
            description="A test CSV dataset",
        )
        
        dataset = await service.create_dataset(mock_csv_upload, dataset_data)
        
        assert dataset.id is not None
        assert dataset.name == "Test CSV Dataset"
        assert dataset.description == "A test CSV dataset"
        assert dataset.file_type == "csv"
        assert dataset.row_count == 3  # 3 data rows
        assert dataset.column_count == 2  # input, output
        assert "input" in dataset.columns
        assert "output" in dataset.columns
    
    @pytest.mark.asyncio
    async def test_create_dataset_json(
        self,
        service: DatasetService,
        mock_json_upload: UploadFile,
    ):
        """Test creating a dataset from a JSON file."""
        dataset_data = DatasetCreate(
            name="Test JSON Dataset",
            description="A test JSON dataset",
        )
        
        dataset = await service.create_dataset(mock_json_upload, dataset_data)
        
        assert dataset.id is not None
        assert dataset.name == "Test JSON Dataset"
        assert dataset.file_type == "json"
        assert dataset.row_count == 2
        assert dataset.column_count == 2
    
    @pytest.mark.asyncio
    async def test_create_dataset_invalid_extension(
        self,
        service: DatasetService,
    ):
        """Test that invalid file extensions are rejected."""
        file = MagicMock(spec=UploadFile)
        file.filename = "test.txt"
        file.read = AsyncMock(return_value=b"some content")
        
        dataset_data = DatasetCreate(name="Invalid")
        
        with pytest.raises(ValueError) as exc_info:
            await service.create_dataset(file, dataset_data)
        
        assert "Unsupported file type" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_create_dataset_invalid_csv(
        self,
        service: DatasetService,
    ):
        """Test that invalid CSV content is rejected."""
        file = MagicMock(spec=UploadFile)
        file.filename = "test.csv"
        file.read = AsyncMock(return_value=b"")  # Empty file
        
        dataset_data = DatasetCreate(name="Empty CSV")
        
        with pytest.raises(ValueError) as exc_info:
            await service.create_dataset(file, dataset_data)
        
        assert "Failed to parse" in str(exc_info.value) or "empty" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_create_dataset_invalid_json(
        self,
        service: DatasetService,
    ):
        """Test that invalid JSON content is rejected."""
        file = MagicMock(spec=UploadFile)
        file.filename = "test.json"
        file.read = AsyncMock(return_value=b'{"not": "an array"}')
        
        dataset_data = DatasetCreate(name="Invalid JSON")
        
        with pytest.raises(ValueError) as exc_info:
            await service.create_dataset(file, dataset_data)
        
        assert "array" in str(exc_info.value).lower()
    
    def test_get_dataset(
        self,
        service: DatasetService,
        test_db_session: Session,
    ):
        """Test getting a dataset by ID."""
        # Create a dataset directly
        dataset = Dataset(
            name="Test Dataset",
            filename="test.csv",
            file_path="/path/to/test.csv",
            file_type="csv",
            file_size=100,
            row_count=10,
            column_count=2,
        )
        test_db_session.add(dataset)
        test_db_session.commit()
        
        # Get the dataset
        retrieved = service.get_dataset(dataset.id)
        
        assert retrieved is not None
        assert retrieved.id == dataset.id
        assert retrieved.name == "Test Dataset"
    
    def test_get_dataset_not_found(self, service: DatasetService):
        """Test getting a non-existent dataset."""
        result = service.get_dataset(99999)
        assert result is None
    
    def test_list_datasets(
        self,
        service: DatasetService,
        test_db_session: Session,
    ):
        """Test listing datasets with pagination."""
        # Create multiple datasets
        for i in range(15):
            dataset = Dataset(
                name=f"Dataset {i}",
                filename=f"data{i}.csv",
                file_path=f"/path/data{i}.csv",
                file_type="csv",
                file_size=100,
            )
            test_db_session.add(dataset)
        test_db_session.commit()
        
        # Test first page
        result = service.list_datasets(page=1, page_size=10)
        
        assert result["total"] == 15
        assert result["page"] == 1
        assert result["page_size"] == 10
        assert result["pages"] == 2
        assert len(result["items"]) == 10
        
        # Test second page
        result = service.list_datasets(page=2, page_size=10)
        assert len(result["items"]) == 5
    
    def test_update_dataset(
        self,
        service: DatasetService,
        test_db_session: Session,
    ):
        """Test updating a dataset's metadata."""
        dataset = Dataset(
            name="Original Name",
            filename="test.csv",
            file_path="/path/test.csv",
            file_type="csv",
            file_size=100,
        )
        test_db_session.add(dataset)
        test_db_session.commit()
        
        update_data = DatasetUpdate(
            name="Updated Name",
            description="New description",
        )
        
        updated = service.update_dataset(dataset.id, update_data)
        
        assert updated is not None
        assert updated.name == "Updated Name"
        assert updated.description == "New description"
    
    def test_update_dataset_not_found(self, service: DatasetService):
        """Test updating a non-existent dataset."""
        update_data = DatasetUpdate(name="New Name")
        result = service.update_dataset(99999, update_data)
        assert result is None
    
    def test_delete_dataset(
        self,
        service: DatasetService,
        test_db_session: Session,
        temp_upload_dir: Path,
    ):
        """Test deleting a dataset."""
        # Create a file in the new structure: data/user/datasetname/
        dataset_dir = temp_upload_dir / "user" / "test_delete"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        file_path = dataset_dir / "test_delete.csv"
        file_path.write_text("input,output\na,b\n")
        
        dataset = Dataset(
            name="To Delete",
            filename="test_delete.csv",
            file_path=str(file_path),
            file_type="csv",
            file_size=100,
        )
        test_db_session.add(dataset)
        test_db_session.commit()
        dataset_id = dataset.id
        
        # Delete
        result = service.delete_dataset(dataset_id)
        
        assert result is True
        assert service.get_dataset(dataset_id) is None
        assert not file_path.exists()
    
    def test_delete_dataset_not_found(self, service: DatasetService):
        """Test deleting a non-existent dataset."""
        result = service.delete_dataset(99999)
        assert result is False
    
    def test_scan_datasets_finds_new_files(
        self,
        service: DatasetService,
        test_db_session: Session,
        temp_upload_dir: Path,
    ):
        """Test scanning directories and auto-adding valid datasets."""
        # Create test dataset files in the expected structure
        author_dir = temp_upload_dir / "user" / "scanned_dataset"
        author_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a valid CSV file
        csv_file = author_dir / "test_scanned.csv"
        csv_file.write_text("input,output\nHello,World\nTest,Data\n")
        
        # Create a valid JSON file
        json_file = author_dir / "test_scanned.json"
        json_file.write_text('[{"input": "Hello", "output": "World"}]')
        
        # Mock settings to return our temp directory
        with patch("app.services.dataset_service.settings") as mock_settings:
            mock_settings.get_upload_path.return_value = temp_upload_dir
            mock_settings.allowed_extensions = {".csv", ".json"}
            
            # Scan for datasets
            result = service.scan_datasets()
        
        # Should find and add 2 datasets
        assert result["scanned"] == 2
        assert result["added"] == 2
        assert result["skipped"] == 0
        assert len(result["added_datasets"]) == 2
        
        # Verify datasets were added to database
        datasets = test_db_session.query(Dataset).all()
        assert len(datasets) == 2
        
        # Verify file paths are correct
        file_paths = {d.file_path for d in datasets}
        assert str(csv_file) in file_paths
        assert str(json_file) in file_paths
    
    def test_scan_datasets_skips_existing(
        self,
        service: DatasetService,
        test_db_session: Session,
        temp_upload_dir: Path,
    ):
        """Test that scanning skips datasets already in database."""
        # Create a dataset file
        author_dir = temp_upload_dir / "user" / "existing_dataset"
        author_dir.mkdir(parents=True, exist_ok=True)
        csv_file = author_dir / "existing.csv"
        csv_file.write_text("input,output\nHello,World\n")
        
        # Add it to database first
        existing_dataset = Dataset(
            name="Existing Dataset",
            filename="existing.csv",
            file_path=str(csv_file),
            file_type="csv",
            file_size=100,
            row_count=1,
            column_count=2,
        )
        test_db_session.add(existing_dataset)
        test_db_session.commit()
        
        # Scan should skip it
        with patch("app.services.dataset_service.settings") as mock_settings:
            mock_settings.get_upload_path.return_value = temp_upload_dir
            mock_settings.allowed_extensions = {".csv", ".json"}
            
            result = service.scan_datasets()
        
        assert result["scanned"] == 1
        assert result["added"] == 0
        assert result["skipped"] == 1
        assert len(result["skipped_paths"]) == 1
        assert str(csv_file) in result["skipped_paths"]
    
    def test_scan_datasets_handles_invalid_files(
        self,
        service: DatasetService,
        temp_upload_dir: Path,
    ):
        """Test that scanning handles invalid files gracefully."""
        # Create invalid files
        author_dir = temp_upload_dir / "user" / "invalid_dataset"
        author_dir.mkdir(parents=True, exist_ok=True)
        
        # Invalid JSON (not an array)
        invalid_json = author_dir / "invalid.json"
        invalid_json.write_text('{"not": "an array"}')
        
        # Empty CSV
        empty_csv = author_dir / "empty.csv"
        empty_csv.write_text("")
        
        with patch("app.services.dataset_service.settings") as mock_settings:
            mock_settings.get_upload_path.return_value = temp_upload_dir
            mock_settings.allowed_extensions = {".csv", ".json"}
            
            result = service.scan_datasets()
        
        # Should skip invalid files
        assert result["scanned"] >= 2
        assert result["added"] == 0
        assert result["skipped"] >= 2
    
    def test_scan_datasets_handles_multiple_authors(
        self,
        service: DatasetService,
        test_db_session: Session,
        temp_upload_dir: Path,
    ):
        """Test scanning datasets from multiple authors."""
        # Create datasets for different authors
        user_dir = temp_upload_dir / "user" / "user_dataset"
        user_dir.mkdir(parents=True, exist_ok=True)
        user_file = user_dir / "user_data.csv"
        user_file.write_text("input,output\nHello,World\n")
        
        author_dir = temp_upload_dir / "test_author" / "author_dataset"
        author_dir.mkdir(parents=True, exist_ok=True)
        author_file = author_dir / "author_data.csv"
        author_file.write_text("input,output\nTest,Data\n")
        
        with patch("app.services.dataset_service.settings") as mock_settings:
            mock_settings.get_upload_path.return_value = temp_upload_dir
            mock_settings.allowed_extensions = {".csv", ".json"}
            
            result = service.scan_datasets()
        
        assert result["added"] == 2
        datasets = test_db_session.query(Dataset).all()
        assert len(datasets) == 2


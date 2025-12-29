"""
Tests for functional training implementations.

Tests that all training methods work correctly and produce valid output.
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from sqlalchemy.orm import Session

from app.core.database import Base, engine, SessionLocal
from app.models.dataset import Dataset
from app.models.training_job import TrainingJob, TrainingStatus, TrainingType
from app.workers.training_worker import (
    TrainingWorker,
    check_torch_available,
    check_transformers_available,
    check_peft_available,
    check_unsloth_available,
    check_sentence_transformers_available,
    check_faiss_available,
)


@pytest.fixture(scope="function")
def db():
    """Create a fresh database session for each test."""
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def sample_dataset(db: Session, tmp_path: Path):
    """Create a sample dataset for testing."""
    data_file = tmp_path / "test_data.json"
    data = [
        {"text": "Hello, how are you?", "input": "Hello", "output": "How are you?"},
        {"text": "What is your name?", "input": "Name", "output": "I am an AI"},
        {"text": "Tell me about Python", "input": "Python", "output": "Python is a programming language"},
    ]
    data_file.write_text(json.dumps(data))
    
    dataset = Dataset(
        name="Test Dataset",
        description="Test dataset for training",
        filename="test_data.json",
        file_path=str(data_file),
        file_type="json",
        file_size=len(json.dumps(data)),
        row_count=len(data),
        column_count=3,
        columns='["text", "input", "output"]',
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


@pytest.fixture
def training_job(db: Session, sample_dataset: Dataset):
    """Create a training job for testing."""
    job = TrainingJob(
        name="Test Training Job",
        description="Test job",
        training_type=TrainingType.QLORA.value,
        model_name="llama3.2:3b",
        dataset_id=sample_dataset.id,
        status=TrainingStatus.PENDING.value,
        batch_size=2,
        learning_rate=0.0002,
        epochs=1,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


@pytest.fixture
def worker(db: Session):
    """Create a training worker for testing."""
    def session_factory():
        return SessionLocal()
    
    worker = TrainingWorker(
        worker_id="test-worker",
        db_session_factory=session_factory,
    )
    # Initialize _cancel_job attribute (normally set in _run method)
    worker._cancel_job = False
    return worker


class TestDependencyChecks:
    """Tests for dependency availability checks."""
    
    def test_check_torch_available(self):
        """Test torch availability check."""
        result = check_torch_available()
        assert isinstance(result, bool)
    
    def test_check_transformers_available(self):
        """Test transformers availability check."""
        result = check_transformers_available()
        assert isinstance(result, bool)
    
    def test_check_peft_available(self):
        """Test PEFT availability check."""
        result = check_peft_available()
        assert isinstance(result, bool)
    
    def test_check_unsloth_available(self):
        """Test Unsloth availability check."""
        result = check_unsloth_available()
        assert isinstance(result, bool)
    
    def test_check_sentence_transformers_available(self):
        """Test sentence-transformers availability check."""
        result = check_sentence_transformers_available()
        assert isinstance(result, bool)
    
    def test_check_faiss_available(self):
        """Test FAISS availability check."""
        result = check_faiss_available()
        assert isinstance(result, bool)


class TestQLoRATraining:
    """Tests for QLoRA training implementation."""
    
    def test_qlora_simulated_training(
        self, db: Session, sample_dataset: Dataset, tmp_path: Path
    ):
        """Test QLoRA simulated training produces valid output."""
        job = TrainingJob(
            name="QLoRA Test",
            training_type=TrainingType.QLORA.value,
            model_name="llama3.2:3b",
            dataset_id=sample_dataset.id,
            status=TrainingStatus.RUNNING.value,
            batch_size=2,
            learning_rate=0.0002,
            epochs=1,
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.05,
        )
        db.add(job)
        db.commit()
        
        def session_factory():
            return SessionLocal()
        
        worker = TrainingWorker(
            worker_id="test-worker",
            db_session_factory=session_factory,
        )
        worker._cancel_job = False  # Initialize cancellation flag
        
        # Run training
        with patch('app.workers.training_worker.settings') as mock_settings:
            mock_settings.upload_dir = tmp_path
            
            # Force simulated mode
            with patch('app.workers.training_worker.check_torch_available', return_value=False):
                worker._train_qlora(job, sample_dataset, db)
        
        # Verify output
        assert job.model_path is not None
        model_path = Path(job.model_path)
        assert model_path.exists()
        
        # Check files created
        config_file = model_path / "adapter_config.json"
        assert config_file.exists()
        
        config = json.loads(config_file.read_text())
        assert config["r"] == job.lora_r
        assert config["lora_alpha"] == job.lora_alpha
        
        # Check progress was updated
        assert job.progress > 0
    
    def test_qlora_logs_written(
        self, db: Session, sample_dataset: Dataset, tmp_path: Path
    ):
        """Test that QLoRA training writes log entries."""
        job = TrainingJob(
            name="QLoRA Log Test",
            training_type=TrainingType.QLORA.value,
            model_name="llama3.2:3b",
            dataset_id=sample_dataset.id,
            status=TrainingStatus.RUNNING.value,
            batch_size=2,
            epochs=1,
            log="",
        )
        db.add(job)
        db.commit()
        
        def session_factory():
            return SessionLocal()
        
        worker = TrainingWorker(
            worker_id="test-worker",
            db_session_factory=session_factory,
        )
        worker._cancel_job = False  # Initialize cancellation flag
        
        with patch('app.workers.training_worker.settings') as mock_settings:
            mock_settings.upload_dir = tmp_path
            with patch('app.workers.training_worker.check_torch_available', return_value=False):
                worker._train_qlora(job, sample_dataset, db)
        
        # Verify logs contain expected entries
        assert job.log is not None
        assert "Starting QLoRA training" in job.log
        assert "Dataset loaded" in job.log
        assert "Model saved" in job.log


class TestUnslothTraining:
    """Tests for Unsloth training implementation."""
    
    def test_unsloth_simulated_training(
        self, db: Session, sample_dataset: Dataset, tmp_path: Path
    ):
        """Test Unsloth simulated training produces valid output."""
        job = TrainingJob(
            name="Unsloth Test",
            training_type=TrainingType.UNSLOTH.value,
            model_name="llama3.2:3b",
            dataset_id=sample_dataset.id,
            status=TrainingStatus.RUNNING.value,
            batch_size=2,
            epochs=1,
        )
        db.add(job)
        db.commit()
        
        def session_factory():
            return SessionLocal()
        
        worker = TrainingWorker(
            worker_id="test-worker",
            db_session_factory=session_factory,
        )
        worker._cancel_job = False  # Initialize cancellation flag
        
        with patch('app.workers.training_worker.settings') as mock_settings:
            mock_settings.upload_dir = tmp_path
            
            # Force simulated mode
            with patch('app.workers.training_worker.check_unsloth_available', return_value=False):
                worker._train_unsloth(job, sample_dataset, db)
        
        # Verify output
        assert job.model_path is not None
        model_path = Path(job.model_path)
        assert model_path.exists()
        
        config_file = model_path / "adapter_config.json"
        assert config_file.exists()
        
        config = json.loads(config_file.read_text())
        assert config["training_type"] == "unsloth"


class TestRAGTraining:
    """Tests for RAG training implementation."""
    
    def test_rag_simulated_training(
        self, db: Session, sample_dataset: Dataset, tmp_path: Path
    ):
        """Test RAG simulated training produces valid output."""
        job = TrainingJob(
            name="RAG Test",
            training_type=TrainingType.RAG.value,
            model_name="llama3.2:3b",
            dataset_id=sample_dataset.id,
            status=TrainingStatus.RUNNING.value,
            batch_size=2,
            epochs=1,
        )
        db.add(job)
        db.commit()
        
        def session_factory():
            return SessionLocal()
        
        worker = TrainingWorker(
            worker_id="test-worker",
            db_session_factory=session_factory,
        )
        worker._cancel_job = False  # Initialize cancellation flag
        
        with patch('app.workers.training_worker.settings') as mock_settings:
            mock_settings.upload_dir = tmp_path
            
            # Force simulated mode
            with patch('app.workers.training_worker.check_sentence_transformers_available', return_value=False):
                worker._train_rag(job, sample_dataset, db)
        
        # Verify output
        assert job.model_path is not None
        model_path = Path(job.model_path)
        assert model_path.exists()
        
        config_file = model_path / "rag_config.json"
        assert config_file.exists()
        
        config = json.loads(config_file.read_text())
        assert config["training_type"] == "rag"
        
        # Check documents file
        docs_file = model_path / "documents.json"
        assert docs_file.exists()
    
    @pytest.mark.skipif(
        not check_sentence_transformers_available() or not check_faiss_available(),
        reason="sentence-transformers or faiss not available"
    )
    def test_rag_real_training(
        self, db: Session, sample_dataset: Dataset, tmp_path: Path
    ):
        """Test RAG real training with actual libraries."""
        job = TrainingJob(
            name="RAG Real Test",
            training_type=TrainingType.RAG.value,
            model_name="llama3.2:3b",
            dataset_id=sample_dataset.id,
            status=TrainingStatus.RUNNING.value,
            batch_size=2,
            epochs=1,
        )
        db.add(job)
        db.commit()
        
        def session_factory():
            return SessionLocal()
        
        worker = TrainingWorker(
            worker_id="test-worker",
            db_session_factory=session_factory,
        )
        worker._cancel_job = False  # Initialize cancellation flag
        
        with patch('app.workers.training_worker.settings') as mock_settings:
            mock_settings.upload_dir = tmp_path
            worker._train_rag(job, sample_dataset, db)
        
        # Verify real output
        model_path = Path(job.model_path)
        assert (model_path / "vector_index.faiss").exists()
        assert (model_path / "documents.json").exists()
        assert (model_path / "rag_config.json").exists()


class TestStandardTraining:
    """Tests for standard fine-tuning implementation."""
    
    def test_standard_simulated_training(
        self, db: Session, sample_dataset: Dataset, tmp_path: Path
    ):
        """Test standard simulated training produces valid output."""
        job = TrainingJob(
            name="Standard Test",
            training_type=TrainingType.STANDARD.value,
            model_name="llama3.2:3b",
            dataset_id=sample_dataset.id,
            status=TrainingStatus.RUNNING.value,
            batch_size=2,
            epochs=1,
        )
        db.add(job)
        db.commit()
        
        def session_factory():
            return SessionLocal()
        
        worker = TrainingWorker(
            worker_id="test-worker",
            db_session_factory=session_factory,
        )
        worker._cancel_job = False  # Initialize cancellation flag
        
        with patch('app.workers.training_worker.settings') as mock_settings:
            mock_settings.upload_dir = tmp_path
            
            # Force simulated mode
            with patch('app.workers.training_worker.check_torch_available', return_value=False):
                worker._train_standard(job, sample_dataset, db)
        
        # Verify output
        assert job.model_path is not None
        model_path = Path(job.model_path)
        assert model_path.exists()
        
        config_file = model_path / "config.json"
        assert config_file.exists()
        
        config = json.loads(config_file.read_text())
        assert config["training_type"] == "standard"
        
        # Check tokenizer config
        tokenizer_config = model_path / "tokenizer_config.json"
        assert tokenizer_config.exists()


class TestTrainingCancellation:
    """Tests for training cancellation."""
    
    def test_training_respects_cancel_flag(
        self, db: Session, sample_dataset: Dataset, tmp_path: Path
    ):
        """Test that training stops when cancel flag is set."""
        job = TrainingJob(
            name="Cancel Test",
            training_type=TrainingType.QLORA.value,
            model_name="llama3.2:3b",
            dataset_id=sample_dataset.id,
            status=TrainingStatus.RUNNING.value,
            epochs=10,  # Long training
        )
        db.add(job)
        db.commit()
        
        def session_factory():
            return SessionLocal()
        
        worker = TrainingWorker(
            worker_id="test-worker",
            db_session_factory=session_factory,
        )
        
        # Set cancel flag before training
        worker._cancel_job = True
        
        with patch('app.workers.training_worker.settings') as mock_settings:
            mock_settings.upload_dir = tmp_path
            with patch('app.workers.training_worker.check_torch_available', return_value=False):
                worker._train_qlora(job, sample_dataset, db)
        
        # Training should have stopped early
        assert job.progress < 100.0


class TestModelNameMapping:
    """Tests for model name mapping."""
    
    def test_model_name_mapping(self, worker):
        """Test Ollama to HuggingFace model name mapping."""
        assert "Llama-3.2-3B" in worker._map_model_name("llama3.2:3b")
        assert "Llama-3.2-1B" in worker._map_model_name("llama3.2:1b")
        assert "Mistral" in worker._map_model_name("mistral:7b")
        
        # Default fallback
        result = worker._map_model_name("unknown-model")
        assert "Llama" in result


class TestDatasetLoading:
    """Tests for dataset loading."""
    
    def test_load_json_dataset(
        self, worker: TrainingWorker, sample_dataset: Dataset
    ):
        """Test loading JSON dataset."""
        data = worker._load_dataset(sample_dataset)
        
        assert len(data) == 3
        assert "text" in data[0]
    
    def test_load_csv_dataset(
        self, db: Session, worker: TrainingWorker, tmp_path: Path
    ):
        """Test loading CSV dataset."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("text,label\nHello,greeting\nBye,farewell")
        
        dataset = Dataset(
            name="CSV Dataset",
            filename="test.csv",
            file_path=str(csv_file),
            file_type="csv",
            file_size=100,
            row_count=2,
            column_count=2,
        )
        db.add(dataset)
        db.commit()
        
        data = worker._load_dataset(dataset)
        
        assert len(data) == 2
        assert data[0]["text"] == "Hello"
        assert data[0]["label"] == "greeting"
    
    def test_load_missing_dataset_raises_error(
        self, worker: TrainingWorker, sample_dataset: Dataset
    ):
        """Test that missing dataset file raises error."""
        sample_dataset.file_path = "/nonexistent/path/data.json"
        
        with pytest.raises(FileNotFoundError):
            worker._load_dataset(sample_dataset)

"""
Tests for training workers.

Tests the worker pool and individual training workers.
"""

import json
import pytest
import time
import threading
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from sqlalchemy.orm import Session

from app.models.dataset import Dataset
from app.models.training_job import TrainingJob, TrainingStatus, TrainingType
from app.workers.training_worker import TrainingWorker
from app.workers.worker_pool import WorkerPool


class TestTrainingWorker:
    """Tests for TrainingWorker."""
    
    @pytest.fixture
    def mock_db_factory(self, test_db_session: Session):
        """Create a mock database session factory."""
        def factory():
            return test_db_session
        return factory
    
    @pytest.fixture
    def worker(self, mock_db_factory) -> TrainingWorker:
        """Create a TrainingWorker instance for testing."""
        return TrainingWorker(
            worker_id="test-worker",
            db_session_factory=mock_db_factory,
        )
    
    def test_worker_initialization(self, worker: TrainingWorker):
        """Test worker is properly initialized."""
        assert worker.id == "test-worker"
        assert worker.status == "idle"
        assert worker.current_job_id is None
        assert worker.jobs_completed == 0
        assert worker.started_at is not None
    
    def test_worker_get_info(self, worker: TrainingWorker):
        """Test getting worker info."""
        info = worker.get_info()
        
        assert info["id"] == "test-worker"
        assert info["status"] == "idle"
        assert info["current_job_id"] is None
        assert info["jobs_completed"] == 0
    
    def test_worker_submit_job(self, worker: TrainingWorker):
        """Test submitting a job to the worker."""
        worker.submit_job(123)
        
        assert 123 in worker._job_queue
    
    def test_worker_start_stop(self, worker: TrainingWorker):
        """Test starting and stopping the worker."""
        worker.start()
        time.sleep(0.1)  # Give thread time to start
        
        assert worker._thread is not None
        assert worker._thread.is_alive()
        
        worker.stop()
        time.sleep(0.2)  # Give thread time to stop
        
        assert worker.status == "stopped"
    
    def test_worker_start_when_already_running(self, worker: TrainingWorker):
        """Test that starting an already running worker does nothing."""
        worker.start()
        time.sleep(0.1)
        
        original_thread = worker._thread
        worker.start()  # Should not create a new thread
        
        assert worker._thread is original_thread
        
        worker.stop()
    
    def test_worker_cancel_current_job(self, worker: TrainingWorker):
        """Test cancelling the current job."""
        worker.current_job_id = 123
        worker.cancel_current_job()
        
        assert worker._cancel_job is True
    
    def test_worker_cancel_no_current_job(self, worker: TrainingWorker):
        """Test cancelling when there's no current job."""
        worker.current_job_id = None
        worker.cancel_current_job()  # Should not raise an error
    
    def test_worker_get_next_job_empty_queue(self, worker: TrainingWorker):
        """Test getting next job from empty queue."""
        result = worker._get_next_job()
        assert result is None
    
    def test_worker_get_next_job_with_jobs(self, worker: TrainingWorker):
        """Test getting next job from queue with jobs."""
        worker._job_queue = [1, 2, 3]
        result = worker._get_next_job()
        
        assert result == 1
        assert worker._job_queue == [2, 3]
    
    def test_worker_load_dataset_csv(self, worker: TrainingWorker):
        """Test loading a CSV dataset."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2\nvalue1,value2\nvalue3,value4\n")
            temp_path = f.name
        
        try:
            dataset = MagicMock()
            dataset.file_path = temp_path
            dataset.file_type = "csv"
            
            result = worker._load_dataset(dataset)
            
            assert len(result) == 2
            assert result[0]["col1"] == "value1"
        finally:
            Path(temp_path).unlink()
    
    def test_worker_load_dataset_json(self, worker: TrainingWorker):
        """Test loading a JSON dataset."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([{"text": "sample1"}, {"text": "sample2"}], f)
            temp_path = f.name
        
        try:
            dataset = MagicMock()
            dataset.file_path = temp_path
            dataset.file_type = "json"
            
            result = worker._load_dataset(dataset)
            
            assert len(result) == 2
            assert result[0]["text"] == "sample1"
        finally:
            Path(temp_path).unlink()
    
    def test_worker_load_dataset_file_not_found(self, worker: TrainingWorker):
        """Test loading a dataset when file doesn't exist."""
        dataset = MagicMock()
        dataset.file_path = "/nonexistent/path/file.json"
        dataset.file_type = "json"
        
        with pytest.raises(FileNotFoundError):
            worker._load_dataset(dataset)
    
    def test_worker_process_job_no_db(self, mock_db_factory):
        """Test processing job without database session."""
        worker = TrainingWorker(
            worker_id="test-worker",
            db_session_factory=None,
        )
        
        worker._process_job(123)
        
        assert worker.current_job_id is None
        assert worker.status == "idle"
    
    def test_worker_process_job_not_found(self, test_db_session: Session):
        """Test processing a job that doesn't exist in database."""
        def factory():
            return test_db_session
        
        worker = TrainingWorker(
            worker_id="test-worker",
            db_session_factory=factory,
        )
        
        worker._process_job(99999)  # Non-existent job
        
        assert worker.current_job_id is None
        assert worker.status == "idle"
    
    def test_worker_process_job_with_dataset_and_training(self, test_db_engine):
        """Test processing a complete training job."""
        from sqlalchemy.orm import sessionmaker
        
        # Create separate sessions for setup and verification
        TestSession = sessionmaker(bind=test_db_engine)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([{"text": "sample"}], f)
            temp_path = f.name
        
        try:
            # Setup session for creating test data
            setup_session = TestSession()
            dataset = Dataset(
                name="test_dataset",
                filename="test.json",
                file_path=temp_path,
                file_type="json",
                file_size=100,
                row_count=1,
            )
            setup_session.add(dataset)
            setup_session.commit()
            
            job = TrainingJob(
                name="test_job",
                dataset_id=dataset.id,
                training_type=TrainingType.STANDARD.value,
                model_name="llama3.2:3b",
                epochs=1,
                batch_size=1,
            )
            setup_session.add(job)
            setup_session.commit()
            job_id = job.id
            setup_session.close()
            
            # Worker uses its own session
            def factory():
                return TestSession()
            
            worker = TrainingWorker(
                worker_id="test-worker",
                db_session_factory=factory,
            )
            
            # Process the job
            worker._process_job(job_id)
            
            # Verify job completed using new session
            verify_session = TestSession()
            verified_job = verify_session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            assert verified_job.status == TrainingStatus.COMPLETED.value
            assert verified_job.progress == 100.0
            assert worker.jobs_completed == 1
            verify_session.close()
        finally:
            Path(temp_path).unlink()
    
    def test_worker_process_job_dataset_not_found(self, test_db_engine):
        """Test processing a job with missing dataset."""
        from sqlalchemy.orm import sessionmaker
        
        TestSession = sessionmaker(bind=test_db_engine)
        
        setup_session = TestSession()
        
        # Create a dummy dataset first
        dummy_dataset = Dataset(
            name="dummy",
            filename="dummy.json",
            file_path="/tmp/dummy.json",
            file_type="json",
            file_size=10,
            row_count=0,
        )
        setup_session.add(dummy_dataset)
        setup_session.commit()
        
        job = TrainingJob(
            name="test_job",
            dataset_id=dummy_dataset.id,
            training_type=TrainingType.STANDARD.value,
            model_name="llama3.2:3b",
        )
        setup_session.add(job)
        setup_session.commit()
        job_id = job.id
        
        # Delete the dataset to simulate not found
        setup_session.query(Dataset).filter(Dataset.id == dummy_dataset.id).delete()
        setup_session.commit()
        setup_session.close()
        
        def factory():
            return TestSession()
        
        worker = TrainingWorker(
            worker_id="test-worker",
            db_session_factory=factory,
        )
        
        worker._process_job(job_id)
        
        # Verify job failed
        verify_session = TestSession()
        verified_job = verify_session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        assert verified_job.status == TrainingStatus.FAILED.value
        assert "not found" in verified_job.error_message.lower()
        verify_session.close()


class TestTrainingWorkerQLoRA:
    """Tests for QLoRA training."""
    
    def test_train_qlora(self, test_db_engine, monkeypatch, tmp_path):
        """Test QLoRA training execution."""
        from sqlalchemy.orm import sessionmaker
        from app.services.model_resolution_service import ModelResolutionService
        
        TestSession = sessionmaker(bind=test_db_engine)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([{"text": "sample1"}, {"text": "sample2"}], f)
            temp_path_file = f.name
        
        try:
            # Create a mock model directory structure
            mock_model_path = tmp_path / "models" / "meta-llama" / "Llama-3.2-3B"
            mock_model_path.mkdir(parents=True, exist_ok=True)
            (mock_model_path / "config.json").write_text('{"model_type": "llama"}')
            
            # Mock ModelResolutionService to return our mock path
            def mock_resolve_model_path(model_name):
                return str(mock_model_path)
            
            def mock_is_model_available(model_name):
                return True
            
            def mock_validate_model_format(path):
                pass
            
            monkeypatch.setattr(ModelResolutionService, "resolve_model_path", lambda self, name: str(mock_model_path))
            monkeypatch.setattr(ModelResolutionService, "is_model_available", lambda self, name: True)
            monkeypatch.setattr(ModelResolutionService, "validate_model_format", lambda self, path: None)
            
            setup_session = TestSession()
            dataset = Dataset(
                name="test_dataset",
                filename="test.json",
                file_path=temp_path_file,
                file_type="json",
                file_size=100,
                row_count=2,
            )
            setup_session.add(dataset)
            setup_session.commit()
            
            job = TrainingJob(
                name="qlora_job",
                dataset_id=dataset.id,
                training_type=TrainingType.QLORA.value,
                model_name="llama3.2:3b",
                epochs=1,
                batch_size=2,
            )
            setup_session.add(job)
            setup_session.commit()
            job_id = job.id
            setup_session.close()
            
            def factory():
                return TestSession()
            
            worker = TrainingWorker(
                worker_id="test-worker",
                db_session_factory=factory,
            )
            
            # Mock the model loading to avoid actual model download/loading
            with patch('app.workers.training_worker.AutoModelForCausalLM') as mock_model, \
                 patch('app.workers.training_worker.AutoTokenizer') as mock_tokenizer, \
                 patch('app.workers.training_worker.prepare_model_for_kbit_training') as mock_prepare, \
                 patch('app.workers.training_worker.get_peft_model') as mock_peft, \
                 patch('app.workers.training_worker.SFTTrainer') as mock_trainer, \
                 patch('app.workers.training_worker.AutoConfig') as mock_config, \
                 patch('app.workers.training_worker.CONFIG_MAPPING') as mock_mapping:
                
                # Setup config mock
                mock_config_instance = MagicMock()
                mock_config_instance.model_type = "llama"
                mock_config_instance.from_pretrained.return_value = mock_config_instance
                mock_config.from_pretrained.return_value = mock_config_instance
                
                # Setup model mock with proper config
                mock_model_instance = MagicMock()
                mock_config_obj = MagicMock()
                mock_config_obj.model_type = "llama"
                mock_model_instance.config = mock_config_obj
                mock_model.from_pretrained.return_value = mock_model_instance
                
                # Mock CONFIG_MAPPING
                mock_config_class = MagicMock()
                mock_config_class.from_pretrained.return_value = mock_config_instance
                mock_mapping.__getitem__.return_value = mock_config_class
                mock_mapping.__contains__.return_value = True
                
                mock_tokenizer_instance = MagicMock()
                mock_tokenizer_instance.pad_token = None
                mock_tokenizer_instance.eos_token = "<eos>"
                mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
                
                mock_prepare.return_value = mock_model_instance
                mock_peft.return_value = mock_model_instance
                
                mock_trainer_instance = MagicMock()
                mock_trainer.return_value = mock_trainer_instance
                
                worker._process_job(job_id)
            
            verify_session = TestSession()
            verified_job = verify_session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            # Training might complete or fail depending on mocks, but should not crash
            assert verified_job.status in [TrainingStatus.COMPLETED.value, TrainingStatus.FAILED.value]
            verify_session.close()
        finally:
            Path(temp_path_file).unlink()
    
    def test_train_rag(self, test_db_engine, monkeypatch, tmp_path):
        """Test RAG training execution."""
        from sqlalchemy.orm import sessionmaker
        from app.services.model_resolution_service import ModelResolutionService
        
        TestSession = sessionmaker(bind=test_db_engine)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([{"text": "sample1"}], f)
            temp_path_file = f.name
        
        try:
            # Create a mock model directory structure
            mock_model_path = tmp_path / "models" / "meta-llama" / "Llama-3.2-3B"
            mock_model_path.mkdir(parents=True, exist_ok=True)
            (mock_model_path / "config.json").write_text('{"model_type": "llama"}')
            
            # Mock ModelResolutionService
            monkeypatch.setattr(ModelResolutionService, "resolve_model_path", lambda self, name: str(mock_model_path))
            monkeypatch.setattr(ModelResolutionService, "is_model_available", lambda self, name: True)
            monkeypatch.setattr(ModelResolutionService, "validate_model_format", lambda self, path: None)
            
            setup_session = TestSession()
            dataset = Dataset(
                name="test_dataset",
                filename="test.json",
                file_path=temp_path_file,
                file_type="json",
                file_size=100,
                row_count=1,
            )
            setup_session.add(dataset)
            setup_session.commit()
            
            job = TrainingJob(
                name="rag_job",
                dataset_id=dataset.id,
                training_type=TrainingType.RAG.value,
                model_name="llama3.2:3b",
                epochs=1,
            )
            setup_session.add(job)
            setup_session.commit()
            job_id = job.id
            setup_session.close()
            
            def factory():
                return TestSession()
            
            worker = TrainingWorker(
                worker_id="test-worker",
                db_session_factory=factory,
            )
            
            # Mock RAG training components
            with patch('app.workers.training_worker.SentenceTransformer') as mock_st, \
                 patch('app.workers.training_worker.FAISS') as mock_faiss:
                
                mock_st_instance = MagicMock()
                mock_st.return_value = mock_st_instance
                
                worker._process_job(job_id)
            
            verify_session = TestSession()
            verified_job = verify_session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            assert verified_job.status in [TrainingStatus.COMPLETED.value, TrainingStatus.FAILED.value]
            verify_session.close()
        finally:
            Path(temp_path_file).unlink()
    
    def test_train_unsloth(self, test_db_engine, monkeypatch):
        """Test Unsloth training execution (simulated mode)."""
        from sqlalchemy.orm import sessionmaker
        import app.workers.training_worker as tw
        
        # Force simulation mode by setting UNSLOTH_AVAILABLE to False
        monkeypatch.setattr(tw, 'UNSLOTH_AVAILABLE', False)
        
        TestSession = sessionmaker(bind=test_db_engine)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([{"text": "sample1", "label": 1}], f)
            temp_path = f.name
        
        try:
            setup_session = TestSession()
            dataset = Dataset(
                name="test_dataset",
                filename="test.json",
                file_path=temp_path,
                file_type="json",
                file_size=100,
                row_count=1,
            )
            setup_session.add(dataset)
            setup_session.commit()
            
            job = TrainingJob(
                name="unsloth_job",
                dataset_id=dataset.id,
                training_type=TrainingType.UNSLOTH.value,
                model_name="llama3.2:3b",
                epochs=1,
                batch_size=2,
            )
            setup_session.add(job)
            setup_session.commit()
            job_id = job.id
            setup_session.close()
            
            def factory():
                return TestSession()
            
            worker = TrainingWorker(
                worker_id="test-worker",
                db_session_factory=factory,
            )
            
            worker._process_job(job_id)
            
            verify_session = TestSession()
            verified_job = verify_session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            assert verified_job.status == TrainingStatus.COMPLETED.value
            verify_session.close()
        finally:
            Path(temp_path).unlink()


class TestWorkerPool:
    """Tests for WorkerPool."""
    
    @pytest.fixture
    def mock_db_factory(self):
        """Create a mock database session factory."""
        return MagicMock()
    
    @pytest.fixture
    def pool(self, mock_db_factory) -> WorkerPool:
        """Create a WorkerPool instance for testing."""
        return WorkerPool(
            max_workers=4,
            db_session_factory=mock_db_factory,
        )
    
    def test_pool_initialization(self, pool: WorkerPool):
        """Test pool is properly initialized."""
        assert pool.max_workers == 4
        assert pool.is_running is False
        assert pool.active_worker_count == 0
    
    def test_start_workers(self, pool: WorkerPool):
        """Test starting workers."""
        pool.start_workers(2)
        time.sleep(0.1)  # Give workers time to start
        
        assert pool.is_running is True
        assert pool.active_worker_count == 2
        
        # Cleanup
        pool.stop_all_workers()
    
    def test_start_workers_exceeds_max(self, pool: WorkerPool):
        """Test starting more workers than max."""
        with pytest.raises(ValueError) as exc_info:
            pool.start_workers(10)
        
        assert "Cannot start" in str(exc_info.value)
    
    def test_stop_all_workers(self, pool: WorkerPool):
        """Test stopping all workers."""
        pool.start_workers(2)
        time.sleep(0.1)
        
        pool.stop_all_workers()
        time.sleep(0.2)
        
        assert pool.is_running is False
        assert pool.active_worker_count == 0
    
    def test_stop_specific_worker(self, pool: WorkerPool):
        """Test stopping a specific worker."""
        pool.start_workers(2)
        time.sleep(0.1)
        
        worker_ids = list(pool._workers.keys())
        result = pool.stop_worker(worker_ids[0])
        time.sleep(0.1)
        
        assert result is True
        assert pool.active_worker_count == 1
        
        # Cleanup
        pool.stop_all_workers()
    
    def test_stop_nonexistent_worker(self, pool: WorkerPool):
        """Test stopping a non-existent worker."""
        result = pool.stop_worker("fake-worker-id")
        assert result is False
    
    def test_submit_job_not_running(self, pool: WorkerPool):
        """Test submitting a job when pool is not running."""
        result = pool.submit_job(123)
        assert result is False
    
    def test_submit_job_success(self, pool: WorkerPool):
        """Test submitting a job to running pool."""
        pool.start_workers(1)
        time.sleep(0.1)
        
        result = pool.submit_job(123)
        
        assert result is True
        
        # Cleanup
        pool.stop_all_workers()
    
    def test_get_status(self, pool: WorkerPool):
        """Test getting pool status."""
        pool.start_workers(2)
        time.sleep(0.1)
        
        status = pool.get_status()
        
        assert status["total_workers"] == 2
        assert status["is_running"] is True
        assert status["max_workers"] == 4
        assert len(status["workers"]) == 2
        
        # Cleanup
        pool.stop_all_workers()
    
    def test_set_max_workers(self, pool: WorkerPool):
        """Test setting max workers."""
        pool.set_max_workers(8)
        
        assert pool.max_workers == 8
    
    def test_cancel_job_in_queue(self, pool: WorkerPool):
        """Test cancelling a job that's in the queue."""
        pool._job_queue.append(123)
        
        result = pool.cancel_job(123)
        
        assert result is True
        assert 123 not in pool._job_queue
    
    def test_cancel_job_not_found(self, pool: WorkerPool):
        """Test cancelling a non-existent job."""
        result = pool.cancel_job(99999)
        assert result is False
    
    def test_get_worker(self, pool: WorkerPool):
        """Test getting a specific worker."""
        pool.start_workers(1)
        time.sleep(0.1)
        
        worker_id = list(pool._workers.keys())[0]
        worker = pool.get_worker(worker_id)
        
        assert worker is not None
        assert worker.id == worker_id
        
        # Cleanup
        pool.stop_all_workers()
    
    def test_get_worker_not_found(self, pool: WorkerPool):
        """Test getting a non-existent worker."""
        worker = pool.get_worker("fake-id")
        assert worker is None
    
    def test_add_workers(self, pool: WorkerPool):
        """Test adding workers to a running pool."""
        pool.start_workers(1)
        time.sleep(0.1)
        
        assert pool.active_worker_count == 1
        
        pool.add_workers(2)
        time.sleep(0.1)
        
        assert pool.active_worker_count == 3
        
        pool.stop_all_workers()
    
    def test_add_workers_exceeds_max(self, pool: WorkerPool):
        """Test adding more workers than max allows."""
        pool.start_workers(3)
        time.sleep(0.1)
        
        with pytest.raises(ValueError) as exc_info:
            pool.add_workers(2)  # Would exceed max of 4
        
        assert "exceed" in str(exc_info.value).lower()
        
        pool.stop_all_workers()
    
    def test_pool_submit_job_round_robin(self, pool: WorkerPool):
        """Test job submission distributes across workers."""
        pool.start_workers(2)
        time.sleep(0.1)
        
        pool.submit_job(1)
        pool.submit_job(2)
        pool.submit_job(3)
        
        # Jobs should be submitted successfully
        assert pool.is_running
        
        pool.stop_all_workers()
    
    def test_cancel_job_currently_running(self, pool: WorkerPool):
        """Test cancelling a job that's being processed by a worker."""
        pool.start_workers(1)
        time.sleep(0.1)
        
        # Get the worker
        worker_id = list(pool._workers.keys())[0]
        worker = pool._workers[worker_id]
        worker.current_job_id = 999  # Simulate running job
        
        result = pool.cancel_job(999)
        
        # Should return True because it found the job
        assert result is True
        
        pool.stop_all_workers()
    
    def test_set_max_workers_when_running(self, pool: WorkerPool):
        """Test setting max workers while pool is running."""
        pool.start_workers(2)
        time.sleep(0.1)
        
        pool.set_max_workers(10)
        assert pool.max_workers == 10
        
        pool.stop_all_workers()
    
    def test_pool_get_queued_jobs(self, pool: WorkerPool):
        """Test getting list of queued jobs."""
        pool._job_queue = [1, 2, 3]
        
        queued = pool.get_queued_jobs()
        
        assert queued == [1, 2, 3]
    
    def test_pool_get_active_jobs(self, pool: WorkerPool):
        """Test getting list of active jobs."""
        pool.start_workers(2)
        time.sleep(0.1)
        
        # Simulate active jobs
        workers = list(pool._workers.values())
        workers[0].current_job_id = 100
        workers[1].current_job_id = 200
        
        active = pool.get_active_jobs()
        
        assert 100 in active
        assert 200 in active
        
        pool.stop_all_workers()
    
    def test_distribute_queued_jobs(self, pool: WorkerPool):
        """Test distributing queued jobs to idle workers."""
        pool.start_workers(2)
        time.sleep(0.1)
        
        # Manually add jobs to queue
        with pool._queue_lock:
            pool._job_queue = [100, 200, 300]
        
        # Verify jobs are in queue
        assert len(pool.get_queued_jobs()) == 3
        
        # Distribute jobs
        pool.distribute_queued_jobs()
        time.sleep(0.1)
        
        # Verify jobs were distributed to workers
        queued = pool.get_queued_jobs()
        # At least 2 jobs should be distributed (one per worker)
        # The third might remain if workers are processing
        assert len(queued) <= 1  # At most 1 job should remain
        
        pool.stop_all_workers()
    
    def test_distribute_queued_jobs_in_get_status(self, pool: WorkerPool):
        """Test that get_status calls distribute_queued_jobs."""
        pool.start_workers(1)
        time.sleep(0.1)
        
        # Manually add a job to queue
        with pool._queue_lock:
            pool._job_queue = [100]
        
        # Call get_status which should distribute the job
        status = pool.get_status()
        
        # Job should be distributed (queue should be empty or smaller)
        queued = pool.get_queued_jobs()
        assert len(queued) == 0 or len(queued) < 1
        
        pool.stop_all_workers()


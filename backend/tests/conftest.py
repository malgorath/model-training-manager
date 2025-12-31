"""
Pytest configuration and fixtures.

Provides common fixtures for testing including database sessions,
test clients, and mock services.
"""

import os
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from app.core.database import Base, get_db
from app.core.config import Settings, settings
from app.main import create_app


@pytest.fixture(scope="function")
def test_settings() -> Settings:
    """Create test settings with temporary paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Settings(
            database_url="sqlite:///:memory:",
            upload_dir=Path(tmpdir) / "data",
            model_dir=Path(tmpdir) / "data" / "models",
            archive_dir=Path(tmpdir) / "data" / "archives",
            debug=True,
            environment="development",
        )


@pytest.fixture(scope="function")
def test_db_engine():
    """Create a test database engine with in-memory SQLite."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def test_db_session(test_db_engine) -> Generator[Session, None, None]:
    """Create a test database session."""
    TestingSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=test_db_engine,
    )
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture(scope="function")
def client(test_db_session: Session) -> Generator[TestClient, None, None]:
    """Create a test client with overridden database dependency."""
    
    def override_get_db():
        try:
            yield test_db_session
        finally:
            pass
    
    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    app.dependency_overrides.clear()


@pytest.fixture
def sample_csv_content() -> bytes:
    """Sample CSV content for testing."""
    return b"input,output\nHello,World\nTest,Data\nFoo,Bar\n"


@pytest.fixture
def sample_json_content() -> bytes:
    """Sample JSON content for testing."""
    return b'[{"input": "Hello", "output": "World"}, {"input": "Test", "output": "Data"}]'


@pytest.fixture
def temp_upload_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory (replaces uploads)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture
def sample_csv_file(temp_upload_dir: Path, sample_csv_content: bytes) -> Path:
    """Create a sample CSV file for testing."""
    file_path = temp_upload_dir / "test_data.csv"
    file_path.write_bytes(sample_csv_content)
    return file_path


@pytest.fixture
def sample_json_file(temp_upload_dir: Path, sample_json_content: bytes) -> Path:
    """Create a sample JSON file for testing."""
    file_path = temp_upload_dir / "test_data.json"
    file_path.write_bytes(sample_json_content)
    return file_path


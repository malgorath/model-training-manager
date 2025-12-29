"""
Database configuration and session management.

Provides SQLAlchemy engine, session factory, and base class for models.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase
from sqlalchemy.pool import StaticPool

from app.core.config import settings


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


# Create engine with SQLite-specific settings
# Using check_same_thread=False for SQLite to work with multiple threads
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
    echo=settings.debug,
)


# Enable foreign key constraints for SQLite
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable foreign key constraints for SQLite."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def get_db() -> Session:
    """
    Get database session.
    
    Yields:
        Session: SQLAlchemy database session.
        
    Example:
        >>> db = next(get_db())
        >>> try:
        ...     # use db
        ... finally:
        ...     db.close()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def migrate_db() -> None:
    """
    Run database migrations to add new columns to existing tables.
    
    This handles schema changes without dropping existing data.
    """
    from sqlalchemy import inspect, text
    
    inspector = inspect(engine)
    
    # Check if training_jobs table exists
    if "training_jobs" in inspector.get_table_names():
        columns = [col["name"] for col in inspector.get_columns("training_jobs")]
        
        # Add log column if it doesn't exist
        if "log" not in columns:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE training_jobs ADD COLUMN log TEXT"))
        
        # Add model_path column if it doesn't exist
        if "model_path" not in columns:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE training_jobs ADD COLUMN model_path VARCHAR(500)"))
    
    # Check if training_config table exists
    if "training_config" in inspector.get_table_names():
        columns = [col["name"] for col in inspector.get_columns("training_config")]
        
        # Add auto_start_workers column if it doesn't exist
        if "auto_start_workers" not in columns:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE training_config ADD COLUMN auto_start_workers BOOLEAN DEFAULT 0"))
        
        # Add model_provider column if it doesn't exist
        if "model_provider" not in columns:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE training_config ADD COLUMN model_provider VARCHAR(20) DEFAULT 'ollama'"))
        
        # Add model_api_url column if it doesn't exist
        if "model_api_url" not in columns:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE training_config ADD COLUMN model_api_url VARCHAR(500) DEFAULT 'http://localhost:11434'"))
        
        # Add output_directory_base column if it doesn't exist
        if "output_directory_base" not in columns:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE training_config ADD COLUMN output_directory_base VARCHAR(512)"))
        
        # Add model_cache_path column if it doesn't exist
        if "model_cache_path" not in columns:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE training_config ADD COLUMN model_cache_path VARCHAR(512)"))


def init_db() -> None:
    """
    Initialize the database by creating all tables.
    
    This should be called on application startup.
    """
    Base.metadata.create_all(bind=engine)
    migrate_db()  # Run migrations after creating tables


def drop_db() -> None:
    """
    Drop all tables from the database.
    
    WARNING: This will delete all data. Use only for testing.
    """
    Base.metadata.drop_all(bind=engine)


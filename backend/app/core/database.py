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
    
    # Check if projects table exists
    if "projects" in inspector.get_table_names():
        columns = [col["name"] for col in inspector.get_columns("projects")]
        
        # Add model_type column if it doesn't exist
        if "model_type" not in columns:
            try:
                with engine.begin() as conn:
                    conn.execute(text("ALTER TABLE projects ADD COLUMN model_type VARCHAR(50)"))
            except Exception:
                pass  # Column might already exist
    
    # Check if projects table exists
    if "projects" in inspector.get_table_names():
        columns = [col["name"] for col in inspector.get_columns("projects")]
        
        # Make max_rows nullable if it exists and is not nullable
        if "max_rows" in columns:
            # Check if column is nullable by checking the column info
            max_rows_col = next(col for col in inspector.get_columns("projects") if col["name"] == "max_rows")
            if not max_rows_col.get("nullable", False):
                # SQLite doesn't support ALTER COLUMN, so we need to recreate the table
                try:
                    with engine.begin() as conn:
                        # SQLite workaround: create a new table, copy data, drop old, rename new
                        conn.execute(text("""
                            CREATE TABLE projects_new (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                name VARCHAR(255) NOT NULL,
                                description TEXT,
                                base_model VARCHAR(255) NOT NULL,
                                model_type VARCHAR(50),
                                training_type VARCHAR(20) NOT NULL DEFAULT 'qlora',
                                max_rows INTEGER,
                                output_directory VARCHAR(512) NOT NULL,
                                status VARCHAR(20) NOT NULL DEFAULT 'pending',
                                progress FLOAT NOT NULL DEFAULT 0.0,
                                current_epoch INTEGER NOT NULL DEFAULT 0,
                                current_loss FLOAT,
                                error_message TEXT,
                                log TEXT,
                                model_path VARCHAR(500),
                                worker_id VARCHAR(50),
                                started_at DATETIME,
                                completed_at DATETIME,
                                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                            )
                        """))
                        # Check if model_type column exists in old table
                        old_columns = [col["name"] for col in inspector.get_columns("projects")]
                        has_model_type = "model_type" in old_columns
                        
                        if has_model_type:
                            conn.execute(text("""
                                INSERT INTO projects_new 
                                (id, name, description, base_model, model_type, training_type, max_rows, output_directory, 
                                 status, progress, current_epoch, current_loss, error_message, log, model_path, 
                                 worker_id, started_at, completed_at, created_at, updated_at)
                                SELECT 
                                    id, name, description, base_model, model_type, training_type, max_rows, output_directory,
                                    status, progress, current_epoch, current_loss, error_message, log, model_path,
                                    worker_id, started_at, completed_at, created_at, updated_at
                                FROM projects
                            """))
                        else:
                            conn.execute(text("""
                                INSERT INTO projects_new 
                                (id, name, description, base_model, training_type, max_rows, output_directory, 
                                 status, progress, current_epoch, current_loss, error_message, log, model_path, 
                                 worker_id, started_at, completed_at, created_at, updated_at)
                                SELECT 
                                    id, name, description, base_model, training_type, max_rows, output_directory,
                                    status, progress, current_epoch, current_loss, error_message, log, model_path,
                                    worker_id, started_at, completed_at, created_at, updated_at
                                FROM projects
                            """))
                        conn.execute(text("DROP TABLE projects"))
                        conn.execute(text("ALTER TABLE projects_new RENAME TO projects"))
                        # Recreate indexes
                        conn.execute(text("CREATE INDEX ix_projects_name ON projects (name)"))
                        conn.execute(text("CREATE INDEX ix_projects_base_model ON projects (base_model)"))
                        conn.execute(text("CREATE INDEX ix_projects_status ON projects (status)"))
                        conn.execute(text("CREATE INDEX ix_projects_worker_id ON projects (worker_id)"))
                except Exception as e:
                    # If migration fails, log but don't crash - the app will work with existing schema
                    import logging
                    logging.getLogger(__name__).warning(f"Could not migrate max_rows to nullable: {e}")


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


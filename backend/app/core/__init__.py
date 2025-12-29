"""
Core module containing configuration and database setup.
"""

from app.core.config import settings
from app.core.database import get_db, engine, Base

__all__ = ["settings", "get_db", "engine", "Base"]


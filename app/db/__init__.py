"""
Database module for handling database connections and operations.
Supports multiple database backends with configuration flexibility.

Default: SQLite for development.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

# Base class for ORM models
Base = declarative_base()

# Default database URL (SQLite for development)
default_db_url = "sqlite:///./development.db"

def get_database_url():
    """Retrieve the database URL from environment variables or use the default."""
    return os.getenv("DATABASE_URL", default_db_url)

# Create the database engine
db_url = get_database_url()
engine = create_engine(db_url, connect_args={"check_same_thread": False} if "sqlite" in db_url else {})

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initialize the database by creating all tables."""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Dependency to get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

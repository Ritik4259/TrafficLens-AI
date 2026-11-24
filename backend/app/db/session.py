"""
Database engine, session factory, and lifecycle management for TrafficLens-AI.

Supports SQLite for local development (enforcing PRAGMA foreign_keys=ON)
and full connection string and dialect compatibility with PostgreSQL.
"""

from __future__ import annotations
import sqlite3
from typing import Generator, Optional
from sqlalchemy import create_engine, event, Engine
from sqlalchemy.orm import sessionmaker, Session

from backend.app.core.config import get_settings
from backend.app.models.base import Base


def create_db_engine(database_url: Optional[str] = None, echo: bool = False) -> Engine:
    """Creates and configures an SQLAlchemy engine with dialect-specific options."""
    settings = get_settings()
    url = database_url or settings.DATABASE_URL
    is_sqlite = url.startswith("sqlite")

    connect_args = {}
    if is_sqlite:
        connect_args["check_same_thread"] = False

    engine = create_engine(
        url,
        echo=echo or settings.DB_ECHO,
        connect_args=connect_args,
        future=True,
    )

    # Enable SQLite foreign key enforcement
    if is_sqlite:
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            if isinstance(dbapi_connection, sqlite3.Connection):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

    return engine


# Default application engine and session factory
engine = create_db_engine()
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    class_=Session,
)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency yielding a request-scoped database session.
    Commits on successful completion, rolls back on uncaught exception, and always closes.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db(target_engine: Optional[Engine] = None) -> None:
    """Initializes tables in the target database."""
    eng = target_engine or engine
    Base.metadata.create_all(bind=eng)

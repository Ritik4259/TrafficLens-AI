"""
SQLAlchemy 2.0 Base class and custom type decorators.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Optional
from sqlalchemy import DateTime, TypeDecorator
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy 2.0 declarative models in TrafficLens-AI."""
    pass


class UTCDateTime(TypeDecorator):
    """
    Timezone-aware UTC DateTime type decorator.
    
    Ensures datetimes are always stored as timezone-aware UTC timestamps
    and converted to UTC datetime objects upon retrieval across both SQLite
    and PostgreSQL dialects.
    """
    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: Optional[datetime], dialect: Any) -> Optional[datetime]:
        if value is not None:
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        return None

    def process_result_value(self, value: Optional[datetime], dialect: Any) -> Optional[datetime]:
        if value is not None:
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        return None

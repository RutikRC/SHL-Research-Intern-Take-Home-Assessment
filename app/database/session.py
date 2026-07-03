"""
SQLAlchemy 2.0 database setup: engine, session factory, and dependency injection.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy import NullPool
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings

settings = get_settings()


# ── Engine ──────────────────────────────────────────────────────────────────
# Use the async URL built from individual Postgres credentials.
# pool_pre_ping checks connection health before each use (handles Neon idle timeouts).
# pool_recycle recycles connections after 300 seconds to avoid stale connections.
engine = create_async_engine(
    url=settings.async_database_url,
    echo=settings.DATABASE_ECHO,
    pool_pre_ping=True,
    pool_recycle=300,
    poolclass=NullPool if settings.TESTING else None,
    pool_size=settings.DATABASE_POOL_SIZE if not settings.TESTING else None,
    max_overflow=settings.DATABASE_MAX_OVERFLOW if not settings.TESTING else None,
)

# ── Session factory ─────────────────────────────────────────────────────────
async_session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, Any]:
    """FastAPI dependency that yields an async database session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ── Declarative base ────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all ORM models."""

"""
Repository for accessing and persisting SHL catalog data in PostgreSQL.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Assessment


class CatalogRepository:
    """Data-access layer for the ``assessments`` table."""

    async def upsert_assessment(
        self,
        session: AsyncSession,
        record: dict,
    ) -> bool:
        """Insert or update a single assessment row.

        Uses PostgreSQL ``ON CONFLICT`` on ``entity_id``.
        Returns ``True`` if the row was inserted, ``False`` if updated.
        """
        stmt = pg_insert(Assessment).values(
            entity_id=record["entity_id"],
            name=record["name"],
            url=record["url"],
            description=record.get("description", ""),
            duration=record.get("duration"),
            remote=record.get("remote", False),
            adaptive=record.get("adaptive", False),
            job_levels=record.get("job_levels", []),
            languages=record.get("languages", []),
            keys=record.get("keys", []),
            scraped_at=record.get("scraped_at"),
        )
        excluded = {
            "name": stmt.excluded.name,
            "url": stmt.excluded.url,
            "description": stmt.excluded.description,
            "duration": stmt.excluded.duration,
            "remote": stmt.excluded.remote,
            "adaptive": stmt.excluded.adaptive,
            "job_levels": stmt.excluded.job_levels,
            "languages": stmt.excluded.languages,
            "keys": stmt.excluded["keys"],
            "scraped_at": stmt.excluded.scraped_at,
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=["entity_id"],
            set_=excluded,
        )
        result = await session.execute(stmt)
        return result.rowcount == 1

    async def count(self, session: AsyncSession) -> int:
        """Return the total number of assessments in the database."""
        stmt = select(func.count(Assessment.id))
        result = await session.execute(stmt)
        return result.scalar_one()

    async def get_by_entity_id(
        self,
        session: AsyncSession,
        entity_id: str,
    ) -> Assessment | None:
        """Fetch a single assessment by its entity_id."""
        stmt = select(Assessment).where(Assessment.entity_id == entity_id)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_all(
        self,
        session: AsyncSession,
        offset: int = 0,
        limit: int = 100,
    ) -> list[Assessment]:
        """Return a paginated list of all assessments."""
        stmt = (
            select(Assessment)
            .order_by(Assessment.name)
            .offset(offset)
            .limit(limit)
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

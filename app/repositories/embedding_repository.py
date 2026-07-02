"""
Repository for vector embedding storage and retrieval using pgvector.

Provides CRUD operations for the ``assessment_embeddings`` table
and cosine similarity search via pgvector.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Assessment, AssessmentEmbedding
from app.core.logging_ import get_logger

logger = get_logger(__name__)


class EmbeddingRepository:
    """Data-access layer for the ``assessment_embeddings`` table."""

    async def store_embedding(
        self,
        session: AsyncSession,
        assessment_id: uuid.UUID,
        embedding: list[float],
        model: str,
        dimension: int,
    ) -> AssessmentEmbedding:
        """Insert a new embedding record for an assessment.

        Args:
            session: An active async database session.
            assessment_id: The UUID of the assessment.
            embedding: The embedding vector as a list of floats.
            model: The embedding model name used to generate the vector.
            dimension: The dimensionality of the embedding vector.

        Returns:
            The newly created AssessmentEmbedding ORM object.
        """
        record = AssessmentEmbedding(
            assessment_id=assessment_id,
            embedding=embedding,
            embedding_model=model,
            embedding_dimension=dimension,
        )
        session.add(record)
        await session.flush()
        logger.debug(
            "embedding_stored",
            assessment_id=str(assessment_id),
            model=model,
            dimension=dimension,
        )
        return record

    async def get_by_assessment_id(
        self,
        session: AsyncSession,
        assessment_id: uuid.UUID,
    ) -> AssessmentEmbedding | None:
        """Fetch the embedding for a specific assessment.

        Args:
            session: An active async database session.
            assessment_id: The UUID of the assessment.

        Returns:
            The AssessmentEmbedding if found, else None.
        """
        stmt = select(AssessmentEmbedding).where(
            AssessmentEmbedding.assessment_id == assessment_id,
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def embedding_exists(
        self,
        session: AsyncSession,
        assessment_id: uuid.UUID,
    ) -> bool:
        """Check whether an embedding already exists for an assessment.

        Args:
            session: An active async database session.
            assessment_id: The UUID of the assessment.

        Returns:
            True if an embedding exists, False otherwise.
        """
        stmt = select(
            select(AssessmentEmbedding)
            .where(AssessmentEmbedding.assessment_id == assessment_id)
            .exists(),
        )
        result = await session.execute(stmt)
        return result.scalar_one()

    async def count_embeddings(self, session: AsyncSession) -> int:
        """Return the total number of embedding records.

        Args:
            session: An active async database session.

        Returns:
            The count of embedding records.
        """
        stmt = select(func.count(AssessmentEmbedding.id))
        result = await session.execute(stmt)
        return result.scalar_one()

    async def delete_all_embeddings(self, session: AsyncSession) -> int:
        """Delete all embedding records.

        Args:
            session: An active async database session.

        Returns:
            The number of deleted records.
        """
        stmt = select(AssessmentEmbedding)
        result = await session.execute(stmt)
        records = list(result.scalars().all())
        for record in records:
            await session.delete(record)
        await session.flush()
        count = len(records)
        logger.info("embeddings_deleted_all", count=count)
        return count

    async def delete_by_assessment_id(
        self,
        session: AsyncSession,
        assessment_id: uuid.UUID,
    ) -> bool:
        """Delete the embedding for a specific assessment.

        Args:
            session: An active async database session.
            assessment_id: The UUID of the assessment.

        Returns:
            True if a record was deleted, False otherwise.
        """
        record = await self.get_by_assessment_id(session, assessment_id)
        if record is None:
            return False
        await session.delete(record)
        await session.flush()
        return True

    async def get_all_assessment_ids_without_embeddings(
        self,
        session: AsyncSession,
    ) -> list[uuid.UUID]:
        """Return assessment IDs that do not yet have an embedding.

        Args:
            session: An active async database session.

        Returns:
            A list of assessment UUIDs missing embeddings.
        """
        subquery = select(AssessmentEmbedding.assessment_id).subquery()
        stmt = (
            select(Assessment.id)
            .where(Assessment.id.notin_(subquery))
            .order_by(Assessment.name)
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

    async def get_all_assessment_ids(
        self,
        session: AsyncSession,
    ) -> list[uuid.UUID]:
        """Return all assessment IDs ordered by name.

        Args:
            session: An active async database session.

        Returns:
            A list of all assessment UUIDs.
        """
        stmt = select(Assessment.id).order_by(Assessment.name)
        result = await session.execute(stmt)
        return list(result.scalars().all())

    async def get_assessment_by_id(
        self,
        session: AsyncSession,
        assessment_id: uuid.UUID,
    ) -> Assessment | None:
        """Fetch a single assessment by its primary key.

        Args:
            session: An active async database session.
            assessment_id: The UUID of the assessment.

        Returns:
            The Assessment if found, else None.
        """
        stmt = select(Assessment).where(Assessment.id == assessment_id)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def search_similar(
        self,
        session: AsyncSession,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[Assessment]:
        """Find assessments whose embeddings are most similar to the query vector.

        Uses pgvector's cosine distance (``<=>`` operator) for similarity search.
        Runs as a raw SQL query to properly handle the pgvector VECTOR type.

        Args:
            session: An active async database session.
            query_embedding: The query vector as a list of floats.
            top_k: Maximum number of results to return.

        Returns:
            A list of Assessment objects ordered by cosine similarity (closest first).
        """
        if not query_embedding:
            return []

        # Build the pgvector-compatible string representation
        vector_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

        sql = text(
            """
            SELECT a.*
            FROM assessments a
            INNER JOIN assessment_embeddings ae ON a.id = ae.assessment_id
            ORDER BY ae.embedding <=> CAST(:query_vec AS vector)
            LIMIT :top_k
            """
        )
        result = await session.execute(
            sql,
            {"query_vec": vector_str, "top_k": top_k},
        )
        rows = result.mappings().all()

        assessments: list[Assessment] = []
        for row in rows:
            assessment = Assessment(**row)
            assessments.append(assessment)

        logger.debug(
            "search_similar_completed",
            top_k=top_k,
            results=len(assessments),
        )
        return assessments

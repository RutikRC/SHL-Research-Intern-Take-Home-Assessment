"""
Orchestrates the embedding generation pipeline.

Reads assessments from the database, generates semantic documents,
produces vectors via Ollama, and stores them in PostgreSQL with pgvector.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging_ import get_logger
from app.database.models import Assessment
from app.repositories.embedding_repository import EmbeddingRepository
from app.services.embedding_client import EmbeddingClient
from app.services.embedding_document_formatter import EmbeddingDocumentFormatter

logger = get_logger(__name__)


class EmbeddingService:
    """Business logic for generating and managing assessment embeddings.

    The pipeline is:
        Assessment → Semantic Document → Embedding Vector → PostgreSQL (pgvector)

    The service is idempotent: running ``generate_all_embeddings`` multiple
    times will skip assessments that already have embeddings.
    """

    def __init__(
        self,
        repository: EmbeddingRepository | None = None,
        client: EmbeddingClient | None = None,
        formatter: EmbeddingDocumentFormatter | None = None,
    ) -> None:
        self._repository = repository or EmbeddingRepository()
        self._client = client or EmbeddingClient()
        self._formatter = formatter or EmbeddingDocumentFormatter()
        self._settings = get_settings()

    # ── Public API ───────────────────────────────────────────────────────────

    async def generate_for_assessment(
        self,
        session: AsyncSession,
        assessment_id: uuid.UUID,
    ) -> bool:
        """Generate and store an embedding for a single assessment.

        Skips if an embedding already exists.

        Args:
            session: An active async database session.
            assessment_id: The UUID of the assessment to embed.

        Returns:
            True if a new embedding was generated, False if skipped.
        """
        # Check if embedding already exists
        exists = await self._repository.embedding_exists(session, assessment_id)
        if exists:
            logger.debug(
                "embedding_skipped_exists",
                assessment_id=str(assessment_id),
            )
            return False

        # Fetch the assessment
        assessment = await self._repository.get_assessment_by_id(
            session, assessment_id,
        )
        if assessment is None:
            logger.warning(
                "embedding_assessment_not_found",
                assessment_id=str(assessment_id),
            )
            return False

        # Generate semantic document
        document = self._formatter.format(assessment)

        # Generate embedding vector
        try:
            vector = await self._client.embed(document)
        except (RuntimeError, ValueError) as exc:
            logger.error(
                "embedding_generation_failed",
                assessment_id=str(assessment_id),
                assessment_name=assessment.name,
                error=str(exc),
            )
            return False

        # Validate dimension
        expected_dim = self._settings.EMBEDDING_DIMENSION
        if len(vector) != expected_dim:
            logger.error(
                "embedding_dimension_mismatch",
                assessment_id=str(assessment_id),
                assessment_name=assessment.name,
                expected=expected_dim,
                actual=len(vector),
            )
            return False

        # Store in database
        await self._repository.store_embedding(
            session=session,
            assessment_id=assessment_id,
            embedding=vector,
            model=self._client.model,
            dimension=len(vector),
        )

        logger.info(
            "embedding_generated",
            assessment_id=str(assessment_id),
            assessment_name=assessment.name,
            dimension=len(vector),
            model=self._client.model,
        )
        return True

    async def generate_all_embeddings(
        self,
        session: AsyncSession,
    ) -> dict[str, Any]:
        """Generate embeddings for all assessments that don't have one yet.

        Args:
            session: An active async database session.

        Returns:
            A summary dict with ``success``, ``processed``, ``generated``,
            ``skipped``, and ``failed`` counts.
        """
        start = time.monotonic()
        logger.info("embedding_batch_started")

        # Get all assessment IDs
        all_ids = await self._repository.get_all_assessment_ids(session)
        total = len(all_ids)
        generated = 0
        skipped = 0
        failed = 0

        for idx, assessment_id in enumerate(all_ids, start=1):
            # Check if already exists
            exists = await self._repository.embedding_exists(session, assessment_id)
            if exists:
                skipped += 1
                logger.debug(
                    "embedding_batch_skipped",
                    progress=f"{idx}/{total}",
                    assessment_id=str(assessment_id),
                )
                continue

            # Fetch assessment for logging
            assessment = await self._repository.get_assessment_by_id(
                session, assessment_id,
            )
            name = assessment.name if assessment else "unknown"

            logger.info(
                "embedding_batch_processing",
                progress=f"{idx}/{total}",
                assessment_name=name,
                assessment_id=str(assessment_id),
            )

            try:
                document = self._formatter.format(assessment) if assessment else ""
                if not document:
                    failed += 1
                    logger.warning(
                        "embedding_batch_empty_document",
                        assessment_id=str(assessment_id),
                        assessment_name=name,
                    )
                    continue

                vector = await self._client.embed(document)

                if len(vector) != self._settings.EMBEDDING_DIMENSION:
                    failed += 1
                    logger.error(
                        "embedding_batch_dimension_mismatch",
                        assessment_id=str(assessment_id),
                        assessment_name=name,
                        expected=self._settings.EMBEDDING_DIMENSION,
                        actual=len(vector),
                    )
                    continue

                await self._repository.store_embedding(
                    session=session,
                    assessment_id=assessment_id,
                    embedding=vector,
                    model=self._client.model,
                    dimension=len(vector),
                )
                generated += 1

                logger.info(
                    "embedding_batch_generated",
                    progress=f"{idx}/{total}",
                    assessment_name=name,
                    dimension=len(vector),
                )

            except (RuntimeError, ValueError) as exc:
                failed += 1
                logger.error(
                    "embedding_batch_failed",
                    progress=f"{idx}/{total}",
                    assessment_id=str(assessment_id),
                    assessment_name=name,
                    error=str(exc),
                )

        await session.commit()

        elapsed = round(time.monotonic() - start, 4)
        logger.info(
            "embedding_batch_completed",
            total=total,
            generated=generated,
            skipped=skipped,
            failed=failed,
            elapsed_seconds=elapsed,
        )

        return {
            "success": True,
            "processed": total,
            "generated": generated,
            "skipped": skipped,
            "failed": failed,
        }

    async def refresh_embeddings(
        self,
        session: AsyncSession,
    ) -> dict[str, Any]:
        """Delete all existing embeddings and regenerate them.

        Args:
            session: An active async database session.

        Returns:
            A summary dict with ``success``, ``processed``, ``generated``,
            ``deleted``, and ``failed`` counts.
        """
        start = time.monotonic()
        logger.info("embedding_refresh_started")

        # Delete all existing embeddings
        deleted = await self._repository.delete_all_embeddings(session)
        await session.flush()

        logger.info("embedding_refresh_deleted", deleted=deleted)

        # Regenerate all embeddings
        result = await self.generate_all_embeddings(session)

        elapsed = round(time.monotonic() - start, 4)
        logger.info(
            "embedding_refresh_completed",
            deleted=deleted,
            generated=result["generated"],
            elapsed_seconds=elapsed,
        )

        return {
            "success": True,
            "processed": result["processed"],
            "generated": result["generated"],
            "deleted": deleted,
            "failed": result["failed"],
        }

    async def count_embeddings(self, session: AsyncSession) -> int:
        """Return the total number of stored embeddings.

        Args:
            session: An active async database session.

        Returns:
            The count of embedding records.
        """
        return await self._repository.count_embeddings(session)

    async def embedding_exists(
        self,
        session: AsyncSession,
        assessment_id: uuid.UUID,
    ) -> bool:
        """Check whether an embedding exists for a given assessment.

        Args:
            session: An active async database session.
            assessment_id: The UUID of the assessment.

        Returns:
            True if an embedding exists, False otherwise.
        """
        return await self._repository.embedding_exists(session, assessment_id)
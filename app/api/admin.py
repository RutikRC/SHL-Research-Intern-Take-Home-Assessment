"""
Admin-only endpoints for catalog management and embedding generation.

These are temporary endpoints used during development to trigger ingestion,
generate embeddings, and inspect system state.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_catalog_service, get_db_dep, get_embedding_service
from app.services.catalog_service import CatalogService
from app.services.embedding_service import EmbeddingService
from app.core.logging_ import get_logger

router = APIRouter(prefix="/admin", tags=["admin"])
logger = get_logger(__name__)


# ── Catalog endpoints ────────────────────────────────────────────────────────


@router.post(
    "/catalog/import",
    summary="Import the SHL catalog",
    description="Download the SHL catalog JSON from URL, normalize it, and upsert all records into PostgreSQL.",
)
async def import_catalog(
    catalog_service: CatalogService = Depends(get_catalog_service),
    session: AsyncSession = Depends(get_db_dep),
) -> dict:
    """Trigger a full catalog import from the SHL API endpoint.

    Returns:
        A summary dict with ``success``, ``total``, ``inserted``, and ``updated`` counts.
    """
    result = await catalog_service.import_catalog(session)
    return result


@router.post(
    "/catalog/import-local",
    summary="Import the SHL catalog from local JSON file",
    description="Read the catalog JSON from the local data/catalogue.json file, normalize it, and upsert all records into PostgreSQL.",
)
async def import_catalog_local(
    catalog_service: CatalogService = Depends(get_catalog_service),
    session: AsyncSession = Depends(get_db_dep),
) -> dict:
    """Trigger a full catalog import from the local data/catalogue.json file.

    Returns:
        A summary dict with ``success``, ``total``, ``inserted``, and ``updated`` counts.
    """
    result = await catalog_service.import_catalog_local(session)
    return result


@router.get(
    "/catalog/count",
    summary="Count assessments in the database",
    description="Return the total number of assessment records currently stored.",
)
async def count_assessments(
    catalog_service: CatalogService = Depends(get_catalog_service),
    session: AsyncSession = Depends(get_db_dep),
) -> dict:
    """Return the total number of assessments in the database.

    Returns:
        A dict with a single ``count`` key.
    """
    count = await catalog_service.count_assessments(session)
    return {"count": count}


# ── Embedding endpoints ─────────────────────────────────────────────────────


@router.post(
    "/embeddings/generate",
    summary="Generate embeddings for all assessments",
    description=(
        "Reads all assessments from the database, generates a semantic document "
        "for each, creates an embedding vector via Ollama (nomic-embed-text), "
        "and stores it in the assessment_embeddings table. "
        "Assessments that already have an embedding are skipped."
    ),
)
async def generate_embeddings(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    session: AsyncSession = Depends(get_db_dep),
) -> dict:
    """Trigger embedding generation for all assessments.

    Returns:
        A summary dict with ``success``, ``processed``, ``generated``,
        ``skipped``, and ``failed`` counts.
    """
    result = await embedding_service.generate_all_embeddings(session)
    return result


@router.post(
    "/embeddings/refresh",
    summary="Delete and regenerate all embeddings",
    description=(
        "Deletes all existing embedding records, then regenerates embeddings "
        "for every assessment in the database. Use carefully on large datasets."
    ),
)
async def refresh_embeddings(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    session: AsyncSession = Depends(get_db_dep),
) -> dict:
    """Delete all embeddings and regenerate them.

    Returns:
        A summary dict with ``success``, ``processed``, ``generated``,
        ``deleted``, and ``failed`` counts.
    """
    result = await embedding_service.refresh_embeddings(session)
    return result


@router.get(
    "/embeddings/count",
    summary="Count stored embeddings",
    description="Return the total number of embedding records in the assessment_embeddings table.",
)
async def count_embeddings(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    session: AsyncSession = Depends(get_db_dep),
) -> dict:
    """Return the total number of stored embeddings.

    Returns:
        A dict with a single ``count`` key.
    """
    count = await embedding_service.count_embeddings(session)
    return {"count": count}

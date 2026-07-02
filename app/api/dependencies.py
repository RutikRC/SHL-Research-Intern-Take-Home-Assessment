"""
FastAPI dependency injection container.

Provides ready-to-use instances of services, repositories, and infrastructure.
"""

from __future__ import annotations

from typing import Any

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings, Settings
from app.database.session import get_db
from app.repositories.catalog_repository import CatalogRepository
from app.repositories.embedding_repository import EmbeddingRepository
from app.services.catalog_service import CatalogService
from app.services.chat_service import ChatService
from app.services.comparison_service import ComparisonService
from app.services.prompt_service import PromptService
from app.services.recommendation_mapper import RecommendationMapper
from app.services.retrieval_service import RetrievalService


# ── Settings ────────────────────────────────────────────────────────────────

async def get_settings_dep() -> Settings:
    """Provide the application Settings singleton."""
    return get_settings()


# ── Database ────────────────────────────────────────────────────────────────

async def get_db_dep() -> AsyncSession:
    """Provide an async database session.

    Yields:
        An AsyncSession managed by the session factory.
    """
    async for session in get_db():
        yield session


# ── Repositories ────────────────────────────────────────────────────────────

async def get_catalog_repository() -> CatalogRepository:
    """Provide a CatalogRepository instance."""
    return CatalogRepository()


async def get_embedding_repository() -> EmbeddingRepository:
    """Provide an EmbeddingRepository instance."""
    return EmbeddingRepository()


# ── Services ────────────────────────────────────────────────────────────────

async def get_chat_service() -> ChatService:
    """Provide a ChatService instance."""
    return ChatService()


async def get_catalog_service() -> CatalogService:
    """Provide a CatalogService instance with its repository injected."""
    repository = CatalogRepository()
    return CatalogService(repository=repository)


async def get_embedding_service() -> EmbeddingService:
    """Provide an EmbeddingService instance with its dependencies injected."""
    from app.services.embedding_client import EmbeddingClient
    from app.services.embedding_document_formatter import EmbeddingDocumentFormatter
    from app.services.embedding_service import EmbeddingService

    repository = EmbeddingRepository()
    client = EmbeddingClient()
    formatter = EmbeddingDocumentFormatter()
    return EmbeddingService(repository=repository, client=client, formatter=formatter)


async def get_retrieval_service(
    repository: EmbeddingRepository | None = None,
) -> RetrievalService:
    """Provide a RetrievalService instance with its dependencies injected."""
    from app.services.embedding_client import EmbeddingClient

    repo = repository or EmbeddingRepository()
    client = EmbeddingClient()
    return RetrievalService(repository=repo, embedding_client=client)


async def get_prompt_service() -> PromptService:
    """Provide a PromptService instance."""
    return PromptService()


async def get_comparison_service() -> ComparisonService:
    """Provide a ComparisonService instance."""
    return ComparisonService()


# ── Request-based helpers ───────────────────────────────────────────────────

def get_request_id(request: Request) -> str:
    """Extract or generate a request ID from the incoming HTTP request."""
    return request.headers.get("X-Request-ID", "")

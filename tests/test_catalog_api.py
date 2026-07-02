"""
End-to-end tests for the admin/catalog/ API endpoints.

Database layer is mocked so no real PostgreSQL is needed.
HTTP calls to the SHL API are patched at the CatalogService level.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_catalog_service, get_db_dep
from app.database.models import Assessment
from app.repositories.catalog_repository import CatalogRepository
from app.services.catalog_service import CatalogService


SAMPLE_CATALOG = [
    {
        "entity_id": "4302",
        "name": "Global Skills Development Report",
        "link": "https://www.shl.com/products/product-catalog/view/test/",
        "scraped_at": "2026-06-30T12:00:00Z",
        "job_levels": ["Manager"],
        "languages": ["en"],
        "duration": "30 minutes",
        "remote": "yes",
        "adaptive": "no",
        "description": "A report.",
        "keys": ["skills"],
    },
    {
        "entity_id": "4303",
        "name": "Java Coding Assessment",
        "link": "https://www.shl.com/products/product-catalog/view/java/",
        "scraped_at": None,
        "job_levels": ["Entry Level"],
        "languages": ["en", "fr"],
        "duration": "",
        "remote": "yes",
        "adaptive": "yes",
        "description": "Measures Java skills.",
        "keys": ["java"],
    },
    {
        "entity_id": "4304",
        "name": "Customer Service Simulator",
        "link": "https://www.shl.com/products/product-catalog/view/cs/",
        "scraped_at": "2026-06-29T08:15:00Z",
        "job_levels": None,
        "languages": None,
        "duration": None,
        "remote": "no",
        "adaptive": "no",
        "description": "",
        "keys": None,
    },
]


@pytest.fixture(autouse=True)
def _override_dependencies(app: FastAPI) -> None:
    """Override FastAPI dependencies so no real database is needed."""
    # ── Mock database session ────────────────────────────────────────────────
    mock_session = AsyncMock(spec=AsyncSession)

    async def _get_mock_db():
        yield mock_session

    app.dependency_overrides[get_db_dep] = _get_mock_db  # type: ignore[assignment]

    # ── Mock repository ──────────────────────────────────────────────────────
    mock_repo = AsyncMock(spec=CatalogRepository)

    # Wire upsert_assessment: first 3 calls return True (inserts),
    # subsequent calls return False (updates).
    upsert_results = [True, True, True]

    async def upsert_side_effect(session, record):
        CatalogRepository.upsert_assessment
        if upsert_results:
            return upsert_results.pop(0)
        return False

    mock_repo.upsert_assessment = AsyncMock(side_effect=upsert_side_effect)
    mock_repo.count = AsyncMock(return_value=0)

    # ── Mocked CatalogService ────────────────────────────────────────────────
    mock_service = CatalogService(repository=mock_repo)

    # Patch _fetch_catalog to avoid real HTTP calls.
    async def fake_fetch():
        return SAMPLE_CATALOG

    mock_service._fetch_catalog = fake_fetch  # type: ignore[method-assign]

    async def _get_mock_service():
        return mock_service

    app.dependency_overrides[get_catalog_service] = _get_mock_service  # type: ignore[assignment]

    yield

    # Clean up overrides so other test modules are not affected.
    app.dependency_overrides.clear()


@pytest.fixture
def app() -> FastAPI:
    """Return the real FastAPI application instance (needed for dependency overrides)."""
    # Import here to avoid circular imports at module level.
    from app.main import app as _app
    return _app


class TestAdminCatalogEndpoints:
    """End-to-end tests for admin/catalog/ endpoints with mocked DB + HTTP."""

    @pytest.mark.asyncio
    async def test_import_returns_summary(self, async_client: AsyncClient) -> None:
        """POST /admin/catalog/import should return a summary dict."""
        response = await async_client.post("/admin/catalog/import")
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["total"] == 3
        assert body["inserted"] == 3
        assert body["updated"] == 0

    @pytest.mark.asyncio
    async def test_duplicate_import_shows_updates(self, async_client: AsyncClient) -> None:
        """Second import should have inserted=0, updated=3 (all rows exist)."""
        await async_client.post("/admin/catalog/import")
        response = await async_client.post("/admin/catalog/import")
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["inserted"] == 0
        assert body["updated"] == 3

    @pytest.mark.asyncio
    async def test_count_after_import(self, async_client: AsyncClient) -> None:
        """GET /admin/catalog/count should return 3 after import."""
        # Manually set mock to return 3 after an import
        from app.main import app as _app
        svc_override = _app.dependency_overrides[get_catalog_service]
        svc = await svc_override()
        svc._repository.count = AsyncMock(return_value=3)  # type: ignore[assignment]

        response = await async_client.get("/admin/catalog/count")
        assert response.status_code == 200
        assert response.json() == {"count": 3}

    @pytest.mark.asyncio
    async def test_count_before_import_is_zero(self, async_client: AsyncClient) -> None:
        """GET /admin/catalog/count before any import returns 0."""
        response = await async_client.get("/admin/catalog/count")
        assert response.status_code == 200
        assert response.json() == {"count": 0}


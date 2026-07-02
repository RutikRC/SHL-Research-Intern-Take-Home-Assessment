"""
Tests for the GET /health endpoint.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient


class TestHealthEndpoint:
    """Suite of tests for the health-check endpoint."""

    @pytest.mark.asyncio
    async def test_health_returns_200(self, async_client: AsyncClient) -> None:
        """GET /health should return HTTP 200 and ``{"status": "ok"}``."""
        response = await async_client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_health_has_correct_content_type(self, async_client: AsyncClient) -> None:
        """GET /health should return JSON content type."""
        response = await async_client.get("/health")
        assert response.headers["content-type"] == "application/json"

    @pytest.mark.asyncio
    async def test_health_returns_only_status_key(self, async_client: AsyncClient) -> None:
        """The response body should contain exactly the ``status`` key."""
        response = await async_client.get("/health")
        body = response.json()
        assert set(body.keys()) == {"status"}

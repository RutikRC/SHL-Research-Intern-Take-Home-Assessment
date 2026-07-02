"""
Pytest configuration and shared fixtures.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
def anyio_backend() -> str:
    """Return the async backend used by pytest-asyncio."""
    return "asyncio"


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, Any]:
    """Provide an async HTTP client pre-configured to talk to the FastAPI app.

    Uses ASGI transport (no server needed).
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

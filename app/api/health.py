"""
Health-check endpoint.
"""

from __future__ import annotations

import time

from fastapi import APIRouter

from app.models.response import HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=200,
    summary="Health check",
    description="Returns a simple status indicating the service is running.",
)
async def health_check() -> HealthResponse:
    """Respond with HTTP 200 and ``{"status": "ok"}``.

    This endpoint is used by load balancers and container orchestrators to
    verify the application is alive.
    """
    return HealthResponse(status="ok")

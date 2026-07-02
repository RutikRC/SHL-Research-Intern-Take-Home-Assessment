"""
Main API route definitions.

Only two endpoints are exposed:
- GET /health
- POST /chat
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_chat_service, get_db_dep
from app.models.request import ChatRequest
from app.models.response import ChatResponse
from app.services.chat_service import ChatService
from app.core.logging_ import get_logger

router = APIRouter(tags=["chat"])
logger = get_logger(__name__)


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=200,
    summary="Process a chat message",
    description=(
        "Accepts a full conversation history and returns a reply along with "
        "SHL assessment recommendations."
    ),
)
async def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
    session: AsyncSession = Depends(get_db_dep),
) -> ChatResponse:
    """Handle an incoming chat request.

    Args:
        request: The validated chat request containing the conversation history.
        chat_service: Injected ChatService instance.
        session: Injected database session for retrieval.

    Returns:
        A ChatResponse with the agent's reply and recommendations.
    """
    start_time = time.monotonic()
    logger.info("chat_request_received", message_count=len(request.messages))

    try:
        response = await chat_service.process_message(request, session=session)
    except Exception as exc:
        logger.error("chat_request_failed", error=str(exc))
        raise

    elapsed = time.monotonic() - start_time
    logger.info(
        "chat_response_sent",
        reply_length=len(response.reply),
        recommendation_count=len(response.recommendations),
        elapsed_seconds=round(elapsed, 4),
    )

    return response

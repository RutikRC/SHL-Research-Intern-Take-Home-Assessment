"""
Service that orchestrates the chat flow.

Coordinates conversation parsing, intent extraction, action selection,
retrieval, comparison, and response formatting.

This is the ONLY class that knows about the SHL response schema.

Pipeline:
    POST /chat
        → ConversationParser.parse()
        → IntentExtractor.extract()
        → Action selection (refuse / clarify / compare / refine / recommend)
        → RetrievalService (if needed)
        → ComparisonService (if needed)
        → RecommendationMapper (if needed)
        → ChatResponse
"""

from __future__ import annotations

import time
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging_ import get_logger
from app.database.models import Assessment
from app.models.intent import HiringIntent
from app.models.request import ChatRequest
from app.models.response import ChatResponse, Recommendation
from app.services.clarification_service import ClarificationService
from app.services.comparison_service import ComparisonService
from app.services.conversation_parser import ConversationParser
from app.services.gemini_service import GeminiService
from app.services.intent_extractor import IntentExtractor
from app.services.recommendation_mapper import RecommendationMapper
from app.services.retrieval_service import RetrievalService

logger = get_logger(__name__)

# Maximum number of recommendations to return
MAX_RECOMMENDATIONS = 10


class ChatService:
    """Orchestrates the conversational recommendation flow.

    Pipeline:
        1. Parse and validate the conversation history.
        2. Extract a structured HiringIntent from the conversation.
        3. Select the appropriate action based on intent flags.
        4. Execute the action (retrieval, comparison, clarification, refusal).
        5. Build and return a ChatResponse conforming to the SHL schema.
    """

    def __init__(
        self,
        conversation_parser: ConversationParser | None = None,
        intent_extractor: IntentExtractor | None = None,
        retrieval_service: RetrievalService | None = None,
        recommendation_mapper: RecommendationMapper | None = None,
        clarification_service: ClarificationService | None = None,
        comparison_service: ComparisonService | None = None,
        gemini_service: GeminiService | None = None,
    ) -> None:
        self._conversation_parser = conversation_parser or ConversationParser()
        self._intent_extractor = intent_extractor or IntentExtractor()
        self._retrieval_service = retrieval_service or RetrievalService()
        self._recommendation_mapper = recommendation_mapper or RecommendationMapper()
        self._clarification_service = clarification_service or ClarificationService()
        self._comparison_service = comparison_service or ComparisonService()
        self._gemini_service = gemini_service or GeminiService()

    async def process_message(
        self,
        request: ChatRequest,
        session: AsyncSession | None = None,
    ) -> ChatResponse:
        """Process an incoming chat request and return a response.

        Args:
            request: Validated chat request from the API layer.
            session: Optional database session. If None, retrieval is skipped.

        Returns:
            A ChatResponse with a reply and recommendations.
        """
        start = time.monotonic()
        logger.info(
            "chat_service_started",
            message_count=len(request.messages),
        )

        # 1. Parse and validate the conversation
        parsed = self._conversation_parser.parse(request.messages)
        logger.info(
            "chat_service_conversation_parsed",
            turn_count=parsed.turn_count,
            latest_user_chars=len(parsed.latest_user_message),
        )

        # 2. Extract structured hiring intent
        intent = self._intent_extractor.extract(parsed)
        logger.info(
            "chat_service_intent_extracted",
            role=intent.role or "none",
            skills=len(intent.skills),
            categories=[c.value for c in intent.assessment_categories],
            clarification_needed=intent.clarification_needed,
            refinement=intent.refinement_requested,
            comparison=intent.comparison_requested,
            off_topic=intent.off_topic,
            injection=intent.prompt_injection_detected,
        )

        # 3. Select action and build response
        response = await self._route(session, intent, parsed)

        elapsed = time.monotonic() - start
        logger.info(
            "chat_service_completed",
            recommendation_count=len(response.recommendations),
            reply_chars=len(response.reply),
            elapsed_seconds=round(elapsed, 4),
        )

        return response

    # ── Action Router ───────────────────────────────────────────────────────

    async def _route(
        self,
        session: AsyncSession | None,
        intent: HiringIntent,
        parsed: Any,  # ParsedConversation
    ) -> ChatResponse:
        """Route the request to the appropriate action handler.

        Priority:
            1. Refusal (off-topic or prompt injection)
            2. Comparison (user specifically asked to compare)
            3. Clarification (vague query — only if not specifically asking)
            4. Recommendation or Refinement (retrieval)
        """
        # Priority 1: Refusal
        if intent.off_topic or intent.prompt_injection_detected:
            return await self._refuse(intent, parsed)

        # Priority 2: Comparison overrides clarification
        # If the user explicitly asked to compare, honour that immediately
        if intent.comparison_requested:
            if session is not None:
                return await self._compare(session, intent, parsed)
            return self._empty_response()

        # Priority 3: Clarification
        if intent.clarification_needed:
            return await self._clarify(intent, parsed)

        # Priority 4: Recommendation or Refinement
        if session is not None:
            return await self._recommend(session, intent, parsed)

        return self._empty_response()

    # ── Actions ─────────────────────────────────────────────────────────────

    async def _refuse(
        self,
        intent: HiringIntent | None = None,
        parsed: Any = None,
    ) -> ChatResponse:
        """Return a refusal response for off-topic or injection queries."""
        messages = parsed.messages if parsed else []
        reply = await self._gemini_service.generate_refusal_reply(messages)
        return ChatResponse(
            reply=reply,
            recommendations=[],
            end_of_conversation=False,
        )

    async def _clarify(
        self,
        intent: HiringIntent,
        parsed: Any = None,
    ) -> ChatResponse:
        """Return a clarification question based on missing intent fields."""
        messages = parsed.messages if parsed else []
        reply = await self._gemini_service.generate_clarification_question(messages, intent)
        return ChatResponse(
            reply=reply,
            recommendations=[],
            end_of_conversation=False,
        )

    async def _compare(
        self,
        session: AsyncSession,
        intent: HiringIntent,
        parsed: Any = None,
    ) -> ChatResponse:
        """Handle a comparison request.

        Searches for each comparison target individually and combines results,
        so that "Compare OPQ and GSA" returns exactly those two assessments
        rather than all OPQ variants.
        """
        start = time.monotonic()
        logger.info(
            "chat_service_comparison_started",
            targets=intent.comparison_targets,
        )

        # Search for each target individually to get precise matches
        seen_names: set[str] = set()
        assessments: list[Assessment] = []

        targets = intent.comparison_targets if intent.comparison_targets else [intent.raw_user_query]
        for target in targets:
            results = await self._retrieve(session, intent, target)
            for a in results:
                if a.name not in seen_names:
                    assessments.append(a)
                    seen_names.add(a.name)
                    # Only take the top match per target
                    break

        logger.info(
            "chat_service_comparison_retrieved",
            count=len(assessments),
            targets=targets,
        )

        # Generate comparison text using Gemini (falls back to deterministic)
        messages = parsed.messages if parsed else []
        reply = await self._gemini_service.generate_comparison_reply(
            messages, intent, assessments,
        )

        elapsed = time.monotonic() - start
        logger.info(
            "chat_service_comparison_completed",
            elapsed_seconds=round(elapsed, 4),
        )

        return ChatResponse(
            reply=reply,
            recommendations=[],
            end_of_conversation=False,
        )

    async def _recommend(
        self,
        session: AsyncSession,
        intent: HiringIntent,
        parsed: Any = None,
    ) -> ChatResponse:
        """Handle a recommendation (or refinement) request."""
        start = time.monotonic()
        action = "refinement" if intent.refinement_requested else "recommendation"
        logger.info("chat_service_%s_started", action)

        query = self._build_search_query(intent)
        logger.info(
            "chat_service_search_query",
            query=query,
            action=action,
        )

        assessments = await self._retrieve(session, intent, query)
        recommendations = self._recommendation_mapper.map_many(assessments)

        messages = parsed.messages if parsed else []

        # Generate reply using Gemini (falls back to deterministic on failure)
        if action == "refinement":
            reply = await self._gemini_service.generate_refinement_reply(
                messages, intent, assessments,
            )
        elif recommendations:
            reply = await self._gemini_service.generate_recommendation_reply(
                messages, intent, assessments,
            )
        else:
            reply = "I couldn't find SHL assessments matching your requirements."

        elapsed = time.monotonic() - start
        logger.info(
            "chat_service_%s_completed",
            action,
            retrieved=len(assessments),
            recommended=len(recommendations),
            elapsed_seconds=round(elapsed, 4),
        )

        return ChatResponse(
            reply=reply,
            recommendations=recommendations[:MAX_RECOMMENDATIONS],
            end_of_conversation=False,
        )

    # ── Private helpers ─────────────────────────────────────────────────────

    async def _retrieve(
        self,
        session: AsyncSession,
        intent: HiringIntent,
        query: str,
    ) -> list[Assessment]:
        """Execute retrieval with error handling.

        Args:
            session: Database session.
            intent: HiringIntent (for safety checks).
            query: Search query text.

        Returns:
            List of Assessment objects, or empty list on failure.
        """
        if not query:
            return []

        try:
            return await self._retrieval_service.retrieve(
                session=session,
                query_text=query,
                top_k=MAX_RECOMMENDATIONS,
            )
        except (RuntimeError, ValueError) as exc:
            logger.error("chat_service_retrieval_failed", error=str(exc))
            return []

    @staticmethod
    def _empty_response() -> ChatResponse:
        """Return a safe empty response when no session is available."""
        return ChatResponse(
            reply="I couldn't find relevant SHL assessments for your request.",
            recommendations=[],
            end_of_conversation=False,
        )

    @staticmethod
    def _build_search_query(intent: HiringIntent) -> str:
        """Build a composite search query from the structured hiring intent.

        When refinement is detected, this merges all accumulated constraints:
        role + skills + categories. This prevents the latest refinement from
        replacing the entire context.

        Args:
            intent: The extracted HiringIntent with merged constraints.

        Returns:
            A search query string that combines all intent fields.
        """
        parts: list[str] = []

        if intent.role:
            parts.append(intent.role)

        if intent.skills:
            parts.extend(intent.skills)

        if intent.assessment_categories:
            for cat in intent.assessment_categories:
                parts.append(cat.value)

        if intent.experience:
            parts.append(intent.experience)

        if intent.remote_required:
            parts.append("remote testing")
        if intent.adaptive_required:
            parts.append("adaptive")

        if not parts:
            return intent.raw_user_query

        return " ".join(parts)
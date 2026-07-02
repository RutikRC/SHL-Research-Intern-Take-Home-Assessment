"""
Gemini NLG service for generating conversational replies.

Gemini ONLY generates natural language text — it NEVER selects assessments,
creates recommendation objects, or queries the database.

All methods return plain strings. The ChatService assembles the final ChatResponse.
"""

from __future__ import annotations

import time
import traceback
from typing import Any

from google import genai

from app.core.config import get_settings
from app.core.logging_ import get_logger
from app.database.models import Assessment
from app.models.intent import HiringIntent
from app.models.request import ChatMessage
from app.services.prompts import SYSTEM_PROMPT

logger = get_logger(__name__)

# Default deterministic fallback replies when Gemini is unavailable
FALLBACK_RECOMMENDATION = "Here are some SHL assessments matching your hiring requirements."
FALLBACK_CLARIFICATION = "Could you provide more details about the role you're hiring for?"
FALLBACK_REFINEMENT = "I've updated the recommendations based on your latest requirements."
FALLBACK_COMPARISON = "Here is a comparison of the requested assessments."
FALLBACK_REFUSAL = "I can only help recommend SHL Individual Test Solutions."
FALLBACK_EMPTY = "I couldn't find SHL assessments matching your requirements."


class GeminiService:
    """Generates conversational replies using Google Gemini.

    Every method returns a plain string. The ChatService is responsible for
    assembling the final ChatResponse with recommendations from the mapper.

    If Gemini fails (timeout, quota, auth, etc.), deterministic fallback
    replies are used automatically. The application never fails because of Gemini.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._model = settings.LLM_MODEL
        self._api_key = settings.GOOGLE_API_KEY
        self._client: genai.Client | None = None
        self._initialized = False

        if self._api_key:
            try:
                self._client = genai.Client(api_key=self._api_key)
                self._initialized = True
                logger.info(
                    "gemini_service_initialized",
                    model=self._model,
                )
            except Exception as exc:
                logger.warning(
                    "gemini_service_init_failed",
                    error=str(exc),
                    fallback="deterministic replies will be used",
                )
        else:
            logger.warning(
                "gemini_service_no_api_key",
                fallback="deterministic replies will be used",
            )

    # ── Public API ───────────────────────────────────────────────────────────

    async def generate_recommendation_reply(
        self,
        conversation_history: list[ChatMessage],
        intent: HiringIntent,
        assessments: list[Assessment],
    ) -> str:
        """Generate a reply explaining why the retrieved assessments are relevant.

        Args:
            conversation_history: Full conversation history.
            intent: The extracted hiring intent.
            assessments: The retrieved Assessment objects.

        Returns:
            A natural language explanation string, or fallback on failure.
        """
        if not self._initialized or not assessments:
            return FALLBACK_RECOMMENDATION if assessments else FALLBACK_EMPTY

        prompt = self._build_recommendation_prompt(conversation_history, intent, assessments)
        return await self._generate(prompt, "recommendation_reply")

    async def generate_clarification_question(
        self,
        conversation_history: list[ChatMessage],
        intent: HiringIntent,
    ) -> str:
        """Generate a single clarification question based on missing intent fields.

        Args:
            conversation_history: Full conversation history.
            intent: The extracted hiring intent.

        Returns:
            A clarification question string, or fallback on failure.
        """
        if not self._initialized:
            return FALLBACK_CLARIFICATION

        prompt = self._build_clarification_prompt(conversation_history, intent)
        return await self._generate(prompt, "clarification_question")

    async def generate_refinement_reply(
        self,
        conversation_history: list[ChatMessage],
        intent: HiringIntent,
        assessments: list[Assessment],
    ) -> str:
        """Generate a reply acknowledging the refinement and explaining updates.

        Args:
            conversation_history: Full conversation history.
            intent: The updated hiring intent.
            assessments: The updated retrieved assessments.

        Returns:
            A refinement acknowledgement string, or fallback on failure.
        """
        if not self._initialized:
            return FALLBACK_REFINEMENT

        prompt = self._build_refinement_prompt(conversation_history, intent, assessments)
        return await self._generate(prompt, "refinement_reply")

    async def generate_comparison_reply(
        self,
        conversation_history: list[ChatMessage],
        intent: HiringIntent,
        assessments: list[Assessment],
    ) -> str:
        """Generate a comparison of two or more assessments.

        Args:
            conversation_history: Full conversation history.
            intent: The extracted hiring intent.
            assessments: The assessments to compare (typically 2).

        Returns:
            A comparison text string, or fallback on failure.
        """
        if not self._initialized:
            return FALLBACK_COMPARISON

        prompt = self._build_comparison_prompt(conversation_history, intent, assessments)
        return await self._generate(prompt, "comparison_reply")

    async def generate_refusal_reply(
        self,
        conversation_history: list[ChatMessage],
        reason: str = "",
    ) -> str:
        """Generate a polite refusal for off-topic or injection queries.

        Args:
            conversation_history: Full conversation history.
            reason: Optional reason for refusal.

        Returns:
            A refusal string, or fallback on failure.
        """
        if not self._initialized:
            return FALLBACK_REFUSAL

        prompt = self._build_refusal_prompt(conversation_history, reason)
        return await self._generate(prompt, "refusal_reply")

    # ── Private helpers ──────────────────────────────────────────────────────

    async def _generate(self, prompt: str, context: str) -> str:
        """Send a prompt to Gemini and return the generated text.

        Args:
            prompt: The full prompt including system instructions.
            context: A label for logging (e.g. 'recommendation_reply').

        Returns:
            Generated text, or fallback on failure.
        """
        if not self._initialized or self._client is None:
            return self._get_fallback(context)

        start = time.monotonic()
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
            )
            elapsed = time.monotonic() - start
            text = response.text.strip() if response.text else ""

            logger.info(
                "gemini_generation_success",
                context=context,
                model=self._model,
                chars=len(text),
                elapsed_seconds=round(elapsed, 4),
            )

            return text if text else self._get_fallback(context)

        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.warning(
                "gemini_generation_failed",
                context=context,
                model=self._model,
                error=str(exc),
                elapsed_seconds=round(elapsed, 4),
                fallback="using deterministic reply",
            )
            return self._get_fallback(context)

    # ── Prompt builders ──────────────────────────────────────────────────────

    def _build_recommendation_prompt(
        self,
        history: list[ChatMessage],
        intent: HiringIntent,
        assessments: list[Assessment],
    ) -> str:
        """Build a prompt for generating a recommendation reply."""
        parts = [SYSTEM_PROMPT]
        parts.append("\n\n## Conversation History\n")
        for msg in history:
            parts.append(f"{msg.role.title()}: {msg.content}")

        parts.append(f"\n\n## Extracted Intent\nRole: {intent.role or 'Not specified'}")
        parts.append(f"Skills: {', '.join(intent.skills) if intent.skills else 'Not specified'}")
        parts.append(f"Experience: {intent.experience or 'Not specified'}")
        parts.append(f"Categories: {', '.join(c.value for c in intent.assessment_categories) if intent.assessment_categories else 'Not specified'}")

        parts.append("\n\n## Retrieved Assessments\n")
        for i, a in enumerate(assessments, 1):
            parts.append(f"Assessment {i}:")
            parts.append(f"  Name: {a.name}")
            parts.append(f"  Description: {a.description[:300]}")
            parts.append(f"  Categories: {', '.join(a.keys) if a.keys else 'General'}")
            parts.append(f"  Job Levels: {', '.join(a.job_levels) if a.job_levels else 'All levels'}")
            parts.append(f"  Languages: {', '.join(a.languages) if a.languages else 'English'}")
            parts.append(f"  Remote: {'Yes' if a.remote else 'No'}")
            parts.append(f"  Adaptive: {'Yes' if a.adaptive else 'No'}")
            parts.append(f"  Duration: {a.duration or 'Variable'}")
            parts.append("")

        parts.append("\n## Task\nExplain why these assessments fit the user's hiring requirements.")
        parts.append("Keep your response concise (2-3 sentences).")
        parts.append("Do NOT list assessment names in your response.")
        parts.append("Focus on explaining the relevance.")

        return "\n".join(parts)

    def _build_clarification_prompt(
        self,
        history: list[ChatMessage],
        intent: HiringIntent,
    ) -> str:
        """Build a prompt for generating a clarification question."""
        parts = [SYSTEM_PROMPT]
        parts.append("\n\n## Conversation History\n")
        for msg in history:
            parts.append(f"{msg.role.title()}: {msg.content}")

        parts.append(f"\n\n## Known Information")
        parts.append(f"Role: {intent.role or 'Not specified'}")
        parts.append(f"Skills: {', '.join(intent.skills) if intent.skills else 'Not specified'}")
        parts.append(f"Experience: {intent.experience or 'Not specified'}")

        parts.append("\n\n## Task\nAsk ONE clarification question to determine the most important missing information.")
        parts.append("Ask about the role first if not known.")
        parts.append("Then skills, then experience level, then assessment type.")
        parts.append("Ask only ONE question.")
        parts.append("Keep it concise and professional.")

        return "\n".join(parts)

    def _build_refinement_prompt(
        self,
        history: list[ChatMessage],
        intent: HiringIntent,
        assessments: list[Assessment],
    ) -> str:
        """Build a prompt for generating a refinement acknowledgement."""
        parts = [SYSTEM_PROMPT]
        parts.append("\n\n## Conversation History\n")
        for msg in history:
            parts.append(f"{msg.role.title()}: {msg.content}")

        parts.append(f"\n\n## Updated Intent")
        parts.append(f"Role: {intent.role or 'Not specified'}")
        parts.append(f"Skills: {', '.join(intent.skills) if intent.skills else 'Not specified'}")
        parts.append(f"Categories: {', '.join(c.value for c in intent.assessment_categories) if intent.assessment_categories else 'Not specified'}")

        parts.append("\n\n## Updated Assessments\n")
        for i, a in enumerate(assessments[:3], 1):
            parts.append(f"{i}. {a.name} — {a.description[:200]}")

        parts.append("\n\n## Task\nAcknowledge that the recommendations have been updated based on the user's latest requirements.")
        parts.append("Explain briefly how the new recommendations address the updated needs.")
        parts.append("Keep it to 1-2 sentences.")

        return "\n".join(parts)

    def _build_comparison_prompt(
        self,
        history: list[ChatMessage],
        intent: HiringIntent,
        assessments: list[Assessment],
    ) -> str:
        """Build a prompt for generating a comparison of assessments."""
        parts = [SYSTEM_PROMPT]
        parts.append("\n\n## Conversation History\n")
        for msg in history:
            parts.append(f"{msg.role.title()}: {msg.content}")

        parts.append("\n\n## Assessments to Compare\n")
        for i, a in enumerate(assessments, 1):
            parts.append(f"Assessment {i}: {a.name}")
            parts.append(f"  Purpose: {a.description[:200]}")
            parts.append(f"  Categories: {', '.join(a.keys) if a.keys else 'General'}")
            parts.append(f"  Job Levels: {', '.join(a.job_levels) if a.job_levels else 'All'}")
            parts.append(f"  Languages: {', '.join(a.languages) if a.languages else 'English'}")
            parts.append(f"  Remote Support: {'Yes' if a.remote else 'No'}")
            parts.append(f"  Adaptive Support: {'Yes' if a.adaptive else 'No'}")
            parts.append(f"  Duration: {a.duration or 'Variable'}")
            parts.append("")

        parts.append("\n\n## Task\nCompare these assessments highlighting key differences.")
        parts.append("Compare: purpose, categories, job levels, languages, remote support, adaptive support, and duration.")
        parts.append("Use ONLY the information provided above.")
        parts.append("Keep the comparison structured and easy to read.")

        return "\n".join(parts)

    def _build_refusal_prompt(
        self,
        history: list[ChatMessage],
        reason: str = "",
    ) -> str:
        """Build a prompt for generating a polite refusal."""
        parts = [SYSTEM_PROMPT]
        parts.append("\n\n## Conversation History\n")
        for msg in history:
            parts.append(f"{msg.role.title()}: {msg.content}")

        parts.append("\n\n## Task\nThe user's request is outside the scope of SHL assessment recommendations.")
        if reason:
            parts.append(f"\nReason: {reason}")
        parts.append("\nGenerate a polite refusal explaining that you can only help with SHL assessment recommendations.")
        parts.append("Keep it concise (1 sentence).")
        parts.append("Do not engage with the off-topic request.")

        return "\n".join(parts)

    # ── Fallback ─────────────────────────────────────────────────────────────

    @staticmethod
    def _get_fallback(context: str) -> str:
        """Return the appropriate deterministic fallback for the given context."""
        fallbacks = {
            "recommendation_reply": FALLBACK_RECOMMENDATION,
            "clarification_question": FALLBACK_CLARIFICATION,
            "refinement_reply": FALLBACK_REFINEMENT,
            "comparison_reply": FALLBACK_COMPARISON,
            "refusal_reply": FALLBACK_REFUSAL,
        }
        return fallbacks.get(context, FALLBACK_EMPTY)
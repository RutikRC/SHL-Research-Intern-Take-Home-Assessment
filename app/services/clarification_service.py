"""
Determines the single most important clarification question based on missing intent fields.

Returns only one question per request — the highest-priority missing information.
"""

from __future__ import annotations

from app.core.logging_ import get_logger
from app.models.intent import HiringIntent

logger = get_logger(__name__)


class ClarificationService:
    """Generates a single deterministic clarification question.

    Priority order:
        1. Role — most critical information for assessment selection.
        2. Skills — specific technical/behavioral skills.
        3. Experience level — seniority / years of experience.
        4. Assessment category — test type preference.
    
    Only one question is returned per call.
    """

    def get_clarification_question(self, intent: HiringIntent) -> str:
        """Return the single best clarification question based on missing intent fields.

        Args:
            intent: The current HiringIntent with what's known so far.

        Returns:
            A single clarification question string.
        """
        if not intent.role:
            return "Could you tell me what role you are hiring for?"

        if not intent.skills:
            return "What specific skills are you looking for in the candidate?"

        if not intent.experience:
            return "What experience level do you require for this role?"

        if not intent.assessment_categories:
            return "What type of assessments are you looking for — technical, personality, or cognitive?"

        # Fallback if everything seems populated
        return "Could you provide more details about your hiring requirements?"
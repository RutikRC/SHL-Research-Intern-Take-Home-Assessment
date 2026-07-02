"""
Pydantic models representing the user's hiring intent extracted from conversation history.

This is the single source of truth for all downstream phases (retrieval,
recommendation, clarification, comparison, guardrails).
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints


class JobLevelEnum(str, Enum):
    """Standardised job levels used in SHL assessments."""

    ENTRY_LEVEL = "Entry-Level"
    GRADUATE = "Graduate"
    PROFESSIONAL = "Professional Individual Contributor"
    MID_PROFESSIONAL = "Mid-Professional"
    FRONT_LINE_MANAGER = "Front Line Manager"
    MANAGER = "Manager"
    DIRECTOR = "Director"
    EXECUTIVE = "Executive"
    GENERAL_POPULATION = "General Population"
    SUPERVISOR = "Supervisor"
    UNKNOWN = "Unknown"


class SkillArea(str, Enum):
    """Broad skill/competency areas detectable from user queries."""

    TECHNICAL = "Technical"
    PERSONALITY = "Personality & Behavior"
    COGNITIVE = "Ability & Aptitude"
    KNOWLEDGE = "Knowledge & Skills"
    SIMULATION = "Simulations"
    COMPETENCY = "Competencies"
    BIODATA = "Biodata & Situational Judgment"
    DEVELOPMENT = "Development & 360"
    ASSESSMENT_EXERCISES = "Assessment Exercises"
    UNKNOWN = "Unknown"


class HiringIntent(BaseModel):
    """Structured representation of the user's hiring intent.

    Extracted from the full conversation history every request.
    Used by all downstream phases for retrieval, recommendation,
    clarification, comparison, and guardrails.
    """

    # Role and skills
    role: Annotated[str, StringConstraints(max_length=500)] = ""
    skills: list[str] = Field(default_factory=list)
    experience: Annotated[str, StringConstraints(max_length=200)] = ""
    job_level: JobLevelEnum = JobLevelEnum.UNKNOWN
    industry: Annotated[str, StringConstraints(max_length=200)] = ""

    # Assessment requirements
    assessment_categories: list[SkillArea] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    remote_required: bool | None = None
    adaptive_required: bool | None = None

    # Refinement
    refinement_requested: bool = False
    previous_role: Annotated[str, StringConstraints(max_length=500)] = ""
    previous_skills: list[str] = Field(default_factory=list)
    previous_categories: list[SkillArea] = Field(default_factory=list)

    # Comparison
    comparison_requested: bool = False
    comparison_targets: list[str] = Field(default_factory=list)

    # Clarification
    clarification_needed: bool = False

    # Safety
    off_topic: bool = False
    prompt_injection_detected: bool = False
    off_topic_reason: Annotated[str, StringConstraints(max_length=500)] = ""

    # Raw query
    raw_user_query: str = ""

    # Metadata
    message_count: int = 0
    turn_count: int = 0
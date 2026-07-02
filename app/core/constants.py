"""
Application-wide constants.
"""

from __future__ import annotations

from enum import StrEnum


class MessageRole(StrEnum):
    """Valid roles for a chat message."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class TestType(StrEnum):
    """Supported SHL individual test-solution types."""

    ABILITY = "Ability"
    BEHAVIOURAL = "Behavioural"
    KNOWLEDGE = "Knowledge"
    SKILL = "Skill"
    PERSONALITY = "Personality"
    SIMULATION = "Simulation"


class TestDuration(StrEnum):
    """Common test-duration buckets, stored as strings for clarity."""

    SHORT = "short"          # < 15 min
    MEDIUM = "medium"        # 15-30 min
    LONG = "long"            # 30-45 min
    EXTENDED = "extended"    # > 45 min


class SupportedLanguage(StrEnum):
    """Languages the catalog metadata supports."""

    ENGLISH = "en"
    ARABIC = "ar"
    CHINESE_SIMPLIFIED = "zh"
    DUTCH = "nl"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    JAPANESE = "ja"
    KOREAN = "ko"
    POLISH = "pl"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    SPANISH = "es"
    SWEDISH = "sv"
    THAI = "th"
    TURKISH = "tr"


# ── HTTP header constants ──────────────────────────────────────────────────
REQUEST_ID_HEADER: str = "X-Request-ID"
CORRELATION_ID_HEADER: str = "X-Correlation-ID"

# ── Application limits ─────────────────────────────────────────────────────
MAX_RECOMMENDATIONS_PER_RESPONSE: int = 10
MAX_CONVERSATION_TURNS: int = 50

# ── Catalog ────────────────────────────────────────────────────────────────
CATALOG_SOURCE_NAME: str = "shl_product_catalog"
CATALOG_EMBEDDINGS_COLLECTION: str = "catalog_embeddings"

"""
General-purpose helper utilities.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone


def generate_request_id() -> str:
    """Generate a unique request identifier.

    Returns:
        A string combining a timestamp prefix and a UUID hex suffix.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{ts}-{uuid.uuid4().hex}"


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to a maximum length, appending an ellipsis if needed.

    Args:
        text: The input string.
        max_length: Maximum allowed length.

    Returns:
        Truncated string.
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."

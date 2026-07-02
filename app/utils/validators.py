"""
Input validators for request data.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import StringConstraints


# Re-usable string constraint for message roles – used in request models.
RoleField = Annotated[
    str,
    StringConstraints(pattern=r"^(user|assistant|system)$"),
]

# Re-usable constraint for short textual identifiers.
IdentifierField = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_-]+$"),
]

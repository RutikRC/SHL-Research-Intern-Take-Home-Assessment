"""
Pydantic models representing SHL catalog data.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints


class CatalogItem(BaseModel):
    """Metadata for a single SHL test solution."""

    name: Annotated[
        str,
        StringConstraints(min_length=1, max_length=500),
    ]
    url: Annotated[
        str,
        StringConstraints(min_length=1, max_length=2048),
    ]
    test_type: Annotated[
        str,
        StringConstraints(min_length=1, max_length=100),
    ]
    description: Annotated[
        str,
        StringConstraints(max_length=5000),
    ] = ""
    duration_minutes: int | None = None
    languages: list[str] = Field(default_factory=list)
    remote_testing: bool = True
    adaptive_irt: bool = False


class CatalogSearchResult(BaseModel):
    """Result of a catalog search query."""

    items: list[CatalogItem] = Field(default_factory=list)
    total_count: int = 0
    query: str = ""

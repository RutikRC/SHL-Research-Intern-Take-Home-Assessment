"""
Pydantic models for API response bodies.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints


class Recommendation(BaseModel):
    """A single product recommendation returned by the agent."""

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


class ChatResponse(BaseModel):
    """The response body for the POST /chat endpoint."""

    reply: Annotated[
        str,
        StringConstraints(min_length=1),
    ]
    recommendations: Annotated[
        list[Recommendation],
        Field(default_factory=list, max_length=10),
    ]
    end_of_conversation: bool = False


class HealthResponse(BaseModel):
    """The response body for the GET /health endpoint."""

    status: Annotated[
        str,
        StringConstraints(pattern=r"^(ok|error)$"),
    ] = "ok"

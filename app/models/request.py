"""
Pydantic models for incoming API request bodies.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints


class ChatMessage(BaseModel):
    """A single message in a conversation."""

    role: Annotated[
        str,
        StringConstraints(pattern=r"^(user|assistant|system)$"),
    ]
    content: Annotated[
        str,
        StringConstraints(min_length=1, max_length=10_000),
    ]


class ChatRequest(BaseModel):
    """The request body for the POST /chat endpoint."""

    messages: Annotated[
        list[ChatMessage],
        Field(min_length=1, max_length=50),
    ]

    model_config = {"json_schema_extra": {"example": {"messages": [{"role": "user", "content": "Hiring a Java developer"}]}}}

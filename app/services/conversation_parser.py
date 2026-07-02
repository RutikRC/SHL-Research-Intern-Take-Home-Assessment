"""
Parses and validates the conversation history from a ChatRequest.

Produces a structured representation of the conversation that the
IntentExtractor can use to build a HiringIntent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fastapi import HTTPException

from app.core.logging_ import get_logger
from app.models.request import ChatMessage

logger = get_logger(__name__)


@dataclass
class ParsedConversation:
    """Structured representation of a validated conversation."""

    messages: list[ChatMessage] = field(default_factory=list)
    user_messages: list[str] = field(default_factory=list)
    assistant_messages: list[str] = field(default_factory=list)
    latest_user_message: str = ""
    latest_assistant_message: str = ""
    message_count: int = 0
    turn_count: int = 0
    is_valid: bool = True
    validation_error: str = ""


class ConversationParser:
    """Validates and parses a list of ChatMessage objects.

    Responsibilities:
        - Validate message order and roles.
        - Extract the latest user and assistant messages.
        - Produce a ParsedConversation dataclass.
        - Detect malformed conversations.
    """

    VALID_ROLES = {"user", "assistant", "system"}

    def parse(self, messages: list[ChatMessage]) -> ParsedConversation:
        """Parse and validate a list of chat messages.

        Args:
            messages: The list of ChatMessage objects from the request.

        Returns:
            A ParsedConversation with extracted fields.

        Raises:
            HTTPException: If the conversation is empty or malformed.
        """
        # Validate messages
        error = self._validate_messages(messages)
        if error:
            raise HTTPException(status_code=400, detail=error)

        user_msgs: list[str] = []
        assistant_msgs: list[str] = []

        for msg in messages:
            if msg.role == "user":
                user_msgs.append(msg.content)
            elif msg.role == "assistant":
                assistant_msgs.append(msg.content)

        # Calculate turns (a turn = user message + optional assistant response)
        turn_count = len(user_msgs)

        parsed = ParsedConversation(
            messages=messages,
            user_messages=user_msgs,
            assistant_messages=assistant_msgs,
            latest_user_message=user_msgs[-1] if user_msgs else "",
            latest_assistant_message=assistant_msgs[-1] if assistant_msgs else "",
            message_count=len(messages),
            turn_count=turn_count,
        )

        logger.debug(
            "conversation_parsed",
            message_count=parsed.message_count,
            turn_count=parsed.turn_count,
            latest_user_chars=len(parsed.latest_user_message),
        )

        return parsed

    def _validate_messages(self, messages: list[ChatMessage]) -> str:
        """Validate the message list structure.

        Args:
            messages: The list of ChatMessage objects.

        Returns:
            An error string if validation fails, empty string otherwise.
        """
        if not messages:
            return "Conversation history is empty."

        for i, msg in enumerate(messages):
            if msg.role not in self.VALID_ROLES:
                return f"Invalid role '{msg.role}' at position {i}. Must be one of: {', '.join(sorted(self.VALID_ROLES))}."

        # Check that the first message is not from assistant (conversation should start with user or system)
        if messages[0].role == "assistant":
            return "Conversation cannot start with an assistant message."

        # Check for consecutive user messages (should alternate)
        for i in range(1, len(messages)):
            if messages[i].role == messages[i - 1].role and messages[i].role != "system":
                if messages[i].role == "user":
                    return f"Consecutive user messages at positions {i-1} and {i}. Messages should alternate."

        return ""

    @staticmethod
    def latest_user_message(messages: list[ChatMessage]) -> str:
        """Extract the content of the most recent user message.

        Args:
            messages: The list of ChatMessage objects.

        Returns:
            The content of the latest user message.

        Raises:
            HTTPException: If no user message is found.
        """
        for msg in reversed(messages):
            if msg.role == "user":
                return msg.content
        raise HTTPException(
            status_code=400,
            detail="No user message found in the conversation history.",
        )

    @staticmethod
    def latest_assistant_message(messages: list[ChatMessage]) -> str:
        """Extract the content of the most recent assistant message.

        Args:
            messages: The list of ChatMessage objects.

        Returns:
            The content of the latest assistant message, or empty string.
        """
        for msg in reversed(messages):
            if msg.role == "assistant":
                return msg.content
        return ""

    @staticmethod
    def conversation_summary(messages: list[ChatMessage]) -> str:
        """Create a compact text summary of the conversation.

        Args:
            messages: The list of ChatMessage objects.

        Returns:
            A string summarising the conversation turns.
        """
        parts: list[str] = []
        for msg in messages:
            prefix = "User" if msg.role == "user" else "Assistant" if msg.role == "assistant" else "System"
            content = msg.content[:200]
            parts.append(f"{prefix}: {content}")
        return "\n".join(parts)
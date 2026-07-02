"""
Service for prompt template management.

Placeholder – will house LangChain prompt templates and few-shot examples.
"""

from __future__ import annotations


class PromptService:
    """Manages prompt templates, system prompts, and few-shot examples."""

    def build_system_prompt(self) -> str:
        """Return the system prompt for the chat agent.

        Raises:
            NotImplementedError: Always – real implementation pending.
        """
        raise NotImplementedError("PromptService.build_system_prompt is not implemented yet.")

    def build_user_prompt(self, messages: list[dict]) -> str:
        """Build the user-facing prompt from conversation history.

        Raises:
            NotImplementedError: Always – real implementation pending.
        """
        raise NotImplementedError("PromptService.build_user_prompt is not implemented yet.")

    def build_few_shot_examples(self) -> list[dict]:
        """Return few-shot examples for in-context learning.

        Raises:
            NotImplementedError: Always – real implementation pending.
        """
        raise NotImplementedError("PromptService.build_few_shot_examples is not implemented yet.")

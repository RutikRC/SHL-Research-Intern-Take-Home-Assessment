"""
Embedding client that uses Google Gemini for vector generation.

Uses Gemini's ``gemini-embedding-2`` model via the official ``google-genai`` SDK.
Configuration is read from the application Settings singleton.
"""

from __future__ import annotations

import asyncio
from functools import partial

from google import genai
from google.genai import types

from app.core.config import get_settings
from app.core.logging_ import get_logger

logger = get_logger(__name__)


class EmbeddingClient:
    """Generates embedding vectors using Google Gemini.

    Uses ``gemini-embedding-2`` via the official ``google-genai`` SDK.
    Works everywhere — no Ollama required.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._model = "gemini-embedding-2"
        self._dimension = settings.EMBEDDING_DIMENSION
        self._api_key = settings.GOOGLE_API_KEY
        self._client = None
        self._initialized = False

        if self._api_key:
            try:
                self._client = genai.Client(api_key=self._api_key)
                self._initialized = True
                logger.info(
                    "EmbeddingClient initialized | model=%s | dimension=%d",
                    self._model,
                    self._dimension,
                )
            except Exception as exc:
                logger.warning(
                    "EmbeddingClient init failed | error=%s",
                    str(exc),
                )
        else:
            logger.warning(
                "EmbeddingClient missing GOOGLE_API_KEY | embeddings disabled",
            )

    @property
    def model(self) -> str:
        """Return the embedding model name."""
        return self._model

    @property
    def dimension(self) -> int:
        """Return the expected embedding dimension."""
        return self._dimension

    @property
    def initialized(self) -> bool:
        """Whether the client is ready to generate embeddings."""
        return self._initialized

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text using Gemini.

        Args:
            text: The input text to embed.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            RuntimeError: If the client is not initialised or the API call fails.
            ValueError: If the input text is empty.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        if not self._initialized or self._client is None:
            raise RuntimeError(
                "Embedding client not initialised. Set GOOGLE_API_KEY in .env",
            )

        try:
            # gemini-embedding-2 returns 3072-dimensional vectors by default
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                partial(
                    self._client.models.embed_content,
                    model=self._model,
                    contents=text,
                ),
            )
        except Exception as exc:
            logger.error(
                "Gemini embedding failed | model=%s | error=%s",
                self._model,
                str(exc),
            )
            raise RuntimeError(
                f"Gemini embedding failed: {exc}",
            ) from exc

        if not result or not result.embeddings:
            raise RuntimeError("Gemini returned empty embeddings")

        # Extract the embedding vector from the response
        embedding_data = result.embeddings[0]
        vector = list(embedding_data.values) if hasattr(embedding_data, 'values') else list(embedding_data)

        if not isinstance(vector, list) or len(vector) == 0:
            raise RuntimeError("Gemini returned an empty vector")

        actual_dim = len(vector)
        if actual_dim != self._dimension:
            logger.warning(
                "Embedding dimension mismatch | expected=%d | actual=%d | model=%s",
                self._dimension,
                actual_dim,
                self._model,
            )

        return vector
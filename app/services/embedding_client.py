"""
Embedding client that uses LangChain + Ollama to generate vectors.

Reads configuration from Settings (OLLAMA_BASE_URL, EMBEDDING_MODEL).
"""

from __future__ import annotations

from langchain_ollama import OllamaEmbeddings

from app.core.config import get_settings
from app.core.logging_ import get_logger

logger = get_logger(__name__)


class EmbeddingClient:
    """Thin wrapper around LangChain's OllamaEmbeddings for vector generation.

    Provides a single ``embed`` method that accepts text and returns
    a list of floats.  Configuration (model name, base URL) is read
    from the application Settings singleton.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._model = settings.EMBEDDING_MODEL
        self._dimension = settings.EMBEDDING_DIMENSION

        self._client = OllamaEmbeddings(
            model=self._model,
            base_url=settings.OLLAMA_BASE_URL,
        )

        logger.info(
            "EmbeddingClient initialized | model=%s | base_url=%s | dimension=%d",
            self._model,
            settings.OLLAMA_BASE_URL,
            self._dimension,
        )

    @property
    def model(self) -> str:
        """Return the embedding model name."""
        return self._model

    @property
    def dimension(self) -> int:
        """Return the expected embedding dimension."""
        return self._dimension

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text.

        Args:
            text: The input text to embed.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            RuntimeError: If the embedding model returns a vector with
                unexpected dimensions or if the Ollama call fails.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        try:
            vector = await self._client.aembed_query(text)
        except Exception as exc:
            logger.error(
                "Embedding generation failed | model=%s | error=%s",
                self._model,
                str(exc),
            )
            raise RuntimeError(
                f"Embedding model '{self._model}' failed: {exc}",
            ) from exc

        if not isinstance(vector, list) or len(vector) == 0:
            raise RuntimeError(
                f"Embedding model '{self._model}' returned an empty or invalid vector",
            )

        actual_dim = len(vector)
        if actual_dim != self._dimension:
            logger.warning(
                "Embedding dimension mismatch | expected=%d | actual=%d | model=%s",
                self._dimension,
                actual_dim,
                self._model,
            )

        return vector
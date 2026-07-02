"""
Embedding client that tries Ollama first, then falls back to Google Gemini.

On Render (or any environment without Ollama), the Gemini fallback ensures
query embeddings are still generated. Batch embedding generation should be
run locally with Ollama before deployment.
"""

from __future__ import annotations

from app.core.config import get_settings
from app.core.logging_ import get_logger

logger = get_logger(__name__)


class EmbeddingClient:
    """Generates embedding vectors using Ollama (preferred) or Gemini (fallback).

    On Render, Ollama is unavailable so the client automatically falls back
    to Google Gemini's ``text-embedding-004`` model.  Configuration is read
    from the application Settings singleton.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._model = settings.EMBEDDING_MODEL
        self._dimension = settings.EMBEDDING_DIMENSION
        self._ollama_url = settings.OLLAMA_BASE_URL
        self._gemini_api_key = settings.GOOGLE_API_KEY

        self._ollama_client = None
        self._gemini_client = None
        self._using_fallback = False

        # Try initialising Ollama first
        self._init_ollama()

        # If Ollama failed and we have a Gemini key, initialise Gemini
        if self._ollama_client is None and self._gemini_api_key:
            self._init_gemini()

        logger.info(
            "EmbeddingClient initialized | model=%s | dimension=%d | using_fallback=%s",
            self._model,
            self._dimension,
            self._using_fallback,
        )

    # ── Initialisation ──────────────────────────────────────────────────────

    def _init_ollama(self) -> None:
        """Attempt to initialise the Ollama embedding client."""
        try:
            from langchain_ollama import OllamaEmbeddings

            self._ollama_client = OllamaEmbeddings(
                model=self._model,
                base_url=self._ollama_url,
            )
            logger.info(
                "EmbeddingClient using Ollama | model=%s | base_url=%s",
                self._model,
                self._ollama_url,
            )
        except Exception as exc:
            logger.warning(
                "EmbeddingClient Ollama init failed | error=%s",
                str(exc),
            )
            self._ollama_client = None

    def _init_gemini(self) -> None:
        """Initialise the Google Gemini embedding client as fallback."""
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            self._gemini_client = GoogleGenerativeAIEmbeddings(
                model="text-embedding-004",
                google_api_key=self._gemini_api_key,
            )
            self._using_fallback = True
            logger.info(
                "EmbeddingClient using Gemini fallback | model=text-embedding-004",
            )
        except Exception as exc:
            logger.warning(
                "EmbeddingClient Gemini init failed | error=%s",
                str(exc),
            )
            self._gemini_client = None

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def model(self) -> str:
        """Return the active embedding model name."""
        if self._using_fallback:
            return "text-embedding-004"
        return self._model

    @property
    def dimension(self) -> int:
        """Return the expected embedding dimension."""
        return self._dimension

    @property
    def using_fallback(self) -> bool:
        """Whether the client is using the Gemini fallback."""
        return self._using_fallback

    # ── Embedding ───────────────────────────────────────────────────────────

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text.

        Tries Ollama first. Falls back to Gemini if Ollama is unavailable.

        Args:
            text: The input text to embed.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            RuntimeError: If both Ollama and Gemini fail.
            ValueError: If the input text is empty.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        # Try Ollama first
        if self._ollama_client is not None:
            try:
                return await self._embed_ollama(text)
            except Exception as exc:
                logger.warning(
                    "Ollama embedding failed, trying Gemini fallback | error=%s",
                    str(exc),
                )

        # Fall back to Gemini
        if self._gemini_client is not None:
            try:
                return await self._embed_gemini(text)
            except Exception as exc:
                logger.error(
                    "Gemini embedding also failed | error=%s",
                    str(exc),
                )
                raise RuntimeError(
                    f"All embedding backends failed: {exc}",
                ) from exc

        raise RuntimeError("No embedding backend available (Ollama and Gemini both unavailable)")

    async def _embed_ollama(self, text: str) -> list[float]:
        """Generate embedding using Ollama."""
        vector = await self._ollama_client.aembed_query(text)

        if not isinstance(vector, list) or len(vector) == 0:
            raise RuntimeError("Ollama returned an empty vector")

        actual_dim = len(vector)
        if actual_dim != self._dimension:
            logger.warning(
                "Ollama dimension mismatch | expected=%d | actual=%d",
                self._dimension,
                actual_dim,
            )

        return vector

    async def _embed_gemini(self, text: str) -> list[float]:
        """Generate embedding using Google Gemini."""
        vector = await self._gemini_client.aembed_query(text)

        if not isinstance(vector, list) or len(vector) == 0:
            raise RuntimeError("Gemini returned an empty vector")

        actual_dim = len(vector)
        if actual_dim != self._dimension:
            logger.warning(
                "Gemini dimension mismatch | expected=%d | actual=%d",
                self._dimension,
                actual_dim,
            )

        return vector
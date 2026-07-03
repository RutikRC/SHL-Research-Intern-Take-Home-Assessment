"""
Application configuration using Pydantic BaseSettings.

Reads from environment variables and .env file.
All credentials and tunable parameters are configured here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar

from pydantic_settings import BaseSettings, SettingsConfigDict


# Root directory of the project
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    APP_NAME: str = "SHL AI Agent"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    APP_ENV: str = "local"

    # ── Server ───────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ── Database ─────────────────────────────────────────────────────────────
    # APP_ENV=local uses individual Postgres credentials below.
    # APP_ENV=prod uses DATABASE_URL (Neon connection string).
    DATABASE_URL: str = ""
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "shl_ai"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    DATABASE_ECHO: bool = False
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20

    # ── Google Gemini ────────────────────────────────────────────────────────
    GOOGLE_API_KEY: str = ""
    LLM_MODEL: str = "gemini-2.0-flash-001"

    # ── Embeddings ───────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "nomic-embed-text"
    EMBEDDING_DIMENSION: int = 3072
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # ── SHL Catalog ──────────────────────────────────────────────────────────
    CATALOG_API_URL: str = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
    CATALOG_LOCAL_PATH: str = str(PROJECT_ROOT / "data" / "catalogue.json")
    CATALOG_CACHE_TTL_SECONDS: int = 3600
    HTTP_REQUEST_TIMEOUT_SECONDS: int = 30
    HTTP_MAX_RETRIES: int = 3

    # ── CORS ─────────────────────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list[str] = ["*"]
    CORS_ALLOW_HEADERS: list[str] = ["*"]

    # ── Served paths (computed) ──────────────────────────────────────────────
    STATIC_DIR: str = str(PROJECT_ROOT / "static")

    # ── Constants used across the app ────────────────────────────────────────
    MAX_MESSAGES_PER_REQUEST: int = 50
    MAX_MESSAGE_LENGTH: int = 10_000

    # ── Test mode ────────────────────────────────────────────────────────────
    TESTING: bool = False

    # ── Internal helpers ─────────────────────────────────────────────────────

    _database_url_override: ClassVar[str | None] = None

    @classmethod
    def set_database_url(cls, url: str) -> None:
        """Override database URL at runtime (used by tests)."""
        cls._database_url_override = url

    @property
    def database_url(self) -> str:
        """Return a sync connection URL.

        When APP_ENV=prod and DATABASE_URL is set (Neon), use it directly.
        Otherwise build from individual Postgres credentials.

        Returns:
            A ``postgresql+psycopg2://`` or ``postgresql://`` connection string.
        """
        if self._database_url_override:
            return self._database_url_override

        if self.APP_ENV == "prod" and self.DATABASE_URL:
            return self.DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://")

        return (
            f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def async_database_url(self) -> str:
        """Return an async connection URL.

        When APP_ENV=prod and DATABASE_URL is set (Neon), convert it to asyncpg.
        Otherwise build from individual Postgres credentials.

        Asyncpg uses ``ssl`` instead of ``sslmode``, so we convert
        ``?sslmode=require`` → ``?ssl=require`` for compatibility.

        Returns:
            A ``postgresql+asyncpg://`` connection string.
        """
        if self.APP_ENV == "prod" and self.DATABASE_URL:
            url = self.DATABASE_URL
            # Convert sslmode to ssl for asyncpg compatibility
            url = url.replace("?sslmode=require", "?ssl=require")
            url = url.replace("&sslmode=require", "&ssl=require")
            return url.replace("postgresql://", "postgresql+asyncpg://")

        return self.database_url.replace(
            "postgresql+psycopg2://", "postgresql+asyncpg://",
        )


# Singleton instance – call `get_settings()` in production code.
_settings_instance: Settings | None = None


def get_settings() -> Settings:
    """Return the singleton Settings instance."""
    global _settings_instance  # noqa: PLW0603
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
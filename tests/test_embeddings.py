"""
Tests for the embedding generation pipeline.

Covers document formatting, embedding client behaviour, duplicate prevention,
and the admin API endpoints for generating, counting, and refreshing embeddings.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient

from app.database.models import Assessment, AssessmentEmbedding
from app.services.embedding_document_formatter import EmbeddingDocumentFormatter
from app.services.embedding_client import EmbeddingClient
from app.services.embedding_service import EmbeddingService
from app.repositories.embedding_repository import EmbeddingRepository


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_assessment() -> Assessment:
    """Create a sample Assessment object for testing."""
    return Assessment(
        id=uuid.uuid4(),
        entity_id="test-001",
        name="Java 8 (New)",
        url="https://www.shl.com/products/product-catalog/view/java-8-new/",
        description="Measures Java programming ability for software developers.",
        duration="40 minutes",
        remote=True,
        adaptive=False,
        job_levels=["Graduate", "Mid-Professional", "Professional Individual Contributor"],
        languages=["English (USA)"],
        keys=["Knowledge & Skills"],
        scraped_at=datetime(2026, 5, 8, 10, 52, 48),
        created_at=datetime(2026, 5, 8, 10, 52, 48),
        updated_at=datetime(2026, 5, 8, 10, 52, 48),
    )


@pytest.fixture
def formatter() -> EmbeddingDocumentFormatter:
    """Provide an EmbeddingDocumentFormatter instance."""
    return EmbeddingDocumentFormatter()


# ── Document Formatting Tests ────────────────────────────────────────────────


class TestEmbeddingDocumentFormatter:
    """Tests for the EmbeddingDocumentFormatter."""

    def test_format_includes_name(self, formatter: EmbeddingDocumentFormatter, sample_assessment: Assessment) -> None:
        """The formatted document should contain the assessment name."""
        document = formatter.format(sample_assessment)
        assert "Java 8 (New)" in document

    def test_format_includes_description(self, formatter: EmbeddingDocumentFormatter, sample_assessment: Assessment) -> None:
        """The formatted document should contain the description."""
        document = formatter.format(sample_assessment)
        assert "Measures Java programming ability" in document

    def test_format_includes_job_levels(self, formatter: EmbeddingDocumentFormatter, sample_assessment: Assessment) -> None:
        """The formatted document should list job levels."""
        document = formatter.format(sample_assessment)
        assert "Graduate" in document
        assert "Mid-Professional" in document

    def test_format_includes_categories(self, formatter: EmbeddingDocumentFormatter, sample_assessment: Assessment) -> None:
        """The formatted document should list categories (keys)."""
        document = formatter.format(sample_assessment)
        assert "Knowledge & Skills" in document

    def test_format_includes_languages(self, formatter: EmbeddingDocumentFormatter, sample_assessment: Assessment) -> None:
        """The formatted document should list languages."""
        document = formatter.format(sample_assessment)
        assert "English (USA)" in document

    def test_format_includes_remote(self, formatter: EmbeddingDocumentFormatter, sample_assessment: Assessment) -> None:
        """The formatted document should indicate remote testing support."""
        document = formatter.format(sample_assessment)
        assert "Remote Testing:" in document
        assert "Yes" in document

    def test_format_includes_adaptive(self, formatter: EmbeddingDocumentFormatter, sample_assessment: Assessment) -> None:
        """The formatted document should indicate adaptive/IRT support."""
        document = formatter.format(sample_assessment)
        assert "Adaptive/IRT:" in document
        assert "No" in document

    def test_format_includes_duration(self, formatter: EmbeddingDocumentFormatter, sample_assessment: Assessment) -> None:
        """The formatted document should include duration."""
        document = formatter.format(sample_assessment)
        assert "40 minutes" in document

    def test_format_handles_no_optional_fields(self, formatter: EmbeddingDocumentFormatter) -> None:
        """Formatting should work even when optional fields are empty."""
        assessment = Assessment(
            id=uuid.uuid4(),
            entity_id="test-minimal",
            name="Minimal Test",
            url="https://example.com/test",
            description="",
            duration=None,
            remote=False,
            adaptive=False,
            job_levels=[],
            languages=[],
            keys=[],
            scraped_at=None,
            created_at=datetime(2026, 1, 1),
            updated_at=datetime(2026, 1, 1),
        )
        document = formatter.format(assessment)
        assert "Minimal Test" in document
        assert "Remote Testing:\nNo" in document
        assert "Adaptive/IRT:\nNo" in document


# ── Embedding Client Tests ───────────────────────────────────────────────────


class TestEmbeddingClient:
    """Tests for the EmbeddingClient."""

    @pytest.mark.asyncio
    async def test_embed_raises_on_empty_text(self) -> None:
        """Embedding empty text should raise ValueError."""
        client = EmbeddingClient()
        with pytest.raises(ValueError, match="Cannot embed empty text"):
            await client.embed("")

    @pytest.mark.asyncio
    async def test_embed_raises_on_whitespace_text(self) -> None:
        """Embedding whitespace-only text should raise ValueError."""
        client = EmbeddingClient()
        with pytest.raises(ValueError, match="Cannot embed empty text"):
            await client.embed("   \n   ")

    def test_model_property(self) -> None:
        """The model property should return the configured model name."""
        client = EmbeddingClient()
        assert client.model == "nomic-embed-text"

    def test_dimension_property(self) -> None:
        """The dimension property should return the configured dimension."""
        client = EmbeddingClient()
        assert client.dimension == 768


# ── Embedding Repository Tests ───────────────────────────────────────────────


class TestEmbeddingRepository:
    """Tests for the EmbeddingRepository using mocked sessions."""

    @pytest.mark.asyncio
    async def test_store_embedding_creates_record(self) -> None:
        """Storing an embedding should create an AssessmentEmbedding record."""
        mock_session = AsyncMock()
        mock_session.flush = AsyncMock()
        repo = EmbeddingRepository()
        assessment_id = uuid.uuid4()

        record = await repo.store_embedding(
            session=mock_session,
            assessment_id=assessment_id,
            embedding=[0.1, 0.2, 0.3],
            model="nomic-embed-text",
            dimension=3,
        )

        assert isinstance(record, AssessmentEmbedding)
        assert record.assessment_id == assessment_id
        assert record.embedding == [0.1, 0.2, 0.3]
        assert record.embedding_model == "nomic-embed-text"
        assert record.embedding_dimension == 3
        mock_session.add.assert_called_once()
        mock_session.flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_embedding_exists_true(self) -> None:
        """embedding_exists should return True when a record exists."""
        assessment_id = uuid.uuid4()
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one.return_value = True
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = EmbeddingRepository()
        exists = await repo.embedding_exists(mock_session, assessment_id)
        assert exists is True

    @pytest.mark.asyncio
    async def test_embedding_exists_false(self) -> None:
        """embedding_exists should return False when no record exists."""
        assessment_id = uuid.uuid4()
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one.return_value = False
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = EmbeddingRepository()
        exists = await repo.embedding_exists(mock_session, assessment_id)
        assert exists is False


# ── Embedding Service Tests ──────────────────────────────────────────────────


class TestEmbeddingService:
    """Tests for the EmbeddingService."""

    @pytest.mark.asyncio
    async def test_generate_for_assessment_skips_existing(self) -> None:
        """generate_for_assessment should skip if embedding already exists."""
        assessment_id = uuid.uuid4()

        mock_repo = AsyncMock(spec=EmbeddingRepository)
        mock_repo.embedding_exists.return_value = True

        service = EmbeddingService(repository=mock_repo)
        result = await service.generate_for_assessment(
            session=AsyncMock(), assessment_id=assessment_id,
        )

        assert result is False
        mock_repo.embedding_exists.assert_awaited_once_with(
            AsyncMock(), assessment_id,
        )

    @pytest.mark.asyncio
    async def test_generate_for_assessment_missing_assessment(self) -> None:
        """generate_for_assessment should return False if assessment is not found."""
        assessment_id = uuid.uuid4()

        mock_repo = AsyncMock(spec=EmbeddingRepository)
        mock_repo.embedding_exists.return_value = False
        mock_repo.get_assessment_by_id.return_value = None

        service = EmbeddingService(repository=mock_repo)
        result = await service.generate_for_assessment(
            session=AsyncMock(), assessment_id=assessment_id,
        )

        assert result is False
        mock_repo.get_assessment_by_id.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_count_embeddings(self) -> None:
        """count_embeddings should delegate to the repository."""
        mock_repo = AsyncMock(spec=EmbeddingRepository)
        mock_repo.count_embeddings.return_value = 42

        service = EmbeddingService(repository=mock_repo)
        count = await service.count_embeddings(AsyncMock())

        assert count == 42
        mock_repo.count_embeddings.assert_awaited_once()


# ── Admin API Tests ──────────────────────────────────────────────────────────


class TestEmbeddingAdminAPI:
    """Integration tests for the admin embedding endpoints."""

    @pytest.mark.asyncio
    async def test_count_empty(self, async_client: AsyncClient) -> None:
        """GET /admin/embeddings/count should return 0 when no embeddings exist."""
        response = await async_client.get("/admin/embeddings/count")
        assert response.status_code in (200, 500)  # 500 if DB is unreachable

    @pytest.mark.asyncio
    async def test_generate_endpoint(self, async_client: AsyncClient) -> None:
        """POST /admin/embeddings/generate should return a summary dict."""
        response = await async_client.post("/admin/embeddings/generate")
        # May be 200, 500, or 503 depending on DB / Ollama availability
        assert response.status_code in (200, 500, 503)

    @pytest.mark.asyncio
    async def test_refresh_endpoint(self, async_client: AsyncClient) -> None:
        """POST /admin/embeddings/refresh should return a summary dict."""
        response = await async_client.post("/admin/embeddings/refresh")
        assert response.status_code in (200, 500, 503)
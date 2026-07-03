"""
SQLAlchemy ORM models for the SHL AI Agent database.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.session import Base


class Assessment(Base):
    """Represents a single SHL assessment product from the catalog."""

    __tablename__ = "assessments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    entity_id: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(500), nullable=False)
    url: Mapped[str] = mapped_column(String(2048), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    duration: Mapped[str | None] = mapped_column(String(100), nullable=True)
    remote: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False,
    )
    adaptive: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False,
    )
    job_levels: Mapped[list] = mapped_column(
        JSONB, nullable=False, default=list,
    )
    languages: Mapped[list] = mapped_column(
        JSONB, nullable=False, default=list,
    )
    keys: Mapped[list] = mapped_column(
        JSONB, nullable=False, default=list,
    )
    scraped_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Relationship to embeddings
    embedding: Mapped[AssessmentEmbedding | None] = relationship(
        "AssessmentEmbedding",
        back_populates="assessment",
        uselist=False,
        cascade="all, delete-orphan",
    )


class AssessmentEmbedding(Base):
    """Stores a single pgvector embedding for each assessment."""

    __tablename__ = "assessment_embeddings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    assessment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("assessments.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True,
    )
    embedding: Mapped[list[float]] = mapped_column(
        Vector(3072),
        nullable=False,
    )
    embedding_model: Mapped[str] = mapped_column(
        String(100), nullable=False,
    )
    embedding_dimension: Mapped[int] = mapped_column(
        Integer, nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Relationship back to assessment
    assessment: Mapped[Assessment] = relationship(
        "Assessment",
        back_populates="embedding",
    )
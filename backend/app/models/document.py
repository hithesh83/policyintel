"""
SQLAlchemy ORM Models — Document & Upload
=========================================

``Document`` — metadata record for each ingested policy document.
``DocumentUpload`` — upload event record (immutable audit trail).

Both models use UUIDs as primary keys and include ``created_at`` /
``updated_at`` timestamps managed at the DB level.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""
    pass


# ---------------------------------------------------------------------------
# Enum types
# ---------------------------------------------------------------------------


class UploadStatus(str, PyEnum):
    """Processing lifecycle states for a document."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


class Document(Base):
    """
    Persistent record for each unique policy document.

    Uniqueness is enforced on ``file_hash`` (SHA-256) to prevent
    duplicate processing of the same PDF.
    """

    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("file_hash", name="uq_documents_file_hash"),
        UniqueConstraint("document_id", name="uq_documents_document_id"),
        Index("ix_documents_upload_status", "upload_status"),
        Index("ix_documents_scheme_name", "scheme_name"),
        Index("ix_documents_ministry", "ministry"),
        Index("ix_documents_created_at", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Internal surrogate key.",
    )
    document_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="Stable UUID5 derived from file_hash — used as external identifier.",
    )
    filename: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        comment="Original filename as uploaded.",
    )
    title: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        comment="Extracted or inferred document title.",
    )
    file_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="SHA-256 hex digest of the raw PDF bytes.",
    )
    file_size_bytes: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=0,
        comment="File size in bytes.",
    )
    total_pages: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Total page count from the parser.",
    )
    upload_status: Mapped[UploadStatus] = mapped_column(
        Enum(UploadStatus, name="upload_status_enum"),
        nullable=False,
        default=UploadStatus.PENDING,
        comment="Current processing lifecycle state.",
    )
    processing_error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Error message if upload_status=failed.",
    )

    # Extracted intelligence (backfilled after AI pipeline completes)
    scheme_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    ministry: Mapped[str | None] = mapped_column(String(256), nullable=True)
    policy_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    geographic_scope: Mapped[str | None] = mapped_column(String(64), nullable=True)
    effective_date: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # Graph & vector stats (for monitoring)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    node_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    relationship_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    vector_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    uploads: Mapped[list[DocumentUpload]] = relationship(
        "DocumentUpload",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<Document id={self.document_id} filename={self.filename!r} status={self.upload_status}>"


# ---------------------------------------------------------------------------
# DocumentUpload
# ---------------------------------------------------------------------------


class DocumentUpload(Base):
    """
    Immutable audit record for each file upload event.

    Multiple uploads may reference the same ``Document`` (re-upload scenario).
    """

    __tablename__ = "document_uploads"
    __table_args__ = (
        Index("ix_document_uploads_document_id", "document_id"),
        Index("ix_document_uploads_created_at", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    content_type: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        default="application/pdf",
    )
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    uploaded_by: Mapped[str | None] = mapped_column(
        String(256),
        nullable=True,
        comment="User identifier (future auth integration).",
    )
    pipeline_metadata: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
        comment="Arbitrary pipeline execution metadata (chunk counts, latencies, etc.).",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    processing_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    processing_completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Relationship
    document: Mapped[Document] = relationship(
        "Document",
        back_populates="uploads",
    )

    def __repr__(self) -> str:
        return f"<DocumentUpload id={self.id} filename={self.original_filename!r}>"

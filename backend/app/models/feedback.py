"""
SQLAlchemy ORM Model — Feedback
=================================

``Feedback`` stores user ratings and comments on document search results.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.document import Base


class Feedback(Base):
    """
    User feedback record for a query or a specific document result.

    Fields
    ------
    query_log_id :
        Optional FK to the QueryLog that triggered this feedback.
        (nullable — feedback can be submitted independently of a query)
    document_id :
        Optional ``Document.document_id`` the user is rating.
    rating :
        Integer 1–5 (1 = very unhelpful, 5 = very helpful).
    comment :
        Optional free-text comment.
    created_at :
        Submission timestamp.
    """

    __tablename__ = "feedback"
    __table_args__ = (
        CheckConstraint("rating BETWEEN 1 AND 5", name="ck_feedback_rating_range"),
        Index("ix_feedback_document_id", "document_id"),
        Index("ix_feedback_created_at", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    query_log_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="FK to query_logs.id (nullable — no CASCADE to preserve feedback on log deletion).",
    )
    document_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="document_id of the rated document (not a FK — document may be deleted).",
    )
    rating: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="1–5 rating scale.",
    )
    comment: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<Feedback id={self.id} rating={self.rating} doc={self.document_id!r}>"

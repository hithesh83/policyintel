"""
SQLAlchemy ORM Model — QueryLog
================================

``QueryLog`` records each search/query request for analytics and
debugging.  Stored in PostgreSQL; never deleted (append-only audit log).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    ARRAY,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.document import Base


class QueryLog(Base):
    """
    Immutable record of a search query issued against the system.

    Fields
    ------
    query_text :
        The raw query string submitted by the user.
    intent :
        Classified intent (e.g., eligibility_check, benefit_lookup, scheme_comparison).
    entities :
        JSON blob of extracted query entities (scheme names, states, etc.).
    latency_ms :
        End-to-end query latency in milliseconds.
    document_ids_returned :
        Array of document_ids surfaced in the response.
    created_at :
        Timestamp of the query request.
    """

    __tablename__ = "query_logs"
    __table_args__ = (
        Index("ix_query_logs_created_at", "created_at"),
        Index("ix_query_logs_intent", "intent"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    intent: Mapped[str | None] = mapped_column(String(64), nullable=True)
    entities: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    document_ids_returned: Mapped[list[str] | None] = mapped_column(
        ARRAY(String),
        nullable=True,
        comment="document_id values returned in the search response.",
    )
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        snippet = self.query_text[:50] if self.query_text else ""
        return f"<QueryLog id={self.id} intent={self.intent!r} q={snippet!r}>"

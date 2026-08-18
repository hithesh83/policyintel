"""
Document Repository
===================

``DocumentRepository`` manages CRUD operations on the ``Document`` and
``DocumentUpload`` ORM models via SQLAlchemy async sessions.

Pattern: Repository per aggregate root.  The repository accepts an
``AsyncSession`` at construction time (injected by the FastAPI DI layer).

All methods raise ``sqlalchemy.exc.SQLAlchemyError`` on DB errors — the
caller (ingestion endpoint) is responsible for catching and handling.

Usage::

    async with postgres_manager.session() as db:
        repo = DocumentRepository(db)
        doc = await repo.get_by_hash(file_hash)
        if doc is None:
            doc = await repo.create(parsed_doc, filename, file_hash, file_size)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentUpload, UploadStatus
from app.schemas.pipeline import ParsedDocument

logger = logging.getLogger(__name__)


class DocumentRepository:
    """
    Repository for ``Document`` and ``DocumentUpload`` persistence.

    Parameters
    ----------
    session :
        An open ``AsyncSession`` — must be committed/closed by the caller.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._db = session

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    async def create(
        self,
        parsed_doc: ParsedDocument,
        *,
        filename: str,
        file_hash: str,
        file_size_bytes: int,
    ) -> Document:
        """
        Persist a new ``Document`` record and its initial ``DocumentUpload``.

        Parameters
        ----------
        parsed_doc :
            Output of the ``PDFParser``.
        filename :
            Original upload filename.
        file_hash :
            SHA-256 hex digest of the raw PDF.
        file_size_bytes :
            Raw PDF byte length.

        Returns
        -------
        Document
            The newly created (flushed, not yet committed) document.
        """
        document = Document(
            document_id=parsed_doc.document_id,
            filename=filename,
            title=parsed_doc.title,
            file_hash=file_hash,
            file_size_bytes=file_size_bytes,
            total_pages=parsed_doc.total_pages,
            upload_status=UploadStatus.PENDING,
        )
        self._db.add(document)
        await self._db.flush()  # Get the DB-assigned PK without committing

        # Create the upload audit record
        upload = DocumentUpload(
            document_id=document.id,
            original_filename=filename,
            content_type="application/pdf",
            file_size_bytes=file_size_bytes,
            processing_started_at=datetime.now(timezone.utc),
        )
        self._db.add(upload)
        await self._db.flush()

        logger.info(
            "DocumentRepository.create() | document_id=%s | filename=%s",
            document.document_id,
            filename,
        )
        return document

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get_by_id(self, document_id: str) -> Document | None:
        """Return the Document with the given ``document_id`` or None."""
        result = await self._db.execute(
            select(Document).where(Document.document_id == document_id)
        )
        return result.scalar_one_or_none()

    async def get_by_hash(self, file_hash: str) -> Document | None:
        """
        Look up a Document by its SHA-256 file hash.

        Used for duplicate detection before running the expensive pipeline.
        """
        result = await self._db.execute(
            select(Document).where(Document.file_hash == file_hash)
        )
        return result.scalar_one_or_none()

    async def list_documents(
        self,
        *,
        skip: int = 0,
        limit: int = 20,
        status: UploadStatus | None = None,
    ) -> list[Document]:
        """
        Return a paginated list of documents, optionally filtered by status.
        """
        stmt = select(Document).order_by(Document.created_at.desc())
        if status is not None:
            stmt = stmt.where(Document.upload_status == status)
        stmt = stmt.offset(skip).limit(limit)
        result = await self._db.execute(stmt)
        return list(result.scalars().all())

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    async def update_status(
        self,
        document_id: str,
        status: UploadStatus,
        *,
        error: str | None = None,
    ) -> None:
        """
        Update the processing status of a document.

        Parameters
        ----------
        document_id :
            The ``Document.document_id`` (not the surrogate UUID pk).
        status :
            New ``UploadStatus`` value.
        error :
            Error message to store if ``status == UploadStatus.FAILED``.
        """
        values: dict = {"upload_status": status}
        if error is not None:
            values["processing_error"] = error[:2000]  # Truncate long stack traces

        await self._db.execute(
            update(Document)
            .where(Document.document_id == document_id)
            .values(**values)
        )
        logger.info(
            "DocumentRepository.update_status() | document_id=%s | status=%s",
            document_id,
            status,
        )

    async def update_metadata(
        self,
        document_id: str,
        *,
        scheme_name: str | None = None,
        ministry: str | None = None,
        policy_type: str | None = None,
        geographic_scope: str | None = None,
        effective_date: str | None = None,
        chunk_count: int = 0,
        node_count: int = 0,
        relationship_count: int = 0,
        vector_count: int = 0,
    ) -> None:
        """
        Backfill extracted intelligence metadata after the AI pipeline completes.
        """
        values: dict = {
            "upload_status": UploadStatus.COMPLETED,
            "chunk_count": chunk_count,
            "node_count": node_count,
            "relationship_count": relationship_count,
            "vector_count": vector_count,
        }
        if scheme_name:
            values["scheme_name"] = scheme_name[:256]
        if ministry:
            values["ministry"] = ministry[:256]
        if policy_type:
            values["policy_type"] = policy_type[:64]
        if geographic_scope:
            values["geographic_scope"] = geographic_scope[:64]
        if effective_date:
            values["effective_date"] = effective_date[:32]

        await self._db.execute(
            update(Document)
            .where(Document.document_id == document_id)
            .values(**values)
        )
        logger.info(
            "DocumentRepository.update_metadata() | document_id=%s | scheme=%s | chunks=%d",
            document_id,
            scheme_name,
            chunk_count,
        )

    async def complete_upload(
        self,
        document_id: str,
        *,
        pipeline_metadata: dict | None = None,
    ) -> None:
        """
        Mark the most recent DocumentUpload as completed.
        """
        doc = await self.get_by_id(document_id)
        if doc and doc.uploads:
            upload = doc.uploads[-1]
            upload.processing_completed_at = datetime.now(timezone.utc)
            if pipeline_metadata:
                upload.pipeline_metadata = pipeline_metadata
            await self._db.flush()

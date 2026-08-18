"""
Document Ingestion Endpoint
===========================

``POST /api/v1/ingest/upload``

Full pipeline:
    Upload PDF → Parse → Chunk → Extract → Graph Build → Index → Persist

``GET /api/v1/ingest/status/{document_id}``

    Return current processing status from PostgreSQL.

``GET /api/v1/ingest/documents``

    Return paginated list of ingested documents.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.db.neo4j import neo4j_manager
from app.core.db.postgres import get_db_session
from app.core.db.qdrant import qdrant_manager
from app.models.document import UploadStatus
from app.pipeline.chunker import DocumentChunker
from app.pipeline.extractor import PolicyExtractor
from app.pipeline.graph_builder import GraphBuilder
from app.pipeline.indexer import DocumentIndexer
from app.pipeline.parser import PDFParser
from app.repositories.document_repository import DocumentRepository
from app.repositories.neo4j_repository import Neo4jRepository
from app.repositories.qdrant_repository import QdrantRepository
from app.services.ai_service import AIService
from app.llm.dependency import get_ai_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
_ALLOWED_CONTENT_TYPES = {"application/pdf", "application/x-pdf"}


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class IngestionResponse(BaseModel):
    """Successful ingestion response."""

    document_id: str = Field(..., description="Stable document identifier.")
    status: str = Field(..., description="Processing status.")
    filename: str
    total_pages: int
    chunk_count: int = 0
    node_count: int = 0
    relationship_count: int = 0
    vector_count: int = 0
    latency_ms: int = 0
    message: str = ""


class DuplicateResponse(BaseModel):
    """Response when the document was already ingested."""

    document_id: str
    status: str = "duplicate"
    message: str
    existing_filename: str | None = None


class DocumentStatusResponse(BaseModel):
    """Document processing status response."""

    document_id: str
    filename: str
    status: str
    title: str | None = None
    scheme_name: str | None = None
    ministry: str | None = None
    chunk_count: int = 0
    node_count: int = 0
    vector_count: int = 0
    created_at: str | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Pipeline factory helpers
# ---------------------------------------------------------------------------


def _make_parser() -> PDFParser:
    return PDFParser(prefer_pymupdf=True)


def _make_pipeline(ai_service: AIService):
    return (
        DocumentChunker(ai_service, enrich_chunks=True),
        PolicyExtractor(ai_service, max_concurrent_chunks=3),
        GraphBuilder(),
        DocumentIndexer(ai_service, max_concurrent_embeddings=5),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/upload",
    summary="Upload and ingest a policy PDF",
    status_code=status.HTTP_201_CREATED,
    response_model=IngestionResponse,
    responses={
        409: {"description": "Document already ingested (duplicate hash)"},
        422: {"description": "Unsupported file type or file too large"},
    },
)
async def upload_document(
    file: UploadFile = File(..., description="PDF file to ingest (max 50 MB)"),
    db: AsyncSession = Depends(get_db_session),
    ai_service: AIService = Depends(get_ai_service),
) -> IngestionResponse:
    """
    Upload a PDF policy document and run the full AI pipeline.

    Workflow
    --------
    1. Validate file type and size.
    2. Read bytes and compute SHA-256 hash.
    3. Check for duplicates in PostgreSQL.
    4. Parse PDF → ``ParsedDocument``.
    5. Run AI pipeline: Chunk → Extract → Graph Build → Index.
    6. Persist concurrently to Neo4j, Qdrant, and PostgreSQL.
    7. Return ``IngestionResponse``.
    """
    start_time = time.monotonic()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    content_type = file.content_type or ""
    filename_lower = (file.filename or "").lower()
    if content_type not in _ALLOWED_CONTENT_TYPES and not filename_lower.endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type: '{content_type}'. Only PDF files are accepted.",
        )

    pdf_bytes = await file.read()
    file_size = len(pdf_bytes)

    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is empty.",
        )
    if file_size > _MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"File too large: {file_size:,} bytes. Maximum is 50 MB.",
        )

    # ------------------------------------------------------------------
    # Duplicate detection
    # ------------------------------------------------------------------
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()
    doc_repo = DocumentRepository(db)

    existing = await doc_repo.get_by_hash(file_hash)
    if existing:
        logger.info(
            "Ingestion duplicate detected | document_id=%s | filename=%s",
            existing.document_id,
            file.filename,
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "document_id": existing.document_id,
                "status": "duplicate",
                "message": "This document has already been ingested.",
                "existing_filename": existing.filename,
            },
        )

    # ------------------------------------------------------------------
    # Parse PDF
    # ------------------------------------------------------------------
    logger.info("Ingestion START | filename=%s | size=%d bytes", file.filename, file_size)

    parser = _make_parser()
    try:
        parsed_doc = await parser.parse_bytes(pdf_bytes, filename=file.filename or "upload.pdf")
    except Exception as exc:
        logger.error("PDF parsing failed | filename=%s | error=%s", file.filename, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to parse PDF: {exc}",
        )

    # ------------------------------------------------------------------
    # Create initial DB record
    # ------------------------------------------------------------------
    doc_record = await doc_repo.create(
        parsed_doc,
        filename=file.filename or "upload.pdf",
        file_hash=file_hash,
        file_size_bytes=file_size,
    )
    await db.commit()  # Commit the initial record before long pipeline

    await doc_repo.update_status(parsed_doc.document_id, UploadStatus.PROCESSING)
    await db.commit()

    # ------------------------------------------------------------------
    # AI Pipeline
    # ------------------------------------------------------------------
    try:
        chunker, extractor, graph_builder, indexer = _make_pipeline(ai_service)

        logger.info(
            "Pipeline START | document_id=%s | pages=%d",
            parsed_doc.document_id,
            parsed_doc.total_pages,
        )

        chunks = await chunker.chunk(parsed_doc)
        logger.info("Chunker DONE | chunks=%d", len(chunks))

        results = await extractor.extract(chunks)
        logger.info("Extractor DONE | results=%d", len(results))

        bundle = graph_builder.build(results, document_id=parsed_doc.document_id)
        logger.info(
            "GraphBuilder DONE | nodes=%d | rels=%d",
            bundle.node_count,
            bundle.relationship_count,
        )

        vector_docs = await indexer.index(chunks, results)
        logger.info("Indexer DONE | vectors=%d", len(vector_docs))

    except Exception as exc:
        logger.exception(
            "AI Pipeline failed | document_id=%s | error=%s",
            parsed_doc.document_id,
            exc,
        )
        await doc_repo.update_status(
            parsed_doc.document_id,
            UploadStatus.FAILED,
            error=str(exc),
        )
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI pipeline failed: {exc}",
        )

    # ------------------------------------------------------------------
    # Parallel persistence: Neo4j + Qdrant + PostgreSQL metadata
    # ------------------------------------------------------------------
    try:
        # Extract best scheme metadata from results
        scheme_name: str | None = None
        ministry: str | None = None
        policy_type: str | None = None
        geographic_scope: str | None = None
        effective_date: str | None = None
        for r in results:
            if r.entities.scheme_name and not scheme_name:
                scheme_name = r.entities.scheme_name
            if r.entities.ministry and not ministry:
                ministry = r.entities.ministry
            if r.entities.policy_type and not policy_type:
                policy_type = r.entities.policy_type
            if r.entities.geographic_scope and not geographic_scope:
                geographic_scope = r.entities.geographic_scope
            if r.entities.effective_date and not effective_date:
                effective_date = r.entities.effective_date

        # Neo4j upsert
        async def persist_neo4j():
            async with neo4j_manager.session() as neo4j_session:
                neo4j_repo = Neo4jRepository(neo4j_session)
                await neo4j_repo.upsert_graph_bundle(bundle)

        # Qdrant upsert
        async def persist_qdrant():
            qdrant_client = qdrant_manager.get_client()
            qdrant_repo = QdrantRepository(qdrant_client, settings.qdrant_collection_name)
            await qdrant_repo.upsert_documents(vector_docs)

        # PostgreSQL metadata update
        async def persist_postgres():
            await doc_repo.update_metadata(
                parsed_doc.document_id,
                scheme_name=scheme_name,
                ministry=ministry,
                policy_type=policy_type,
                geographic_scope=geographic_scope,
                effective_date=effective_date,
                chunk_count=len(chunks),
                node_count=bundle.node_count,
                relationship_count=bundle.relationship_count,
                vector_count=len(vector_docs),
            )
            await doc_repo.complete_upload(
                parsed_doc.document_id,
                pipeline_metadata={
                    "chunk_count": len(chunks),
                    "node_count": bundle.node_count,
                    "relationship_count": bundle.relationship_count,
                    "vector_count": len(vector_docs),
                    "scheme_name": scheme_name,
                    "extraction_errors": sum(1 for r in results if r.extraction_error),
                },
            )
            await db.commit()

        await asyncio.gather(
            persist_neo4j(),
            persist_qdrant(),
            persist_postgres(),
        )

        logger.info(
            "Persistence DONE | document_id=%s | neo4j=%d nodes | qdrant=%d vectors | pg=updated",
            parsed_doc.document_id,
            bundle.node_count,
            len(vector_docs),
        )

    except Exception as exc:
        logger.exception(
            "Persistence failed | document_id=%s | error=%s",
            parsed_doc.document_id,
            exc,
        )
        await doc_repo.update_status(
            parsed_doc.document_id,
            UploadStatus.FAILED,
            error=f"Persistence error: {exc}",
        )
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database persistence failed: {exc}",
        )

    # ------------------------------------------------------------------
    # Response
    # ------------------------------------------------------------------
    latency_ms = int((time.monotonic() - start_time) * 1000)
    logger.info(
        "Ingestion COMPLETE | document_id=%s | latency_ms=%d",
        parsed_doc.document_id,
        latency_ms,
    )

    return IngestionResponse(
        document_id=parsed_doc.document_id,
        status="completed",
        filename=file.filename or "upload.pdf",
        total_pages=parsed_doc.total_pages,
        chunk_count=len(chunks),
        node_count=bundle.node_count,
        relationship_count=bundle.relationship_count,
        vector_count=len(vector_docs),
        latency_ms=latency_ms,
        message=f"Successfully ingested '{file.filename}' — {len(chunks)} chunks, "
                f"{bundle.node_count} graph nodes, {len(vector_docs)} vectors.",
    )


@router.get(
    "/status/{document_id}",
    summary="Get document processing status",
    response_model=DocumentStatusResponse,
)
async def get_document_status(
    document_id: str,
    db: AsyncSession = Depends(get_db_session),
) -> DocumentStatusResponse:
    """Return the current processing status and metadata for a document."""
    repo = DocumentRepository(db)
    doc = await repo.get_by_id(document_id)

    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    return DocumentStatusResponse(
        document_id=doc.document_id,
        filename=doc.filename,
        status=doc.upload_status.value,
        title=doc.title,
        scheme_name=doc.scheme_name,
        ministry=doc.ministry,
        chunk_count=doc.chunk_count,
        node_count=doc.node_count,
        vector_count=doc.vector_count,
        created_at=doc.created_at.isoformat() if doc.created_at else None,
        error=doc.processing_error,
    )


@router.get(
    "/documents",
    summary="List ingested documents",
    response_model=list[DocumentStatusResponse],
)
async def list_documents(
    skip: int = Query(default=0, ge=0, description="Pagination offset."),
    limit: int = Query(default=20, ge=1, le=100, description="Page size."),
    db: AsyncSession = Depends(get_db_session),
) -> list[DocumentStatusResponse]:
    """Return a paginated list of ingested documents."""
    repo = DocumentRepository(db)
    docs = await repo.list_documents(skip=skip, limit=limit)

    return [
        DocumentStatusResponse(
            document_id=d.document_id,
            filename=d.filename,
            status=d.upload_status.value,
            title=d.title,
            scheme_name=d.scheme_name,
            ministry=d.ministry,
            chunk_count=d.chunk_count,
            node_count=d.node_count,
            vector_count=d.vector_count,
            created_at=d.created_at.isoformat() if d.created_at else None,
            error=d.processing_error,
        )
        for d in docs
    ]

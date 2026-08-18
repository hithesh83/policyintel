"""
Document Indexer
================

``DocumentIndexer`` prepares ``VectorDocument`` objects from
``DocumentChunk`` and ``ExtractionResult`` data, ready for upsert into a
Qdrant vector collection.

Responsibilities
----------------
1. **Embedding generation** — calls ``AIService.generate()`` with a
   semantically rich representation prompt to get the model's internal
   representation of each chunk.

   .. note::
       In Week 2, Qdrant is not yet connected.  The indexer generates a
       **placeholder vector** (list of zeros with dimension 1024) so the
       ``VectorDocument`` schema is fully specified and Week 3 can swap in
       a real embedding model (e.g., ``nomic-embed-text``,
       ``sentence-transformers``) without touching any consumer code.

2. **Payload assembly** — flattens chunk + extraction data into a Qdrant
   payload dict suitable for metadata filtering.  Kept intentionally flat
   for efficient Qdrant index usage.

3. **No Qdrant I/O** — returns ``list[VectorDocument]``; the actual upsert
   call is the responsibility of the Week 3 writer module.

Concurrency
-----------
Embedding calls run concurrently via ``asyncio.gather`` with a semaphore
bounding simultaneous calls (default: 5).

Error Isolation
---------------
If embedding generation fails for a chunk, the indexer logs a warning and
falls back to the placeholder zero vector rather than failing the entire
batch.

Usage
-----
::

    from app.pipeline.indexer import DocumentIndexer

    indexer = DocumentIndexer(ai_service=ai_service)
    vector_docs = await indexer.index(chunks=chunks, results=results)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from app.llm.exceptions import LLMError
from app.schemas.pipeline import (
    DocumentChunk,
    ExtractionResult,
    VectorDocument,
)
from app.services.ai_service import AIService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Placeholder embedding dimension — matches ``nomic-embed-text:v1.5`` (Week 3).
# Changing this requires a full Qdrant collection recreation.
_EMBEDDING_DIM = 1024

# Maximum characters of chunk text used in the embedding prompt
_EMBED_TEXT_LIMIT = 1500


def _placeholder_vector() -> list[float]:
    """Return a zero vector of the configured embedding dimension."""
    return [0.0] * _EMBEDDING_DIM


def _build_embedding_prompt(chunk: DocumentChunk, extraction: ExtractionResult | None) -> str:
    """
    Build a semantically rich text representation for embedding.

    Combines the chunk text with key extracted metadata so the embedding
    captures both the literal content and the structured semantics.  This
    technique (called "enriched embedding" or "context-aware embedding")
    significantly improves retrieval precision.
    """
    parts: list[str] = []

    # Title / section context
    if chunk.title:
        parts.append(f"Title: {chunk.title}")
    if chunk.section:
        parts.append(f"Section: {chunk.section}")

    # Extracted metadata enrichment
    if extraction:
        e = extraction.entities
        if e.scheme_name:
            parts.append(f"Scheme: {e.scheme_name}")
        # Organisational context — helps match queries like
        # "Which ministry issued X?" and "Which bank supports Y?"
        if e.issuing_ministry:
            parts.append(f"Issuing Ministry: {e.issuing_ministry}")
        if e.implementing_organizations:
            parts.append(
                f"Implementing Organizations: {', '.join(e.implementing_organizations[:5])}"
            )
        if e.departments:
            parts.append(f"Departments: {', '.join(e.departments[:5])}")
        if e.supporting_agencies:
            parts.append(f"Supporting Agencies: {', '.join(e.supporting_agencies[:5])}")
        if e.states:
            parts.append(f"States: {', '.join(e.states[:5])}")
        if e.beneficiary_categories:
            parts.append(f"Beneficiaries: {', '.join(e.beneficiary_categories[:5])}")
        if extraction.entities.eligibility_criteria:
            descs = [c.description for c in e.eligibility_criteria[:3]]
            parts.append(f"Eligibility: {'; '.join(descs)}")
        if e.benefits:
            benefit_descs = [b.description for b in e.benefits[:3]]
            parts.append(f"Benefits: {'; '.join(benefit_descs)}")

    # Chunk metadata summary
    if chunk.metadata.topic:
        parts.append(f"Topic: {chunk.metadata.topic}")
    if chunk.metadata.summary:
        parts.append(f"Summary: {chunk.metadata.summary}")

    # Raw text (truncated)
    parts.append(f"\nContent:\n{chunk.text[:_EMBED_TEXT_LIMIT]}")

    return "\n".join(parts)


def _build_payload(
    chunk: DocumentChunk,
    extraction: ExtractionResult | None,
) -> dict[str, Any]:
    """
    Build a flat Qdrant payload dict from chunk and extraction data.

    All values are JSON-serialisable primitives or lists of primitives.
    No nested dicts — Qdrant's filtering engine works best on flat structures.
    """
    payload: dict[str, Any] = {
        # Chunk identity
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "parent_id": chunk.parent_id,
        "hierarchy_level": chunk.hierarchy_level.value,
        "page_number": chunk.page_number,
        # Structure
        "title": chunk.title,
        "section": chunk.section,
        # Chunk metadata
        "content_type": chunk.metadata.content_type,
        "topic": chunk.metadata.topic,
        "summary": chunk.metadata.summary,
        "has_eligibility_criteria": chunk.metadata.has_eligibility_criteria,
        "has_procedure_steps": chunk.metadata.has_procedure_steps,
        "key_entities": chunk.metadata.key_entities,
        "key_dates": chunk.metadata.key_dates,
        "key_amounts": chunk.metadata.key_amounts,
        "word_count": chunk.metadata.word_count,
        "char_count": chunk.metadata.char_count,
    }

    # Add extracted entities if available
    if extraction and not extraction.extraction_error:
        e = extraction.entities
        payload.update(
            {
                "scheme_name": e.scheme_name,
                # Organisational fields — all searchable as Qdrant payload filters
                "issuing_ministry": e.issuing_ministry,
                "implementing_organizations": e.implementing_organizations,
                "supporting_agencies": e.supporting_agencies,
                "departments": e.departments,
                "stakeholders": e.stakeholders,
                "funding_pattern": e.funding_pattern,
                # Other policy metadata
                "policy_type": e.policy_type,
                "geographic_scope": e.geographic_scope,
                "states": e.states,
                "effective_date": e.effective_date,
                "issue_date": e.issue_date,
                "beneficiary_categories": e.beneficiary_categories,
                "eligible_categories": e.eligible_categories,
                "income_limit_annual": e.income_limit_annual,
                "age_min": e.age_min,
                "age_max": e.age_max,
                "is_direct_benefit_transfer": e.is_direct_benefit_transfer,
                "total_annual_benefit_inr": e.total_annual_benefit_inr,
                "benefit_types": [b.benefit_type for b in e.benefits],
                "benefit_amounts_inr": [
                    b.amount_inr for b in e.benefits if b.amount_inr is not None
                ],
                "deadlines": e.deadlines,
                "documents_required": e.documents_required,
                "has_amendment": bool(e.amendment_references),
            }
        )

    # Remove None values — Qdrant ignores None-valued payload fields anyway
    return {k: v for k, v in payload.items() if v is not None}


# ---------------------------------------------------------------------------
# DocumentIndexer
# ---------------------------------------------------------------------------


class DocumentIndexer:
    """
    Prepares ``VectorDocument`` objects from chunks and extraction results.

    Parameters
    ----------
    ai_service :
        The application-level AI abstraction.  Used only for the embedding
        representation prompt generation.
    max_concurrent_embeddings :
        Maximum concurrent embedding calls.
    """

    def __init__(
        self,
        ai_service: AIService,
        *,
        max_concurrent_embeddings: int = 5,
    ) -> None:
        self._ai = ai_service
        self._semaphore = asyncio.Semaphore(max_concurrent_embeddings)
        logger.info(
            "DocumentIndexer initialised | embedding_dim=%d | max_concurrent=%d",
            _EMBEDDING_DIM,
            max_concurrent_embeddings,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def index(
        self,
        chunks: list[DocumentChunk],
        results: list[ExtractionResult],
    ) -> list[VectorDocument]:
        """
        Prepare ``VectorDocument`` objects for all chunks concurrently.

        Parameters
        ----------
        chunks :
            ``DocumentChunk`` list from the chunker.
        results :
            ``ExtractionResult`` list from the extractor.  Must be in the
            same order as *chunks* (1-to-1 mapping by ``chunk_id``).

        Returns
        -------
        list[VectorDocument]
            One ``VectorDocument`` per input chunk, in the same order.
        """
        logger.info(
            "DocumentIndexer.index() START | chunks=%d | results=%d",
            len(chunks),
            len(results),
        )
        start = time.monotonic()

        # Build lookup: chunk_id → ExtractionResult
        result_map: dict[str, ExtractionResult] = {r.chunk_id: r for r in results}

        vector_docs = await asyncio.gather(
            *[
                self._index_chunk(chunk, result_map.get(chunk.chunk_id))
                for chunk in chunks
            ],
        )

        elapsed_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "DocumentIndexer.index() DONE | vector_docs=%d | latency_ms=%d",
            len(vector_docs),
            elapsed_ms,
        )
        return list(vector_docs)

    # ------------------------------------------------------------------
    # Per-Chunk Indexing
    # ------------------------------------------------------------------

    async def _index_chunk(
        self,
        chunk: DocumentChunk,
        extraction: ExtractionResult | None,
    ) -> VectorDocument:
        """
        Generate embedding and assemble the VectorDocument for one chunk.

        Falls back to a placeholder zero vector if the embedding call fails.
        """
        async with self._semaphore:
            logger.debug(
                "DocumentIndexer._index_chunk() | chunk_id=%s | level=%s",
                chunk.chunk_id,
                chunk.hierarchy_level,
            )

            vector = await self._embed(chunk, extraction)
            payload = _build_payload(chunk, extraction)

            return VectorDocument(
                vector_id=chunk.chunk_id,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                vector=vector,
                payload=payload,
                text=chunk.text,
            )

    async def _embed(
        self,
        chunk: DocumentChunk,
        extraction: ExtractionResult | None,
    ) -> list[float]:
        """
        Generate an embedding vector for the chunk.

        Week 2 implementation: uses the LLM to generate a condensed
        representation text, then returns a placeholder zero vector of the
        target dimension.

        Week 3 upgrade path: replace the body of this method with a call to
        a dedicated embedding model (e.g. ``nomic-embed-text``) and return
        the real float vector.

        The method signature and return type must NOT change.
        """
        try:
            embed_prompt = _build_embedding_prompt(chunk, extraction)

            # Week 2 stub: the LLM call validates the embedding pipeline end-to-end.
            # Week 3: replace with real embedding model call.
            # The generate() call here ensures AIService integration is exercised.
            _ = await self._ai.generate(
                f"Represent this policy document chunk for semantic search:\n\n{embed_prompt}",
                temperature=0.0,
                max_tokens=64,  # We only need the call to succeed, not the text
            )

            # TODO (Week 3): extract float vector from embedding model response
            # and return it instead of the placeholder.
            return _placeholder_vector()

        except LLMError as exc:
            logger.warning(
                "DocumentIndexer: embedding failed | chunk_id=%s | error=%s — "
                "falling back to placeholder vector",
                chunk.chunk_id,
                exc,
            )
            return _placeholder_vector()

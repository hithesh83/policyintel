"""
Document Chunker
================

``DocumentChunker`` transforms a ``ParsedDocument`` (parser.py output) into
an ordered, hierarchical list of ``DocumentChunk`` objects.

Hierarchy produced
------------------
DOCUMENT  (one per document — always present)
  └── CHAPTER  (detected from numbered headings like "Chapter 1", "PART I")
        └── SECTION  (detected from decimal-numbered headings like "3.2")
              └── CLAUSE  (detected from clause markers like "(a)", "1.", "i.")
                    └── PARAGRAPH  (remaining prose blocks)

Chunk IDs
---------
Each ``chunk_id`` is a UUID5 derived from ``(document_id, sha256_prefix(text))``
so the same text always produces the same ID — pipeline re-runs are idempotent
and safe to upsert into databases.

AI Enrichment
-------------
After structural splitting, the chunker calls
``AIService.generate_chunk_description()`` **concurrently** (via
``asyncio.gather``) to enrich chunk metadata with:
- ``topic`` — 5–10 word label
- ``content_type`` — semantic type enum
- ``key_entities`` / ``key_dates`` / ``key_amounts``
- ``summary`` — 1–2 sentence summary

This enrichment is best-effort: if the LLM call fails for a chunk the
chunker logs a warning and leaves the metadata fields as ``None`` / empty
lists rather than propagating the error.

Usage
-----
::

    from app.pipeline.chunker import DocumentChunker
    from app.services.ai_service import AIService

    chunker = DocumentChunker(ai_service=ai_service)
    chunks = await chunker.chunk(parsed_doc)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import uuid
from typing import Sequence

from app.llm.exceptions import LLMError
from app.schemas.pipeline import (
    ChunkMetadata,
    DocumentChunk,
    HierarchyLevel,
    ParsedDocument,
    ParsedPage,
)
from app.services.ai_service import AIService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class ChunkEnrichmentError(ValueError):
    """
    Raised when AI-returned metadata cannot be normalised into a valid
    ``ChunkMetadata`` even after sanitisation.

    Carries structured context so callers can log or report the exact failure
    without relying solely on a generic Pydantic ValidationError.

    Attributes
    ----------
    document_id : str
    chunk_id : str
    page_number : int
    field : str
        Name of the ``ChunkMetadata`` field that failed validation.
    offending_value : object
        The raw (post-normalisation) value that Pydantic rejected.
    """

    def __init__(
        self,
        message: str,
        *,
        document_id: str,
        chunk_id: str,
        page_number: int,
        field: str,
        offending_value: object,
    ) -> None:
        super().__init__(message)
        self.document_id    = document_id
        self.chunk_id       = chunk_id
        self.page_number    = page_number
        self.field          = field
        self.offending_value = offending_value

    def __str__(self) -> str:
        return (
            f"ChunkEnrichmentError: {self.args[0]} | "
            f"document_id={self.document_id!r} | "
            f"chunk_id={self.chunk_id!r} | "
            f"page={self.page_number} | "
            f"field={self.field!r} | "
            f"offending_value={self.offending_value!r}"
        )

# ---------------------------------------------------------------------------
# Regex patterns for structural detection
# ---------------------------------------------------------------------------

# Chapter / Part headings  (e.g. "Chapter 1", "CHAPTER I", "PART III", "Part A")
_CHAPTER_RE = re.compile(
    r"^(?:CHAPTER|Chapter|PART|Part)\s+(?:[IVXivx]+|\d+|[A-Z])\b",
    re.MULTILINE,
)

# Section headings with decimal notation  (e.g. "3.2", "10.1.3", "Section 4")
_SECTION_RE = re.compile(
    r"^(?:Section\s+\d+(?:\.\d+)*|\d{1,2}(?:\.\d+){1,3})\s+\S",
    re.MULTILINE,
)

# Clause markers  (e.g. "(a)", "(i)", "1.", "a)", "i)")
_CLAUSE_RE = re.compile(
    r"^(?:\([a-z]{1,3}\)|\([ivxIVX]{1,5}\)|\d{1,2}\.|[a-z]\))\s+\S",
    re.MULTILINE,
)

# Paragraph separator — two or more newlines
_PARA_SPLIT_RE = re.compile(r"\n{2,}")

# Minimum character threshold below which a chunk is merged with its parent
_MIN_CHUNK_CHARS = 80


def _make_chunk_id(document_id: str, text: str) -> str:
    """
    Derive a stable UUID5 from ``(document_id, sha256_hex_prefix(text))``.

    Using the first 64 chars of the SHA-256 hex digest as the name component
    ensures:
    - Different texts produce different IDs.
    - The same text always produces the same ID (idempotent).
    - The full UUID namespace is used (UUID5 is deterministic by design).
    """
    text_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:64]
    name = f"{document_id}:{text_hash}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))


def _make_rel_id(source_id: str, target_id: str, rel_type: str) -> str:
    """Stable UUID5 for a graph relationship."""
    name = f"{rel_type}:{source_id}:{target_id}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))


# ---------------------------------------------------------------------------
# Normalisation utilities
# ---------------------------------------------------------------------------


def _normalize_str_list(raw: object, *, field: str) -> list[str]:
    """
    Coerce an AI-returned list field into a clean ``list[str]``.

    Normalisation rules (applied in order):
    1. If *raw* is not a list (e.g. ``None``, a stray string), return ``[]``.
    2. For each item in the list:
       a. Skip ``None``.
       b. Skip empty strings and whitespace-only strings after stripping.
       c. Convert ``int`` / ``float`` values to their string representations
          (e.g. ``6000`` → ``"6000"``).  These arise when the LLM omits
          currency symbols but returns a valid numeric amount.
       d. Reject any other non-string type with a ``TypeError`` so the caller
          can surface a ``ChunkEnrichmentError`` rather than a cryptic
          Pydantic ``ValidationError``.
       e. Strip leading/trailing whitespace from valid strings.
    3. Deduplicate while preserving original insertion order.

    Parameters
    ----------
    raw :
        Raw value from ``response.data.get(field)``.
    field :
        Field name — included in error messages for traceability.

    Returns
    -------
    list[str]
        Sanitised, deduplicated list of non-empty strings.

    Raises
    ------
    TypeError
        If an item is a non-primitive type (dict, list, etc.) that cannot
        be meaningfully coerced to a string.
    """
    if not isinstance(raw, list):
        return []

    seen:   set[str]  = set()
    result: list[str] = []

    for item in raw:
        if item is None:
            continue
        if isinstance(item, bool):
            # bool is a subclass of int — treat as invalid for string lists
            raise TypeError(
                f"Field '{field}': boolean value {item!r} is not a valid string list item"
            )
        if isinstance(item, (int, float)):
            # Numerics from LLM (e.g. amount without currency symbol)
            coerced = str(item)
        elif isinstance(item, str):
            coerced = item.strip()
        else:
            raise TypeError(
                f"Field '{field}': unexpected type {type(item).__name__!r} "
                f"for value {item!r} — expected str, int, float, or None"
            )

        if not coerced:          # skip empty / whitespace-only
            continue
        if coerced in seen:      # skip duplicates
            continue

        seen.add(coerced)
        result.append(coerced)

    return result


def _page_for_offset(pages: list[ParsedPage], char_offset: int) -> int:
    """
    Estimate the page number for a character offset in the concatenated text.

    Iterates over pages accumulating character counts until the offset is
    passed.  Returns the last page number if the offset exceeds total length.
    """
    running = 0
    for page in pages:
        running += len(page.text)
        if char_offset <= running:
            return page.page_number
    return pages[-1].page_number if pages else 1


def _extract_title_from_block(text: str) -> str | None:
    """
    Extract the first non-empty line of a text block as a title candidate.

    Returns ``None`` if the first line is longer than 120 chars (likely prose,
    not a heading).
    """
    first_line = text.strip().split("\n")[0].strip()
    if first_line and len(first_line) <= 120:
        return first_line
    return None


def _extract_section_number(text: str) -> str | None:
    """Extract leading section/clause number from text (e.g. '3.2', '(a)')."""
    match = re.match(r"^(\d{1,2}(?:\.\d+)*|\([a-z]{1,3}\)|\([ivxIVX]{1,5}\))\s", text.strip())
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# DocumentChunker
# ---------------------------------------------------------------------------


class DocumentChunker:
    """
    Transforms a ``ParsedDocument`` into a hierarchical list of
    ``DocumentChunk`` objects.

    Parameters
    ----------
    ai_service :
        The application-level AI abstraction.  Used only for chunk description
        generation — never called directly against an LLM provider.
    enrich_chunks :
        If ``True`` (default), calls ``AIService.generate_chunk_description()``
        concurrently to populate metadata fields.  Set to ``False`` in tests
        or batch scenarios where LLM cost matters.
    max_concurrent_enrichments :
        Limits concurrent LLM calls during metadata enrichment to avoid
        overwhelming the backend.
    """

    def __init__(
        self,
        ai_service: AIService,
        *,
        enrich_chunks: bool = True,
        max_concurrent_enrichments: int = 5,
    ) -> None:
        self._ai = ai_service
        self._enrich = enrich_chunks
        self._semaphore = asyncio.Semaphore(max_concurrent_enrichments)
        logger.info(
            "DocumentChunker initialised | enrich=%s | max_concurrent=%d",
            enrich_chunks,
            max_concurrent_enrichments,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def chunk(self, doc: ParsedDocument) -> list[DocumentChunk]:
        """
        Chunk a parsed document into a hierarchical list of ``DocumentChunk``
        objects.

        The list is ordered depth-first: DOCUMENT → CHAPTER(s) → SECTION(s)
        → CLAUSE(s) → PARAGRAPH(s).

        Parameters
        ----------
        doc :
            A ``ParsedDocument`` from the parser.

        Returns
        -------
        list[DocumentChunk]
            All chunks including the top-level DOCUMENT chunk.
        """
        logger.info(
            "DocumentChunker.chunk() START | document_id=%s | filename=%s | pages=%d",
            doc.document_id,
            doc.filename,
            doc.total_pages,
        )

        # Step 1: assemble full text (prefer page-level, fallback to raw_text)
        full_text = self._assemble_text(doc)

        # Step 2: create top-level DOCUMENT chunk
        doc_chunk_id = _make_chunk_id(doc.document_id, full_text)
        doc_chunk = DocumentChunk(
            chunk_id=doc_chunk_id,
            document_id=doc.document_id,
            parent_id=None,
            hierarchy_level=HierarchyLevel.DOCUMENT,
            page_number=1,
            title=doc.title or doc.filename,
            section=None,
            text=full_text,
            metadata=ChunkMetadata(
                word_count=len(full_text.split()),
                char_count=len(full_text),
            ),
        )

        # Step 3: hierarchical splitting
        all_chunks: list[DocumentChunk] = [doc_chunk]

        chapter_blocks = self._split_by_pattern(full_text, _CHAPTER_RE)

        if chapter_blocks:
            all_chunks.extend(
                await self._process_chapters(
                    chapter_blocks, doc.document_id, doc_chunk_id, doc.pages
                )
            )
        else:
            # No chapters — treat the whole document as one implicit chapter
            # and split directly into sections / paragraphs
            all_chunks.extend(
                await self._process_sections(
                    [full_text],
                    doc.document_id,
                    doc_chunk_id,
                    doc.pages,
                )
            )

        logger.info(
            "DocumentChunker.chunk() SPLIT DONE | document_id=%s | chunks=%d",
            doc.document_id,
            len(all_chunks),
        )

        # Step 4: AI enrichment (optional)
        if self._enrich:
            all_chunks = await self._enrich_all(all_chunks, doc.filename)

        logger.info(
            "DocumentChunker.chunk() DONE | document_id=%s | final_chunks=%d",
            doc.document_id,
            len(all_chunks),
        )
        return all_chunks

    # ------------------------------------------------------------------
    # Text Assembly
    # ------------------------------------------------------------------

    def _assemble_text(self, doc: ParsedDocument) -> str:
        """Concatenate page texts with a page-break marker, or use raw_text."""
        if doc.pages:
            return "\n\n".join(
                f"[PAGE {p.page_number}]\n{p.text}" for p in doc.pages
            )
        return doc.raw_text

    # ------------------------------------------------------------------
    # Structural Splitting
    # ------------------------------------------------------------------

    def _split_by_pattern(self, text: str, pattern: re.Pattern[str]) -> list[str]:
        """
        Split *text* at every match of *pattern* (multiline) and return a list
        of non-empty blocks.  The heading line is kept as the first line of
        each block.
        """
        splits = list(pattern.finditer(text))
        if not splits:
            return []

        blocks: list[str] = []
        for i, match in enumerate(splits):
            start = match.start()
            end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
            block = text[start:end].strip()
            if len(block) >= _MIN_CHUNK_CHARS:
                blocks.append(block)
        return blocks

    async def _process_chapters(
        self,
        chapter_blocks: list[str],
        document_id: str,
        doc_chunk_id: str,
        pages: list[ParsedPage],
    ) -> list[DocumentChunk]:
        """Create CHAPTER chunks and recursively process their sections."""
        result: list[DocumentChunk] = []
        full_text = self._assemble_text_from_pages(pages)

        for block in chapter_blocks:
            chapter_id = _make_chunk_id(document_id, block)
            offset = full_text.find(block[:100])
            page_num = _page_for_offset(pages, max(0, offset)) if pages else 1
            title = _extract_title_from_block(block)
            section = _extract_section_number(block)

            chapter_chunk = DocumentChunk(
                chunk_id=chapter_id,
                document_id=document_id,
                parent_id=doc_chunk_id,
                hierarchy_level=HierarchyLevel.CHAPTER,
                page_number=page_num,
                title=title,
                section=section,
                text=block,
                metadata=ChunkMetadata(
                    word_count=len(block.split()),
                    char_count=len(block),
                ),
            )
            result.append(chapter_chunk)

            # Recurse into sections
            result.extend(
                await self._process_sections([block], document_id, chapter_id, pages)
            )

        return result

    async def _process_sections(
        self,
        parent_blocks: list[str],
        document_id: str,
        parent_chunk_id: str,
        pages: list[ParsedPage],
    ) -> list[DocumentChunk]:
        """Create SECTION chunks and recursively process their clauses."""
        result: list[DocumentChunk] = []
        full_text = self._assemble_text_from_pages(pages)

        for parent_block in parent_blocks:
            section_blocks = self._split_by_pattern(parent_block, _SECTION_RE)
            if not section_blocks:
                # No sections — go straight to clauses / paragraphs
                result.extend(
                    self._process_clauses(
                        [parent_block], document_id, parent_chunk_id, pages
                    )
                )
                continue

            for block in section_blocks:
                section_id = _make_chunk_id(document_id, block)
                offset = full_text.find(block[:100])
                page_num = _page_for_offset(pages, max(0, offset)) if pages else 1
                title = _extract_title_from_block(block)
                section = _extract_section_number(block)

                section_chunk = DocumentChunk(
                    chunk_id=section_id,
                    document_id=document_id,
                    parent_id=parent_chunk_id,
                    hierarchy_level=HierarchyLevel.SECTION,
                    page_number=page_num,
                    title=title,
                    section=section,
                    text=block,
                    metadata=ChunkMetadata(
                        word_count=len(block.split()),
                        char_count=len(block),
                    ),
                )
                result.append(section_chunk)

                # Recurse into clauses
                result.extend(
                    self._process_clauses([block], document_id, section_id, pages)
                )

        return result

    def _process_clauses(
        self,
        parent_blocks: list[str],
        document_id: str,
        parent_chunk_id: str,
        pages: list[ParsedPage],
    ) -> list[DocumentChunk]:
        """Create CLAUSE chunks and their PARAGRAPH children."""
        result: list[DocumentChunk] = []
        full_text = self._assemble_text_from_pages(pages)

        for parent_block in parent_blocks:
            clause_blocks = self._split_by_pattern(parent_block, _CLAUSE_RE)
            if not clause_blocks:
                # No clause markers — split into paragraphs directly
                result.extend(
                    self._process_paragraphs(
                        [parent_block], document_id, parent_chunk_id, pages
                    )
                )
                continue

            for block in clause_blocks:
                clause_id = _make_chunk_id(document_id, block)
                offset = full_text.find(block[:100])
                page_num = _page_for_offset(pages, max(0, offset)) if pages else 1
                title = _extract_title_from_block(block)
                section = _extract_section_number(block)

                clause_chunk = DocumentChunk(
                    chunk_id=clause_id,
                    document_id=document_id,
                    parent_id=parent_chunk_id,
                    hierarchy_level=HierarchyLevel.CLAUSE,
                    page_number=page_num,
                    title=title,
                    section=section,
                    text=block,
                    metadata=ChunkMetadata(
                        word_count=len(block.split()),
                        char_count=len(block),
                    ),
                )
                result.append(clause_chunk)

                # Recurse into paragraphs
                result.extend(
                    self._process_paragraphs([block], document_id, clause_id, pages)
                )

        return result

    def _process_paragraphs(
        self,
        parent_blocks: list[str],
        document_id: str,
        parent_chunk_id: str,
        pages: list[ParsedPage],
    ) -> list[DocumentChunk]:
        """Split parent blocks into PARAGRAPH chunks by blank-line separation."""
        result: list[DocumentChunk] = []
        full_text = self._assemble_text_from_pages(pages)

        for parent_block in parent_blocks:
            raw_paras = _PARA_SPLIT_RE.split(parent_block)
            paras = [p.strip() for p in raw_paras if len(p.strip()) >= _MIN_CHUNK_CHARS]

            # If the entire block is one paragraph, skip (avoid duplicating parent)
            if len(paras) <= 1:
                continue

            for para in paras:
                para_id = _make_chunk_id(document_id, para)
                offset = full_text.find(para[:80])
                page_num = _page_for_offset(pages, max(0, offset)) if pages else 1

                para_chunk = DocumentChunk(
                    chunk_id=para_id,
                    document_id=document_id,
                    parent_id=parent_chunk_id,
                    hierarchy_level=HierarchyLevel.PARAGRAPH,
                    page_number=page_num,
                    title=None,
                    section=None,
                    text=para,
                    metadata=ChunkMetadata(
                        word_count=len(para.split()),
                        char_count=len(para),
                    ),
                )
                result.append(para_chunk)

        return result

    # ------------------------------------------------------------------
    # AI Enrichment
    # ------------------------------------------------------------------

    async def _enrich_all(
        self, chunks: list[DocumentChunk], document_name: str
    ) -> list[DocumentChunk]:
        """
        Concurrently enrich all chunks with AI-generated metadata.

        Uses a semaphore to cap concurrent LLM calls.  ``ChunkEnrichmentError``
        is caught per-chunk and logged — enrichment failure is non-fatal
        (the original chunk is returned unchanged).  LLM errors are already
        handled inside ``_enrich_chunk``.
        """
        logger.info(
            "DocumentChunker._enrich_all() START | chunks=%d | document=%s",
            len(chunks),
            document_name,
        )

        async def _safe_enrich(chunk: DocumentChunk) -> DocumentChunk:
            try:
                return await self._enrich_chunk(chunk, document_name)
            except ChunkEnrichmentError as exc:
                logger.warning(
                    "Chunk enrichment normalisation error — using original chunk | %s",
                    exc,
                )
                return chunk

        enriched = await asyncio.gather(
            *[_safe_enrich(chunk) for chunk in chunks],
            return_exceptions=False,
        )

        logger.info(
            "DocumentChunker._enrich_all() DONE | enriched=%d", len(enriched)
        )
        return list(enriched)

    async def _enrich_chunk(
        self, chunk: DocumentChunk, document_name: str
    ) -> DocumentChunk:
        """
        Enrich a single chunk using ``AIService.generate_chunk_description()``.

        Normalises the AI response before constructing ``ChunkMetadata`` so that
        occasional LLM artefacts (``null`` items, empty strings, duplicates)
        never cause a Pydantic ``ValidationError``.

        Returns the original chunk unchanged on any LLM error (best-effort).

        Raises
        ------
        ChunkEnrichmentError
            If normalised metadata is still invalid after sanitisation.
            This indicates a structural problem with the AI response that
            normalisation cannot recover — callers should log and skip.
        """
        # Skip enrichment for very short chunks or the top-level DOCUMENT chunk
        if (
            len(chunk.text) < _MIN_CHUNK_CHARS * 2
            or chunk.hierarchy_level == HierarchyLevel.DOCUMENT
        ):
            return chunk

        async with self._semaphore:
            try:
                response = await self._ai.generate_chunk_description(
                    chunk_text=chunk.text[:2000],  # stay within token budget
                    document_name=document_name,
                )
                data = response.data

                # ------------------------------------------------------------------
                # Normalise list fields before Pydantic construction.
                # _normalize_str_list() removes None, empty strings, duplicates,
                # and converts numeric primitives to strings.
                # ------------------------------------------------------------------
                try:
                    norm_entities = _normalize_str_list(
                        data.get("key_entities"), field="key_entities"
                    )
                    norm_dates = _normalize_str_list(
                        data.get("key_dates"), field="key_dates"
                    )
                    norm_amounts = _normalize_str_list(
                        data.get("key_amounts"), field="key_amounts"
                    )
                except TypeError as exc:
                    raise ChunkEnrichmentError(
                        str(exc),
                        document_id=chunk.document_id,
                        chunk_id=chunk.chunk_id,
                        page_number=chunk.page_number,
                        field=str(exc).split("Field '")[1].split("'")[0]
                        if "Field '" in str(exc)
                        else "unknown",
                        offending_value=data,
                    ) from exc

                logger.debug(
                    "Chunk enrichment | chunk_id=%s | page=%d | "
                    "raw_response=%r | "
                    "norm_entities=%r | norm_dates=%r | norm_amounts=%r",
                    chunk.chunk_id,
                    chunk.page_number,
                    data,
                    norm_entities,
                    norm_dates,
                    norm_amounts,
                )

                try:
                    enriched_meta = ChunkMetadata(
                        content_type=data.get("content_type") or None,
                        topic=data.get("topic") or None,
                        key_entities=norm_entities,
                        key_dates=norm_dates,
                        key_amounts=norm_amounts,
                        has_eligibility_criteria=bool(
                            data.get("has_eligibility_criteria", False)
                        ),
                        has_procedure_steps=bool(
                            data.get("has_procedure_steps", False)
                        ),
                        summary=data.get("summary") or None,
                        word_count=chunk.metadata.word_count,
                        char_count=chunk.metadata.char_count,
                    )
                except Exception as exc:  # Pydantic ValidationError or similar
                    raise ChunkEnrichmentError(
                        f"ChunkMetadata construction failed after normalisation: {exc}",
                        document_id=chunk.document_id,
                        chunk_id=chunk.chunk_id,
                        page_number=chunk.page_number,
                        field="ChunkMetadata",
                        offending_value={
                            "key_entities": norm_entities,
                            "key_dates": norm_dates,
                            "key_amounts": norm_amounts,
                        },
                    ) from exc

                return DocumentChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    parent_id=chunk.parent_id,
                    hierarchy_level=chunk.hierarchy_level,
                    page_number=chunk.page_number,
                    title=chunk.title,
                    section=chunk.section,
                    text=chunk.text,
                    metadata=enriched_meta,
                )

            except ChunkEnrichmentError:
                # Re-raise — caller (_enrich_all) must decide whether to skip
                raise
            except LLMError as exc:
                logger.warning(
                    "Chunk enrichment failed | chunk_id=%s | error=%s",
                    chunk.chunk_id,
                    exc,
                )
                return chunk

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _assemble_text_from_pages(pages: Sequence[ParsedPage]) -> str:
        """Reconstruct concatenated text from a page list (for offset lookup)."""
        return "\n\n".join(f"[PAGE {p.page_number}]\n{p.text}" for p in pages)

"""
PDF Parser
==========

``PDFParser`` converts raw PDF bytes or a file path into a ``ParsedDocument``.

Strategy
--------
Dual-pass extraction:

1. **PyMuPDF** (``fitz``) — primary.  Fast, accurate text + document metadata.
2. **pdfplumber** — secondary, table pass only.  Runs concurrently to extract
   structured tables from every page that PyMuPDF found text on.

If PyMuPDF is unavailable (import error at container build time), the parser
falls back to pdfplumber for both text and tables.

The results are merged: PyMuPDF text + pdfplumber tables → ``ParsedPage``.

Contract
--------
- Input:  PDF bytes or filesystem path
- Output: ``ParsedDocument`` (defined in ``app.schemas.pipeline``)
- No AI calls.  Pure PDF → structured data transformation.
- Thread-safe: both PyMuPDF and pdfplumber are called in a thread-pool
  executor so they do not block the asyncio event loop.
- ``file_metadata`` on the returned document always contains:
  - ``sha256``       — hex digest of the raw PDF bytes
  - ``size_bytes``   — byte length
  - ``mime_type``    — always "application/pdf"
  - ``author``       — from PDF metadata (may be None)
  - ``creator``      — from PDF metadata (may be None)
  - ``subject``      — from PDF metadata (may be None)

Usage
-----
::

    from app.pipeline.parser import PDFParser

    parser = PDFParser()

    # From file path
    doc = await parser.parse("path/to/policy.pdf")

    # From uploaded bytes
    doc = await parser.parse_bytes(pdf_bytes, filename="policy.pdf")
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from app.schemas.pipeline import ParsedDocument, ParsedPage

logger = logging.getLogger(__name__)

# Try importing optional PDF libraries — fail gracefully so unit tests
# that mock these can still import the module.
try:
    import fitz  # PyMuPDF

    _FITZ_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FITZ_AVAILABLE = False
    logger.warning("PyMuPDF (fitz) not available — falling back to pdfplumber")

try:
    import pdfplumber

    _PDFPLUMBER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not available — table extraction disabled")

# Shared thread pool for blocking PDF I/O (max 4 workers — PDF parsing is CPU-bound)
_THREAD_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="pdf_parser")


# ---------------------------------------------------------------------------
# PDFParser
# ---------------------------------------------------------------------------


class PDFParser:
    """
    Async PDF parser producing ``ParsedDocument`` objects.

    Parameters
    ----------
    prefer_pymupdf :
        Use PyMuPDF as the primary text extractor (recommended).
        Falls back automatically if PyMuPDF is not installed.
    max_text_per_page :
        Character cap per page for ``raw_text`` concatenation.
        Full text is always available in ``ParsedPage.text``.
    """

    def __init__(
        self,
        *,
        prefer_pymupdf: bool = True,
        max_text_per_page: int = 50_000,
    ) -> None:
        self._use_fitz = prefer_pymupdf and _FITZ_AVAILABLE
        self._use_pdfplumber = _PDFPLUMBER_AVAILABLE
        self._max_text_per_page = max_text_per_page
        logger.info(
            "PDFParser initialised | pymupdf=%s | pdfplumber=%s",
            self._use_fitz,
            self._use_pdfplumber,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def parse(self, file_path: Path | str) -> ParsedDocument:
        """
        Parse a PDF from a filesystem path.

        Parameters
        ----------
        file_path :
            Absolute or relative path to a PDF file.

        Returns
        -------
        ParsedDocument
        """
        path = Path(file_path)
        logger.info("PDFParser.parse() | file=%s", path.name)
        pdf_bytes = await asyncio.get_running_loop().run_in_executor(
            _THREAD_POOL, path.read_bytes
        )
        return await self.parse_bytes(pdf_bytes, filename=path.name)

    async def parse_bytes(self, content: bytes, filename: str) -> ParsedDocument:
        """
        Parse a PDF from raw bytes (e.g., from a FastAPI ``UploadFile``).

        Parameters
        ----------
        content :
            Raw PDF bytes.
        filename :
            Original filename (used for logging and stored in ``file_metadata``).

        Returns
        -------
        ParsedDocument
        """
        logger.info(
            "PDFParser.parse_bytes() | filename=%s | size_bytes=%d",
            filename,
            len(content),
        )

        file_hash = _sha256(content)
        document_id = str(uuid.uuid5(uuid.NAMESPACE_URL, file_hash))

        # Run both extraction passes concurrently in thread pool
        loop = asyncio.get_running_loop()

        async def _noop_pages():
            return ([], {})

        async def _noop_tables():
            return {}

        fitz_future = loop.run_in_executor(
            _THREAD_POOL, self._extract_fitz, content
        ) if self._use_fitz else _noop_pages()

        plumber_future = loop.run_in_executor(
            _THREAD_POOL, self._extract_pdfplumber, content
        ) if self._use_pdfplumber else _noop_tables()

        fitz_result, plumber_result = await asyncio.gather(
            fitz_future, plumber_future, return_exceptions=True
        )

        # Unpack fitz result
        if isinstance(fitz_result, Exception):
            logger.warning("PyMuPDF extraction failed: %s — using pdfplumber", fitz_result)
            fitz_pages: list[dict[str, Any]] = []
            fitz_meta: dict[str, Any] = {}
        else:
            fitz_pages, fitz_meta = fitz_result  # type: ignore[misc]

        # Unpack pdfplumber result
        if isinstance(plumber_result, Exception):
            logger.warning("pdfplumber extraction failed: %s", plumber_result)
            plumber_page_tables: dict[int, list[dict]] = {}
        else:
            plumber_page_tables = plumber_result  # type: ignore[assignment]

        # If fitz produced no pages, fall back to pdfplumber for text too
        if not fitz_pages and self._use_pdfplumber:
            fitz_pages, fitz_meta = await loop.run_in_executor(
                _THREAD_POOL, self._extract_pdfplumber_text, content
            )

        pages = self._merge_pages(fitz_pages, plumber_page_tables)

        if not pages:
            raise ValueError(f"No text could be extracted from '{filename}'")

        raw_text = "\n\n".join(p.text[: self._max_text_per_page] for p in pages)
        title = fitz_meta.get("title") or _infer_title(pages)

        file_metadata: dict[str, Any] = {
            "sha256": file_hash,
            "size_bytes": len(content),
            "mime_type": "application/pdf",
            "author": fitz_meta.get("author"),
            "creator": fitz_meta.get("creator"),
            "subject": fitz_meta.get("subject"),
            "pdf_version": fitz_meta.get("format"),
            "page_count": len(pages),
        }
        # Remove None values
        file_metadata = {k: v for k, v in file_metadata.items() if v is not None}

        doc = ParsedDocument(
            document_id=document_id,
            filename=filename,
            title=title,
            total_pages=len(pages),
            pages=pages,
            raw_text=raw_text,
            file_metadata=file_metadata,
        )

        logger.info(
            "PDFParser.parse_bytes() DONE | document_id=%s | pages=%d | chars=%d",
            document_id,
            len(pages),
            len(raw_text),
        )
        return doc

    # ------------------------------------------------------------------
    # PyMuPDF extraction (blocking — run in thread pool)
    # ------------------------------------------------------------------

    def _extract_fitz(
        self, content: bytes
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Extract text and metadata from PDF bytes using PyMuPDF.

        Returns
        -------
        (page_data_list, document_metadata)
        """
        import fitz  # noqa: PLC0415  (local import — may not be available)

        pages: list[dict[str, Any]] = []
        doc = fitz.open(stream=content, filetype="pdf")

        try:
            meta: dict[str, Any] = dict(doc.metadata or {})
            meta["format"] = doc.pdf_catalog().get("Version", None) if hasattr(doc, "pdf_catalog") else None

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text") or ""
                # Also try extracting text blocks for better structure
                blocks = page.get_text("blocks") or []
                structured_blocks = [
                    {
                        "bbox": list(b[:4]),
                        "text": b[4].strip(),
                        "block_type": b[6],
                    }
                    for b in blocks
                    if len(b) > 4 and b[4].strip()
                ]
                pages.append(
                    {
                        "page_number": page_num + 1,
                        "text": text,
                        "blocks": structured_blocks,
                        "width": page.rect.width,
                        "height": page.rect.height,
                        "rotation": page.rotation,
                    }
                )
        finally:
            doc.close()

        return pages, meta

    # ------------------------------------------------------------------
    # pdfplumber table extraction (blocking — run in thread pool)
    # ------------------------------------------------------------------

    def _extract_pdfplumber(self, content: bytes) -> dict[int, list[dict]]:
        """
        Extract tables from each page using pdfplumber.

        Returns a mapping of {page_number (1-based): [table_dict, ...]}.
        """
        import pdfplumber  # noqa: PLC0415

        page_tables: dict[int, list[dict]] = {}
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                pnum = page.page_number  # 1-based
                tables = page.extract_tables() or []
                parsed = []
                for table in tables:
                    if not table:
                        continue
                    # Use first row as header if it looks like one
                    if table and all(isinstance(c, str) for c in (table[0] or [])):
                        headers = [str(h or "").strip() for h in table[0]]
                        rows = []
                        for row in table[1:]:
                            if row:
                                rows.append(
                                    {
                                        headers[i] if i < len(headers) else str(i): str(cell or "").strip()
                                        for i, cell in enumerate(row)
                                    }
                                )
                        parsed.append({"headers": headers, "rows": rows})
                    else:
                        # No clear headers — store raw
                        parsed.append({"raw": [[str(c or "") for c in r] for r in table if r]})
                if parsed:
                    page_tables[pnum] = parsed
        return page_tables

    def _extract_pdfplumber_text(
        self, content: bytes
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Full text + table extraction using pdfplumber (used when fitz unavailable).
        """
        import pdfplumber  # noqa: PLC0415

        pages: list[dict[str, Any]] = []
        meta: dict[str, Any] = {}

        with pdfplumber.open(io.BytesIO(content)) as pdf:
            if pdf.metadata:
                meta = {k: v for k, v in pdf.metadata.items() if v}
            for page in pdf.pages:
                text = page.extract_text() or ""
                tables = page.extract_tables() or []
                pages.append(
                    {
                        "page_number": page.page_number,
                        "text": text,
                        "blocks": [],
                        "width": float(page.width or 0),
                        "height": float(page.height or 0),
                        "rotation": 0,
                        "_plumber_tables": tables,
                    }
                )
        return pages, meta

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def _merge_pages(
        self,
        fitz_pages: list[dict[str, Any]],
        plumber_tables: dict[int, list[dict]],
    ) -> list[ParsedPage]:
        """
        Merge fitz page data and pdfplumber table data into ``ParsedPage`` objects.
        """
        result: list[ParsedPage] = []
        for page_data in fitz_pages:
            pnum = page_data["page_number"]
            text = page_data.get("text", "").strip()
            tables = plumber_tables.get(pnum, [])

            # If pdfplumber found tables but no fitz text, use plumber tables as text too
            if not text and page_data.get("_plumber_tables"):
                raw_tables = page_data["_plumber_tables"]
                text_parts = []
                for table in raw_tables:
                    for row in table:
                        if row:
                            text_parts.append(" | ".join(str(c or "") for c in row))
                text = "\n".join(text_parts)

            page_meta: dict[str, Any] = {}
            if page_data.get("width"):
                page_meta["width"] = page_data["width"]
            if page_data.get("height"):
                page_meta["height"] = page_data["height"]
            if page_data.get("rotation"):
                page_meta["rotation"] = page_data["rotation"]
            if page_data.get("blocks"):
                page_meta["block_count"] = len(page_data["blocks"])

            result.append(
                ParsedPage(
                    page_number=pnum,
                    text=text,
                    tables=tables,
                    metadata=page_meta,
                )
            )
        return result


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _sha256(data: bytes) -> str:
    """Return the hex SHA-256 digest of *data*."""
    return hashlib.sha256(data).hexdigest()


def _infer_title(pages: list[ParsedPage]) -> str | None:
    """
    Infer a document title from the first non-empty line of page 1.

    Returns None if the first page text is empty.
    """
    if not pages:
        return None
    first_text = pages[0].text.strip()
    if not first_text:
        return None
    first_line = first_text.splitlines()[0].strip()
    # Only use it as a title if it's reasonably short
    return first_line[:200] if len(first_line) < 200 else None

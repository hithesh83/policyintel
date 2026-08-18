"""
Unit tests for PDFParser.

Strategy: We patch the instance's _extract_* methods directly AND force
_use_fitz/_use_pdfplumber=True so parse_bytes runs the mocked methods
via the thread pool (MagicMock is callable and works in executors).

This approach avoids needing fitz/pdfplumber to actually be installed
in the test environment, while still fully exercising parse_bytes logic.

Covers
------
- parse_bytes() returns a valid ParsedDocument
- SHA-256 hash stored in file_metadata
- document_id is a stable UUID5
- Pages have correct structure and page numbers
- Title extracted from PDF metadata
- pdfplumber table extraction merged into pages
- Fallback to pdfplumber when fitz fails
- Different bytes → different document_ids
- _sha256 helper
- _infer_title helper
"""

from __future__ import annotations

import hashlib
import uuid
from unittest.mock import MagicMock

import pytest

from app.schemas.pipeline import ParsedDocument, ParsedPage


# ---------------------------------------------------------------------------
# Controlled extraction return values
# ---------------------------------------------------------------------------

_FAKE_PDF_BYTES = b"%PDF-1.4 fake pdf content for testing purposes only"

_PAGE1_TEXT = """\
PRADHAN MANTRI KISAN SAMMAN NIDHI SCHEME
Ministry of Agriculture and Farmers Welfare

The PM Kisan scheme provides Rs. 6,000 per year to eligible farmers.
"""

_PAGE2_TEXT = """\
Eligibility Criteria

Farmers must have cultivable land and annual income below Rs. 2 lakh.
"""

_FITZ_PAGES = [
    {
        "page_number": 1,
        "text": _PAGE1_TEXT,
        "blocks": [{"bbox": [0, 0, 100, 20], "text": "PRADHAN MANTRI", "block_type": 0}],
        "width": 595.0,
        "height": 842.0,
        "rotation": 0,
    },
    {
        "page_number": 2,
        "text": _PAGE2_TEXT,
        "blocks": [],
        "width": 595.0,
        "height": 842.0,
        "rotation": 0,
    },
]

_FITZ_META = {
    "title": "PM Kisan SOP",
    "author": "Ministry of Agriculture",
    "creator": "Adobe PDF",
    "subject": "Agricultural Scheme",
    "format": "1.4",
}

_PLUMBER_TABLES: dict[int, list[dict]] = {
    1: [
        {
            "headers": ["Scheme", "Amount", "Frequency"],
            "rows": [{"Scheme": "PM Kisan", "Amount": "6000", "Frequency": "Annual"}],
        }
    ]
}


# ---------------------------------------------------------------------------
# Helper: build a PDFParser with mocked extraction methods
# ---------------------------------------------------------------------------


def _make_parser(
    fitz_pages=None,
    fitz_meta=None,
    plumber_tables=None,
    fitz_raises=None,
):
    """
    Create a PDFParser whose blocking extraction methods are replaced with
    synchronous MagicMocks.  Also forces _use_fitz=True/_use_pdfplumber=True
    so parse_bytes actually calls these methods via the thread pool.

    run_in_executor with a MagicMock works correctly — the mock is callable
    and returns synchronously from the thread pool worker.
    """
    from app.pipeline.parser import PDFParser

    parser = PDFParser.__new__(PDFParser)
    # Force availability flags — don't rely on actual library installation
    parser._use_fitz = True
    parser._use_pdfplumber = True
    parser._max_text_per_page = 50_000

    if fitz_raises:
        parser._extract_fitz = MagicMock(side_effect=fitz_raises)
    else:
        pages = fitz_pages if fitz_pages is not None else _FITZ_PAGES
        meta = fitz_meta if fitz_meta is not None else _FITZ_META
        parser._extract_fitz = MagicMock(return_value=(pages, meta))

    tables = plumber_tables if plumber_tables is not None else {}
    parser._extract_pdfplumber = MagicMock(return_value=tables)

    # pdfplumber text fallback (used when fitz returns empty pages)
    parser._extract_pdfplumber_text = MagicMock(return_value=(
        [
            {
                "page_number": 1,
                "text": _PAGE1_TEXT,
                "blocks": [],
                "width": 595.0,
                "height": 842.0,
                "rotation": 0,
            }
        ],
        {"title": "Fallback Title"},
    ))

    return parser


# ---------------------------------------------------------------------------
# Tests: basic parse_bytes
# ---------------------------------------------------------------------------


class TestPDFParserBasic:
    @pytest.mark.asyncio
    async def test_parse_bytes_returns_parsed_document(self):
        parser = _make_parser()
        doc = await parser.parse_bytes(_FAKE_PDF_BYTES, filename="test.pdf")
        assert isinstance(doc, ParsedDocument)

    @pytest.mark.asyncio
    async def test_document_id_is_stable_uuid5(self):
        """Same bytes → same document_id regardless of filename."""
        parser1 = _make_parser()
        parser2 = _make_parser()

        doc1 = await parser1.parse_bytes(_FAKE_PDF_BYTES, filename="a.pdf")
        doc2 = await parser2.parse_bytes(_FAKE_PDF_BYTES, filename="b.pdf")

        assert doc1.document_id == doc2.document_id
        # Valid UUID format
        uuid.UUID(doc1.document_id)

    @pytest.mark.asyncio
    async def test_document_id_is_valid_uuid(self):
        parser = _make_parser()
        doc = await parser.parse_bytes(_FAKE_PDF_BYTES, filename="test.pdf")
        parsed_uuid = uuid.UUID(doc.document_id)
        assert str(parsed_uuid) == doc.document_id

    @pytest.mark.asyncio
    async def test_file_hash_in_metadata(self):
        parser = _make_parser()
        doc = await parser.parse_bytes(_FAKE_PDF_BYTES, filename="test.pdf")

        expected_hash = hashlib.sha256(_FAKE_PDF_BYTES).hexdigest()
        assert doc.file_metadata["sha256"] == expected_hash

    @pytest.mark.asyncio
    async def test_file_size_in_metadata(self):
        parser = _make_parser()
        doc = await parser.parse_bytes(_FAKE_PDF_BYTES, filename="test.pdf")

        assert doc.file_metadata["size_bytes"] == len(_FAKE_PDF_BYTES)

    @pytest.mark.asyncio
    async def test_mime_type_in_metadata(self):
        parser = _make_parser()
        doc = await parser.parse_bytes(_FAKE_PDF_BYTES, filename="test.pdf")
        assert doc.file_metadata["mime_type"] == "application/pdf"

    @pytest.mark.asyncio
    async def test_pages_have_correct_count(self):
        parser = _make_parser(fitz_pages=_FITZ_PAGES)
        doc = await parser.parse_bytes(_FAKE_PDF_BYTES, filename="test.pdf")

        assert doc.total_pages == 2
        assert len(doc.pages) == 2

    @pytest.mark.asyncio
    async def test_pages_are_one_indexed(self):
        parser = _make_parser()
        doc = await parser.parse_bytes(_FAKE_PDF_BYTES, filename="test.pdf")

        page_numbers = sorted(p.page_number for p in doc.pages)
        assert page_numbers[0] == 1

    @pytest.mark.asyncio
    async def test_raw_text_is_non_empty(self):
        parser = _make_parser()
        doc = await parser.parse_bytes(_FAKE_PDF_BYTES, filename="test.pdf")
        assert len(doc.raw_text) > 0

    @pytest.mark.asyncio
    async def test_raw_text_contains_page_content(self):
        parser = _make_parser()
        doc = await parser.parse_bytes(_FAKE_PDF_BYTES, filename="test.pdf")
        # PRADHAN is from page 1 text
        assert "PRADHAN" in doc.raw_text or "Eligibility" in doc.raw_text

    @pytest.mark.asyncio
    async def test_title_extracted_from_pdf_metadata(self):
        parser = _make_parser(fitz_meta={"title": "PM Kisan SOP 2024"})
        doc = await parser.parse_bytes(_FAKE_PDF_BYTES, filename="test.pdf")
        assert doc.title == "PM Kisan SOP 2024"

    @pytest.mark.asyncio
    async def test_filename_stored_on_document(self):
        parser = _make_parser()
        doc = await parser.parse_bytes(_FAKE_PDF_BYTES, filename="my_policy_2024.pdf")
        assert doc.filename == "my_policy_2024.pdf"

    @pytest.mark.asyncio
    async def test_parsed_pages_are_parsedpage_instances(self):
        parser = _make_parser()
        doc = await parser.parse_bytes(_FAKE_PDF_BYTES, filename="test.pdf")
        for page in doc.pages:
            assert isinstance(page, ParsedPage)
            assert page.page_number >= 1


# ---------------------------------------------------------------------------
# Tests: table merging
# ---------------------------------------------------------------------------


class TestPDFParserTables:
    @pytest.mark.asyncio
    async def test_tables_merged_from_pdfplumber(self):
        parser = _make_parser(plumber_tables=_PLUMBER_TABLES)
        doc = await parser.parse_bytes(_FAKE_PDF_BYTES, filename="test.pdf")

        page1 = next(p for p in doc.pages if p.page_number == 1)
        assert len(page1.tables) >= 1
        assert page1.tables[0]["headers"] == ["Scheme", "Amount", "Frequency"]

    @pytest.mark.asyncio
    async def test_no_tables_gives_empty_list(self):
        parser = _make_parser(plumber_tables={})
        doc = await parser.parse_bytes(_FAKE_PDF_BYTES, filename="test.pdf")
        for page in doc.pages:
            assert isinstance(page.tables, list)

    @pytest.mark.asyncio
    async def test_page_2_without_tables_is_empty(self):
        parser = _make_parser(plumber_tables={1: [{"headers": ["H"], "rows": []}]})
        doc = await parser.parse_bytes(_FAKE_PDF_BYTES, filename="test.pdf")
        page2 = next((p for p in doc.pages if p.page_number == 2), None)
        if page2:
            assert page2.tables == []


# ---------------------------------------------------------------------------
# Tests: fallback behaviour
# ---------------------------------------------------------------------------


class TestPDFParserFallback:
    @pytest.mark.asyncio
    async def test_fitz_exception_is_caught(self):
        """fitz extraction raises → fitz_result is an exception, falls back."""
        parser = _make_parser(fitz_raises=RuntimeError("GPU memory error"))
        # Fallback to pdfplumber text — should NOT raise
        doc = await parser.parse_bytes(_FAKE_PDF_BYTES, filename="test.pdf")
        assert isinstance(doc, ParsedDocument)
        assert len(doc.pages) >= 1

    @pytest.mark.asyncio
    async def test_different_bytes_produce_different_ids(self):
        p1 = _make_parser()
        p2 = _make_parser()
        doc1 = await p1.parse_bytes(b"pdf bytes version one", filename="a.pdf")
        doc2 = await p2.parse_bytes(b"pdf bytes version two", filename="b.pdf")
        assert doc1.document_id != doc2.document_id

    @pytest.mark.asyncio
    async def test_same_bytes_always_same_id(self):
        content = b"stable policy pdf content with fixed hash"
        p1 = _make_parser()
        p2 = _make_parser()
        doc1 = await p1.parse_bytes(content, filename="x.pdf")
        doc2 = await p2.parse_bytes(content, filename="y.pdf")
        assert doc1.document_id == doc2.document_id


# ---------------------------------------------------------------------------
# Tests: _sha256 helper
# ---------------------------------------------------------------------------


class TestSHA256Helper:
    def test_matches_hashlib(self):
        from app.pipeline.parser import _sha256
        data = b"test data for hashing"
        assert _sha256(data) == hashlib.sha256(data).hexdigest()

    def test_is_64_hex_chars(self):
        from app.pipeline.parser import _sha256
        result = _sha256(b"any bytes")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_empty_bytes(self):
        from app.pipeline.parser import _sha256
        assert _sha256(b"") == hashlib.sha256(b"").hexdigest()

    def test_different_inputs_different_outputs(self):
        from app.pipeline.parser import _sha256
        assert _sha256(b"abc") != _sha256(b"xyz")


# ---------------------------------------------------------------------------
# Tests: _infer_title helper
# ---------------------------------------------------------------------------


class TestTitleInference:
    def test_title_from_first_line(self):
        from app.pipeline.parser import _infer_title
        page = ParsedPage(page_number=1, text="PRADHAN MANTRI SCHEME\n\nBody text.")
        assert _infer_title([page]) == "PRADHAN MANTRI SCHEME"

    def test_none_for_empty_pages(self):
        from app.pipeline.parser import _infer_title
        assert _infer_title([]) is None

    def test_none_for_whitespace_text(self):
        from app.pipeline.parser import _infer_title
        page = ParsedPage(page_number=1, text="   \n  ")
        assert _infer_title([page]) is None

    def test_long_line_returns_none(self):
        """Lines of 200+ characters are not used as title."""
        from app.pipeline.parser import _infer_title
        page = ParsedPage(page_number=1, text="A" * 200)
        assert _infer_title([page]) is None

    def test_short_line_used_as_title(self):
        from app.pipeline.parser import _infer_title
        page = ParsedPage(page_number=1, text="PM Kisan\nBody text.")
        assert _infer_title([page]) == "PM Kisan"

    def test_uses_only_first_page(self):
        """Title is inferred from first page only."""
        from app.pipeline.parser import _infer_title
        pages = [
            ParsedPage(page_number=1, text="FIRST PAGE TITLE\nContent."),
            ParsedPage(page_number=2, text="SECOND PAGE\nMore content."),
        ]
        assert _infer_title(pages) == "FIRST PAGE TITLE"

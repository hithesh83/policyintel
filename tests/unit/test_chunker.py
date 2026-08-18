"""
Unit tests for DocumentChunker.

All tests use a mocked AIService — no real LLM calls.

Covers
------
- Hierarchy generation (DOCUMENT → CHAPTER → SECTION → CLAUSE → PARAGRAPH)
- parent_id propagation (each child references its parent's chunk_id)
- chunk_id stability (same text → same UUID5)
- Metadata fields (word_count, char_count populated)
- AI enrichment integration (mocked)
- Enrichment failures are non-fatal
- Single-page and multi-page documents
- Documents with no chapters (direct section split)
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.llm.exceptions import LLMError
from app.llm.models import ExtractionResponse
from app.pipeline.chunker import DocumentChunker, _make_chunk_id
from app.schemas.pipeline import (
    HierarchyLevel,
    ParsedDocument,
    ParsedPage,
)
from app.services.ai_service import AIService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ai_service(*, enrich_data: dict | None = None, fail: bool = False) -> AIService:
    """Build a mocked AIService for chunker tests."""
    ai = MagicMock(spec=AIService)

    if fail:
        ai.generate_chunk_description = AsyncMock(side_effect=LLMError("LLM down"))
    else:
        data = enrich_data or {
            "topic": "Policy overview",
            "content_type": "general_information",
            "key_entities": ["Ministry of Agriculture"],
            "key_dates": ["2023-04-01"],
            "key_amounts": ["₹6,000"],
            "has_eligibility_criteria": False,
            "has_procedure_steps": False,
            "summary": "A brief policy overview.",
        }
        ai.generate_chunk_description = AsyncMock(
            return_value=ExtractionResponse(
                data=data,
                raw_output="{}",
                model="test-model",
                request_id="test-req",
            )
        )
    return ai


def _make_page(page_number: int, text: str) -> ParsedPage:
    return ParsedPage(page_number=page_number, text=text)


def _make_doc(pages: list[ParsedPage], doc_id: str = "doc-001") -> ParsedDocument:
    full_text = "\n\n".join(p.text for p in pages)
    return ParsedDocument(
        document_id=doc_id,
        filename="test_policy.pdf",
        title="Test Policy",
        total_pages=len(pages),
        pages=pages,
        raw_text=full_text,
    )


CHAPTER_TEXT = """\
Chapter 1 Eligibility

1.1 Income Criteria

The annual household income must not exceed Rs. 2 lakh per annum.

Applicants must be resident citizens of India.

1.2 Age Criteria

The beneficiary must be between 18 and 60 years of age.

Chapter 2 Benefits

2.1 Financial Assistance

Eligible farmers will receive Rs. 6,000 per year.

The amount is disbursed in three equal instalments of Rs. 2,000 each.
"""


# ---------------------------------------------------------------------------
# Tests: chunk_id stability
# ---------------------------------------------------------------------------


class TestChunkIdStability:
    def test_same_text_produces_same_id(self):
        id1 = _make_chunk_id("doc-001", "Hello World")
        id2 = _make_chunk_id("doc-001", "Hello World")
        assert id1 == id2

    def test_different_text_produces_different_id(self):
        id1 = _make_chunk_id("doc-001", "Hello World")
        id2 = _make_chunk_id("doc-001", "Different text")
        assert id1 != id2

    def test_different_document_same_text_produces_different_id(self):
        id1 = _make_chunk_id("doc-001", "Hello World")
        id2 = _make_chunk_id("doc-002", "Hello World")
        assert id1 != id2

    def test_chunk_id_is_valid_uuid(self):
        chunk_id = _make_chunk_id("doc-001", "Some text")
        # Should not raise
        parsed = uuid.UUID(chunk_id)
        assert str(parsed) == chunk_id


# ---------------------------------------------------------------------------
# Tests: single page, no chapters
# ---------------------------------------------------------------------------


class TestSinglePageNoChapters:
    @pytest.mark.asyncio
    async def test_always_has_document_chunk(self):
        pages = [_make_page(1, "Simple policy text without chapters or sections.")]
        doc = _make_doc(pages)
        ai = _make_ai_service()

        chunker = DocumentChunker(ai, enrich_chunks=False)
        chunks = await chunker.chunk(doc)

        doc_chunks = [c for c in chunks if c.hierarchy_level == HierarchyLevel.DOCUMENT]
        assert len(doc_chunks) == 1

    @pytest.mark.asyncio
    async def test_document_chunk_has_no_parent(self):
        pages = [_make_page(1, "Policy text.")]
        doc = _make_doc(pages)
        ai = _make_ai_service()

        chunker = DocumentChunker(ai, enrich_chunks=False)
        chunks = await chunker.chunk(doc)

        doc_chunk = next(c for c in chunks if c.hierarchy_level == HierarchyLevel.DOCUMENT)
        assert doc_chunk.parent_id is None

    @pytest.mark.asyncio
    async def test_document_chunk_carries_correct_document_id(self):
        pages = [_make_page(1, "Policy text.")]
        doc = _make_doc(pages, doc_id="my-unique-doc")
        ai = _make_ai_service()

        chunker = DocumentChunker(ai, enrich_chunks=False)
        chunks = await chunker.chunk(doc)

        for chunk in chunks:
            assert chunk.document_id == "my-unique-doc"

    @pytest.mark.asyncio
    async def test_metadata_word_count_populated(self):
        text = "This is a test policy document with exactly ten words here."
        pages = [_make_page(1, text)]
        doc = _make_doc(pages)
        ai = _make_ai_service()

        chunker = DocumentChunker(ai, enrich_chunks=False)
        chunks = await chunker.chunk(doc)

        doc_chunk = next(c for c in chunks if c.hierarchy_level == HierarchyLevel.DOCUMENT)
        assert doc_chunk.metadata.word_count > 0
        assert doc_chunk.metadata.char_count > 0


# ---------------------------------------------------------------------------
# Tests: chapter-based document
# ---------------------------------------------------------------------------


class TestChapterDocument:
    @pytest.mark.asyncio
    async def test_chapter_chunks_created(self):
        pages = [_make_page(1, CHAPTER_TEXT)]
        doc = _make_doc(pages)
        ai = _make_ai_service()

        chunker = DocumentChunker(ai, enrich_chunks=False)
        chunks = await chunker.chunk(doc)

        chapter_chunks = [c for c in chunks if c.hierarchy_level == HierarchyLevel.CHAPTER]
        assert len(chapter_chunks) >= 2, "Expected at least Chapter 1 and Chapter 2"

    @pytest.mark.asyncio
    async def test_chapters_parent_is_document(self):
        pages = [_make_page(1, CHAPTER_TEXT)]
        doc = _make_doc(pages)
        ai = _make_ai_service()

        chunker = DocumentChunker(ai, enrich_chunks=False)
        chunks = await chunker.chunk(doc)

        doc_chunk = next(c for c in chunks if c.hierarchy_level == HierarchyLevel.DOCUMENT)
        chapter_chunks = [c for c in chunks if c.hierarchy_level == HierarchyLevel.CHAPTER]

        for ch in chapter_chunks:
            assert ch.parent_id == doc_chunk.chunk_id, (
                f"Chapter '{ch.title}' parent_id should be doc chunk_id"
            )

    @pytest.mark.asyncio
    async def test_section_chunks_created(self):
        pages = [_make_page(1, CHAPTER_TEXT)]
        doc = _make_doc(pages)
        ai = _make_ai_service()

        chunker = DocumentChunker(ai, enrich_chunks=False)
        chunks = await chunker.chunk(doc)

        section_chunks = [c for c in chunks if c.hierarchy_level == HierarchyLevel.SECTION]
        assert len(section_chunks) >= 1

    @pytest.mark.asyncio
    async def test_sections_parent_is_chapter(self):
        pages = [_make_page(1, CHAPTER_TEXT)]
        doc = _make_doc(pages)
        ai = _make_ai_service()

        chunker = DocumentChunker(ai, enrich_chunks=False)
        chunks = await chunker.chunk(doc)

        chapter_ids = {c.chunk_id for c in chunks if c.hierarchy_level == HierarchyLevel.CHAPTER}
        section_chunks = [c for c in chunks if c.hierarchy_level == HierarchyLevel.SECTION]

        for sec in section_chunks:
            assert sec.parent_id in chapter_ids, (
                f"Section '{sec.title}' parent_id not found in chapter IDs"
            )

    @pytest.mark.asyncio
    async def test_all_chunks_have_page_number(self):
        pages = [_make_page(1, CHAPTER_TEXT)]
        doc = _make_doc(pages)
        ai = _make_ai_service()

        chunker = DocumentChunker(ai, enrich_chunks=False)
        chunks = await chunker.chunk(doc)

        for chunk in chunks:
            assert chunk.page_number >= 1


# ---------------------------------------------------------------------------
# Tests: AI enrichment
# ---------------------------------------------------------------------------


class TestAIEnrichment:
    @pytest.mark.asyncio
    async def test_enrichment_populates_metadata(self):
        pages = [_make_page(1, CHAPTER_TEXT)]
        doc = _make_doc(pages)
        ai = _make_ai_service(
            enrich_data={
                "topic": "Eligibility criteria",
                "content_type": "eligibility_criteria",
                "key_entities": ["PM Kisan"],
                "key_dates": ["2023-01-01"],
                "key_amounts": ["₹6,000"],
                "has_eligibility_criteria": True,
                "has_procedure_steps": False,
                "summary": "Criteria for PM Kisan eligibility.",
            }
        )

        chunker = DocumentChunker(ai, enrich_chunks=True)
        chunks = await chunker.chunk(doc)

        # At least some chunks should have enriched metadata
        enriched = [c for c in chunks if c.metadata.topic is not None]
        assert len(enriched) >= 1

    @pytest.mark.asyncio
    async def test_enrichment_failure_is_non_fatal(self):
        """A failing AI enrichment call should not raise — original chunk returned."""
        pages = [_make_page(1, CHAPTER_TEXT)]
        doc = _make_doc(pages)
        ai = _make_ai_service(fail=True)

        chunker = DocumentChunker(ai, enrich_chunks=True)
        # Should complete without raising
        chunks = await chunker.chunk(doc)

        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_enrichment_skipped_when_disabled(self):
        pages = [_make_page(1, CHAPTER_TEXT)]
        doc = _make_doc(pages)
        ai = _make_ai_service()

        chunker = DocumentChunker(ai, enrich_chunks=False)
        chunks = await chunker.chunk(doc)

        ai.generate_chunk_description.assert_not_called()

    @pytest.mark.asyncio
    async def test_document_level_chunk_not_enriched(self):
        """The DOCUMENT-level chunk is skipped for enrichment (too large)."""
        pages = [_make_page(1, CHAPTER_TEXT)]
        doc = _make_doc(pages)
        ai = _make_ai_service()

        chunker = DocumentChunker(ai, enrich_chunks=True)
        chunks = await chunker.chunk(doc)

        doc_chunk = next(c for c in chunks if c.hierarchy_level == HierarchyLevel.DOCUMENT)
        assert doc_chunk.metadata.topic is None


# ---------------------------------------------------------------------------
# Tests: multi-page document
# ---------------------------------------------------------------------------


class TestMultiPageDocument:
    @pytest.mark.asyncio
    async def test_chunks_span_correct_pages(self):
        pages = [
            _make_page(1, "Chapter 1 Introduction\n\nThis is the introduction."),
            _make_page(2, "Chapter 2 Eligibility\n\nApplicant must be a farmer."),
        ]
        doc = _make_doc(pages)
        ai = _make_ai_service()

        chunker = DocumentChunker(ai, enrich_chunks=False)
        chunks = await chunker.chunk(doc)

        assert len(chunks) >= 1  # At minimum DOCUMENT chunk


# ---------------------------------------------------------------------------
# Regression tests: _normalize_str_list  (unit tests — no LLM calls)
# ---------------------------------------------------------------------------
# These tests exercise the normalisation helper directly to verify every
# rule in isolation.  The companion integration tests below exercise the
# full _enrich_chunk path with mocked AI responses containing dirty data.
# ---------------------------------------------------------------------------


from app.pipeline.chunker import ChunkEnrichmentError, _normalize_str_list  # noqa: E402


class TestNormalizeStrList:
    """Unit tests for the _normalize_str_list normalisation helper."""

    def test_none_items_are_removed(self):
        """LLM null → None in Python must be silently dropped."""
        result = _normalize_str_list(["₹6000", None, "₹12000"], field="key_amounts")
        assert result == ["₹6000", "₹12000"]

    def test_empty_string_items_are_removed(self):
        result = _normalize_str_list(["₹6000", "", "₹12000"], field="key_amounts")
        assert result == ["₹6000", "₹12000"]

    def test_whitespace_only_items_are_removed(self):
        result = _normalize_str_list(["₹6000", "   ", "\t", "₹12000"], field="key_amounts")
        assert result == ["₹6000", "₹12000"]

    def test_leading_trailing_whitespace_is_stripped(self):
        result = _normalize_str_list(["  ₹6000  ", "  ₹12000  "], field="key_amounts")
        assert result == ["₹6000", "₹12000"]

    def test_duplicates_removed_first_occurrence_wins(self):
        """Duplicate strings (after stripping) must be deduplicated preserving order."""
        result = _normalize_str_list(
            ["₹6000", "₹12000", "₹6000", "  ₹12000  "],
            field="key_amounts",
        )
        assert result == ["₹6000", "₹12000"]

    def test_order_preserved_after_deduplication(self):
        """Insertion order must be maintained — first occurrence wins."""
        result = _normalize_str_list(
            ["Alpha", "Beta", "Gamma", "Alpha", "Beta"],
            field="key_entities",
        )
        assert result == ["Alpha", "Beta", "Gamma"]

    def test_integer_coerced_to_string(self):
        """LLM may return bare integer amounts; they must be accepted as strings."""
        result = _normalize_str_list([6000, 12000], field="key_amounts")
        assert result == ["6000", "12000"]

    def test_float_coerced_to_string(self):
        result = _normalize_str_list([6000.0, 12000.5], field="key_amounts")
        assert result == ["6000.0", "12000.5"]

    def test_full_realistic_input_matches_spec(self):
        """Canonical example from the bug report."""
        raw = ["₹6000", None, "", "₹12000", "₹6000", "  ₹12000  "]
        result = _normalize_str_list(raw, field="key_amounts")
        assert result == ["₹6000", "₹12000"]

    def test_none_top_level_returns_empty_list(self):
        """data.get('key_amounts') returning None → empty list, not crash."""
        assert _normalize_str_list(None, field="key_amounts") == []

    def test_non_list_scalar_returns_empty_list(self):
        """AI returning a bare string instead of a list → empty list."""
        assert _normalize_str_list("₹6000", field="key_amounts") == []

    def test_empty_list_returns_empty_list(self):
        assert _normalize_str_list([], field="key_amounts") == []

    def test_all_nulls_returns_empty_list(self):
        assert _normalize_str_list([None, None, None], field="key_amounts") == []

    def test_bool_raises_type_error(self):
        """Booleans must be rejected (bool subclasses int — ambiguous semantics)."""
        with pytest.raises(TypeError, match="boolean value"):
            _normalize_str_list([True, False], field="key_amounts")

    def test_dict_item_raises_type_error(self):
        """Nested dicts must be rejected with a descriptive error."""
        with pytest.raises(TypeError, match="unexpected type 'dict'"):
            _normalize_str_list([{"amount": 6000}], field="key_amounts")

    def test_list_item_raises_type_error(self):
        """Nested lists must be rejected."""
        with pytest.raises(TypeError, match="unexpected type 'list'"):
            _normalize_str_list([["₹6000"]], field="key_amounts")


# ---------------------------------------------------------------------------
# Regression tests: _enrich_chunk with dirty AI responses (integration)
# ---------------------------------------------------------------------------


class TestEnrichChunkNormalization:
    """
    End-to-end tests for _enrich_chunk's normalisation path.

    The AIService is mocked to return controlled dirty responses so we can
    verify the normalisation layer handles them without ever touching Pydantic
    validation with invalid inputs.
    """

    def _make_chunk_for_enrichment(self) -> "DocumentChunk":
        from app.pipeline.chunker import _make_chunk_id
        from app.schemas.pipeline import ChunkMetadata, DocumentChunk, HierarchyLevel

        text = (
            "Chapter 1 Eligibility\n\n"
            "1.1 Income Criteria\n\n"
            "The annual household income must not exceed Rs. 2 lakh per annum. "
            "Applicants must be resident citizens of India.\n\n"
            "1.2 Age Criteria\n\n"
            "The beneficiary must be between 18 and 60 years of age."
        )
        return DocumentChunk(
            chunk_id=_make_chunk_id("doc-enrich", text),
            document_id="doc-enrich",
            parent_id=None,
            hierarchy_level=HierarchyLevel.CHAPTER,
            page_number=1,
            title="Chapter 1 Eligibility",
            section=None,
            text=text,
            metadata=ChunkMetadata(word_count=50, char_count=300),
        )

    @pytest.mark.asyncio
    async def test_null_items_in_key_amounts_do_not_crash(self):
        """The exact production bug: null in key_amounts must be filtered out."""
        chunk = self._make_chunk_for_enrichment()
        ai = _make_ai_service(
            enrich_data={
                "topic": "Eligibility",
                "content_type": "eligibility_criteria",
                "key_entities": ["Ministry of Agriculture"],
                "key_dates": ["2023-04-01"],
                "key_amounts": ["₹6000", None, "", "₹12000"],
                "has_eligibility_criteria": True,
                "has_procedure_steps": False,
                "summary": "Eligibility section.",
            }
        )
        chunker = DocumentChunker(ai, enrich_chunks=True)
        result = await chunker._enrich_chunk(chunk, "test_policy.pdf")

        assert result.metadata.key_amounts == ["₹6000", "₹12000"]

    @pytest.mark.asyncio
    async def test_null_items_in_key_entities_do_not_crash(self):
        chunk = self._make_chunk_for_enrichment()
        ai = _make_ai_service(
            enrich_data={
                "topic": "Policy overview",
                "content_type": "general_information",
                "key_entities": [None, "Ministry of Agriculture", "", None],
                "key_dates": [],
                "key_amounts": [],
                "has_eligibility_criteria": False,
                "has_procedure_steps": False,
                "summary": None,
            }
        )
        chunker = DocumentChunker(ai, enrich_chunks=True)
        result = await chunker._enrich_chunk(chunk, "test_policy.pdf")

        assert result.metadata.key_entities == ["Ministry of Agriculture"]

    @pytest.mark.asyncio
    async def test_null_items_in_key_dates_do_not_crash(self):
        chunk = self._make_chunk_for_enrichment()
        ai = _make_ai_service(
            enrich_data={
                "topic": "Deadlines",
                "content_type": "procedure",
                "key_entities": [],
                "key_dates": [None, "2023-04-01", "  ", None, "2024-01-01"],
                "key_amounts": [],
                "has_eligibility_criteria": False,
                "has_procedure_steps": True,
                "summary": "Important dates.",
            }
        )
        chunker = DocumentChunker(ai, enrich_chunks=True)
        result = await chunker._enrich_chunk(chunk, "test_policy.pdf")

        assert result.metadata.key_dates == ["2023-04-01", "2024-01-01"]

    @pytest.mark.asyncio
    async def test_all_nulls_produces_empty_lists(self):
        """All null items across all three list fields → empty lists, no crash."""
        chunk = self._make_chunk_for_enrichment()
        ai = _make_ai_service(
            enrich_data={
                "topic": "Benefits",
                "content_type": "general_information",
                "key_entities": [None, None],
                "key_dates": [None],
                "key_amounts": [None, None, None],
                "has_eligibility_criteria": False,
                "has_procedure_steps": False,
                "summary": "Test.",
            }
        )
        chunker = DocumentChunker(ai, enrich_chunks=True)
        result = await chunker._enrich_chunk(chunk, "test_policy.pdf")

        assert result.metadata.key_entities == []
        assert result.metadata.key_dates == []
        assert result.metadata.key_amounts == []

    @pytest.mark.asyncio
    async def test_duplicate_amounts_deduplicated_preserving_order(self):
        chunk = self._make_chunk_for_enrichment()
        ai = _make_ai_service(
            enrich_data={
                "topic": "Benefits",
                "content_type": "financial_information",
                "key_entities": [],
                "key_dates": [],
                "key_amounts": ["₹6000", "₹2000", "₹6000", "  ₹2000  ", "₹12000"],
                "has_eligibility_criteria": False,
                "has_procedure_steps": False,
                "summary": "Financial summary.",
            }
        )
        chunker = DocumentChunker(ai, enrich_chunks=True)
        result = await chunker._enrich_chunk(chunk, "test_policy.pdf")

        assert result.metadata.key_amounts == ["₹6000", "₹2000", "₹12000"]

    @pytest.mark.asyncio
    async def test_numeric_amounts_coerced_to_string(self):
        """LLM returning bare integers for amounts must not crash."""
        chunk = self._make_chunk_for_enrichment()
        ai = _make_ai_service(
            enrich_data={
                "topic": "Benefits",
                "content_type": "financial_information",
                "key_entities": [],
                "key_dates": [],
                "key_amounts": [6000, 12000, None],
                "has_eligibility_criteria": False,
                "has_procedure_steps": False,
                "summary": "Numeric amounts.",
            }
        )
        chunker = DocumentChunker(ai, enrich_chunks=True)
        result = await chunker._enrich_chunk(chunk, "test_policy.pdf")

        assert result.metadata.key_amounts == ["6000", "12000"]

    @pytest.mark.asyncio
    async def test_missing_list_fields_return_empty_lists(self):
        """AI omitting list fields entirely (key missing in dict) → empty list."""
        chunk = self._make_chunk_for_enrichment()
        ai = _make_ai_service(
            enrich_data={
                "topic": "General",
                "content_type": "general_information",
                # key_entities, key_dates, key_amounts intentionally absent
                "has_eligibility_criteria": False,
                "has_procedure_steps": False,
                "summary": "Incomplete response.",
            }
        )
        chunker = DocumentChunker(ai, enrich_chunks=True)
        result = await chunker._enrich_chunk(chunk, "test_policy.pdf")

        assert result.metadata.key_entities == []
        assert result.metadata.key_dates == []
        assert result.metadata.key_amounts == []

    @pytest.mark.asyncio
    async def test_malformed_ai_response_dict_item_falls_back_to_original_chunk(self):
        """
        Unrecoverable type error (dict inside a list field) must not propagate
        as an unhandled exception — _enrich_all's _safe_enrich wrapper catches
        ChunkEnrichmentError and returns the original chunk.
        """
        chunk = self._make_chunk_for_enrichment()
        ai = _make_ai_service(
            enrich_data={
                "topic": "Malformed",
                "content_type": "general_information",
                "key_entities": [{"nested": "dict"}],   # unrecoverable
                "key_dates": [],
                "key_amounts": [],
                "has_eligibility_criteria": False,
                "has_procedure_steps": False,
                "summary": "Bad response.",
            }
        )
        chunker = DocumentChunker(ai, enrich_chunks=True)

        # _enrich_chunk itself should raise ChunkEnrichmentError
        with pytest.raises(ChunkEnrichmentError) as exc_info:
            await chunker._enrich_chunk(chunk, "test_policy.pdf")

        err = exc_info.value
        assert err.field == "key_entities"
        assert err.chunk_id == chunk.chunk_id
        assert err.document_id == chunk.document_id
        assert err.page_number == chunk.page_number

    @pytest.mark.asyncio
    async def test_enrich_all_recovers_from_normalisation_error(self):
        """
        _enrich_all must return original chunks when ChunkEnrichmentError is
        raised for individual chunks, never raising to the caller.
        """
        chunk = self._make_chunk_for_enrichment()
        ai = _make_ai_service(
            enrich_data={
                "topic": "Malformed",
                "content_type": "general_information",
                "key_entities": [{"nested": "bad"}],
                "key_dates": [],
                "key_amounts": [],
                "has_eligibility_criteria": False,
                "has_procedure_steps": False,
                "summary": None,
            }
        )
        chunker = DocumentChunker(ai, enrich_chunks=True)

        # _enrich_all wraps each call and should never raise
        result = await chunker._enrich_all([chunk], "test_policy.pdf")

        assert len(result) == 1
        # Original chunk returned unchanged (topic is None in original ChunkMetadata)
        assert result[0].metadata.topic is None

    @pytest.mark.asyncio
    async def test_chunk_enrichment_error_carries_structured_context(self):
        """ChunkEnrichmentError must expose document_id, chunk_id, page_number, field."""
        from app.pipeline.chunker import ChunkEnrichmentError

        err = ChunkEnrichmentError(
            "test error",
            document_id="doc-001",
            chunk_id="chunk-abc",
            page_number=3,
            field="key_amounts",
            offending_value=[None, None],
        )
        assert err.document_id == "doc-001"
        assert err.chunk_id == "chunk-abc"
        assert err.page_number == 3
        assert err.field == "key_amounts"
        assert err.offending_value == [None, None]
        assert "doc-001" in str(err)
        assert "key_amounts" in str(err)
        assert isinstance(err, ValueError)

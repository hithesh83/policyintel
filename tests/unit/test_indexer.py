"""
Unit tests for DocumentIndexer.

All tests use a mocked AIService — no real LLM calls, no Qdrant connection.

Covers
------
- VectorDocument structure (all required fields present)
- vector_id equals chunk_id (1:1 mapping for Qdrant)
- vector dimension is correct (_EMBEDDING_DIM)
- payload contains expected metadata fields
- payload flattening (no nested dicts)
- chunk → extraction result mapping by chunk_id
- Missing extraction result → payload still assembled from chunk metadata
- Embedding failure → placeholder zero vector, no exception
- Concurrent processing (multiple chunks)
- Extraction error chunks → payload assembled, no crash
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.llm.exceptions import LLMError
from app.llm.models import GenerateResponse
from app.pipeline.indexer import (
    DocumentIndexer,
    _EMBEDDING_DIM,
    _build_payload,
    _placeholder_vector,
)
from app.schemas.pipeline import (
    Benefit,
    ChunkMetadata,
    DocumentChunk,
    EligibilityCriterion,
    ExtractionResult,
    ExtractedEntities,
    HierarchyLevel,
    VectorDocument,
)
from app.services.ai_service import AIService


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


def _generate_response() -> GenerateResponse:
    return GenerateResponse(
        text="Embedding representation placeholder.",
        model="test-model",
        request_id="req-test",
    )


def _make_ai_service(*, fail: bool = False) -> AIService:
    ai = MagicMock(spec=AIService)
    if fail:
        ai.generate = AsyncMock(side_effect=LLMError("Connection failed"))
    else:
        ai.generate = AsyncMock(return_value=_generate_response())
    return ai


def _make_chunk(
    chunk_id: str = "chunk-001",
    doc_id: str = "doc-001",
    text: str = "Eligible farmers receive Rs. 6,000 per year from PM Kisan.",
    hierarchy_level: HierarchyLevel = HierarchyLevel.SECTION,
    title: str | None = "Eligibility Section",
    section: str | None = "3.1",
    parent_id: str | None = "parent-001",
    page_number: int = 2,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        parent_id=parent_id,
        hierarchy_level=hierarchy_level,
        page_number=page_number,
        title=title,
        section=section,
        text=text,
        metadata=ChunkMetadata(
            content_type="eligibility_criteria",
            topic="Farmer eligibility",
            key_entities=["PM Kisan", "Ministry of Agriculture"],
            key_dates=["2023-04-01"],
            key_amounts=["Rs. 6,000"],
            has_eligibility_criteria=True,
            has_procedure_steps=False,
            summary="Farmers are eligible for Rs. 6,000.",
            word_count=11,
            char_count=60,
        ),
    )


def _make_extraction(
    chunk_id: str = "chunk-001",
    doc_id: str = "doc-001",
    hierarchy_level: HierarchyLevel = HierarchyLevel.SECTION,
    extraction_error: str | None = None,
) -> ExtractionResult:
    entities = ExtractedEntities(
        scheme_name="PM Kisan",
        issuing_ministry="Ministry of Agriculture and Farmers Welfare",
        implementing_organizations=["State Agriculture Departments"],
        supporting_agencies=["NABARD", "SBI"],
        departments=["Department of Agriculture, Cooperation & Farmers Welfare"],
        stakeholders=["DBT Mission"],
        funding_pattern="100% Central",
        policy_type="central_scheme",
        geographic_scope="national",
        states=["Rajasthan", "Maharashtra"],
        effective_date="2019-02-01",
        beneficiary_categories=["small farmers"],
        eligible_categories=["marginal farmers"],
        income_limit_annual=200000.0,
        age_min=18.0,
        age_max=60.0,
        is_direct_benefit_transfer=True,
        total_annual_benefit_inr=6000.0,
        benefits=[
            Benefit(
                benefit_type="cash_transfer",
                description="Rs. 6,000 per year",
                amount_inr=6000.0,
                frequency="annual",
            )
        ],
        deadlines=["31 March 2024"],
        documents_required=["Aadhaar", "Land record"],
        amendment_references=[],
    )
    return ExtractionResult(
        chunk_id=chunk_id,
        document_id=doc_id,
        hierarchy_level=hierarchy_level,
        page_number=2,
        section="3.1",
        title="Eligibility Section",
        entities=entities,
        raw_text="Eligible farmers receive Rs. 6,000 per year.",
        extraction_error=extraction_error,
        model_used="test-model",
    )


# ---------------------------------------------------------------------------
# Tests: placeholder vector
# ---------------------------------------------------------------------------


class TestPlaceholderVector:
    def test_placeholder_has_correct_dimension(self):
        v = _placeholder_vector()
        assert len(v) == _EMBEDDING_DIM

    def test_placeholder_is_all_zeros(self):
        v = _placeholder_vector()
        assert all(x == 0.0 for x in v)

    def test_placeholder_is_list_of_floats(self):
        v = _placeholder_vector()
        assert all(isinstance(x, float) for x in v)


# ---------------------------------------------------------------------------
# Tests: VectorDocument structure
# ---------------------------------------------------------------------------


class TestVectorDocumentStructure:
    @pytest.mark.asyncio
    async def test_returns_one_vector_doc_per_chunk(self):
        chunks = [_make_chunk("c1"), _make_chunk("c2")]
        extractions = [_make_extraction("c1"), _make_extraction("c2")]
        ai = _make_ai_service()

        indexer = DocumentIndexer(ai)
        docs = await indexer.index(chunks, extractions)

        assert len(docs) == 2

    @pytest.mark.asyncio
    async def test_vector_doc_is_pydantic_model(self):
        chunks = [_make_chunk()]
        extractions = [_make_extraction()]
        ai = _make_ai_service()

        indexer = DocumentIndexer(ai)
        docs = await indexer.index(chunks, extractions)

        assert isinstance(docs[0], VectorDocument)

    @pytest.mark.asyncio
    async def test_vector_id_equals_chunk_id(self):
        chunk = _make_chunk("my-chunk-id")
        extraction = _make_extraction("my-chunk-id")
        ai = _make_ai_service()

        indexer = DocumentIndexer(ai)
        docs = await indexer.index([chunk], [extraction])

        assert docs[0].vector_id == "my-chunk-id"
        assert docs[0].chunk_id == "my-chunk-id"

    @pytest.mark.asyncio
    async def test_document_id_propagated(self):
        chunk = _make_chunk(doc_id="my-doc-id")
        extraction = _make_extraction(doc_id="my-doc-id")
        ai = _make_ai_service()

        indexer = DocumentIndexer(ai)
        docs = await indexer.index([chunk], [extraction])

        assert docs[0].document_id == "my-doc-id"

    @pytest.mark.asyncio
    async def test_text_field_contains_chunk_text(self):
        chunk = _make_chunk(text="Special policy text for this test.")
        extraction = _make_extraction()
        ai = _make_ai_service()

        indexer = DocumentIndexer(ai)
        docs = await indexer.index([chunk], [extraction])

        assert docs[0].text == "Special policy text for this test."

    @pytest.mark.asyncio
    async def test_vector_dimension_is_correct(self):
        chunk = _make_chunk()
        extraction = _make_extraction()
        ai = _make_ai_service()

        indexer = DocumentIndexer(ai)
        docs = await indexer.index([chunk], [extraction])

        assert len(docs[0].vector) == _EMBEDDING_DIM


# ---------------------------------------------------------------------------
# Tests: payload contents
# ---------------------------------------------------------------------------


class TestPayloadContents:
    @pytest.mark.asyncio
    async def test_payload_has_chunk_identity_fields(self):
        chunk = _make_chunk(
            chunk_id="chunk-abc",
            doc_id="doc-xyz",
            parent_id="parent-001",
        )
        extraction = _make_extraction("chunk-abc", "doc-xyz")
        ai = _make_ai_service()

        indexer = DocumentIndexer(ai)
        docs = await indexer.index([chunk], [extraction])
        payload = docs[0].payload

        assert payload["chunk_id"] == "chunk-abc"
        assert payload["document_id"] == "doc-xyz"
        assert payload["parent_id"] == "parent-001"

    @pytest.mark.asyncio
    async def test_payload_has_hierarchy_level(self):
        chunk = _make_chunk(hierarchy_level=HierarchyLevel.CLAUSE)
        extraction = _make_extraction(hierarchy_level=HierarchyLevel.CLAUSE)
        ai = _make_ai_service()

        indexer = DocumentIndexer(ai)
        docs = await indexer.index([chunk], [extraction])

        assert docs[0].payload["hierarchy_level"] == "clause"

    @pytest.mark.asyncio
    async def test_payload_has_extraction_entities(self):
        chunk = _make_chunk()
        extraction = _make_extraction()
        ai = _make_ai_service()

        indexer = DocumentIndexer(ai)
        docs = await indexer.index([chunk], [extraction])
        payload = docs[0].payload

        assert payload["scheme_name"] == "PM Kisan"
        assert payload["issuing_ministry"] == "Ministry of Agriculture and Farmers Welfare"
        assert "State Agriculture Departments" in payload["implementing_organizations"]
        assert "NABARD" in payload["supporting_agencies"]
        assert "Rajasthan" in payload["states"]
        assert payload["income_limit_annual"] == 200000.0

    @pytest.mark.asyncio
    async def test_payload_has_metadata_fields(self):
        chunk = _make_chunk()
        extraction = _make_extraction()
        ai = _make_ai_service()

        indexer = DocumentIndexer(ai)
        docs = await indexer.index([chunk], [extraction])
        payload = docs[0].payload

        assert payload["has_eligibility_criteria"] is True
        assert payload["has_procedure_steps"] is False
        assert payload["summary"] == "Farmers are eligible for Rs. 6,000."
        assert payload["word_count"] == 11

    @pytest.mark.asyncio
    async def test_payload_has_benefit_types(self):
        chunk = _make_chunk()
        extraction = _make_extraction()
        ai = _make_ai_service()

        indexer = DocumentIndexer(ai)
        docs = await indexer.index([chunk], [extraction])

        assert "cash_transfer" in docs[0].payload["benefit_types"]

    @pytest.mark.asyncio
    async def test_payload_no_none_values(self):
        """Payload should not contain explicit None values (cleaner Qdrant storage)."""
        chunk = _make_chunk()
        extraction = _make_extraction()
        ai = _make_ai_service()

        indexer = DocumentIndexer(ai)
        docs = await indexer.index([chunk], [extraction])

        for key, value in docs[0].payload.items():
            assert value is not None, f"Payload key '{key}' has None value"


# ---------------------------------------------------------------------------
# Tests: chunk → extraction mapping
# ---------------------------------------------------------------------------


class TestChunkExtractionMapping:
    @pytest.mark.asyncio
    async def test_chunks_without_matching_extraction_still_indexed(self):
        """A chunk with no matching ExtractionResult should still produce a VectorDocument."""
        chunk = _make_chunk("orphan-chunk")
        ai = _make_ai_service()

        indexer = DocumentIndexer(ai)
        docs = await indexer.index([chunk], [])  # No extractions

        assert len(docs) == 1
        assert docs[0].chunk_id == "orphan-chunk"
        # Extraction-derived fields should be absent
        assert "scheme_name" not in docs[0].payload

    @pytest.mark.asyncio
    async def test_extraction_error_chunks_still_indexed(self):
        """Chunks with extraction_error should still be indexed (error in payload omitted)."""
        chunk = _make_chunk()
        extraction = _make_extraction(extraction_error="LLM failed")
        ai = _make_ai_service()

        indexer = DocumentIndexer(ai)
        docs = await indexer.index([chunk], [extraction])

        assert len(docs) == 1
        # Extraction fields should be absent for errored extractions
        assert "scheme_name" not in docs[0].payload


# ---------------------------------------------------------------------------
# Tests: embedding failure isolation
# ---------------------------------------------------------------------------


class TestEmbeddingFailure:
    @pytest.mark.asyncio
    async def test_llm_error_falls_back_to_placeholder_vector(self):
        chunk = _make_chunk()
        extraction = _make_extraction()
        ai = _make_ai_service(fail=True)

        indexer = DocumentIndexer(ai)
        docs = await indexer.index([chunk], [extraction])

        assert len(docs) == 1
        assert len(docs[0].vector) == _EMBEDDING_DIM
        assert all(x == 0.0 for x in docs[0].vector)

    @pytest.mark.asyncio
    async def test_embedding_failure_does_not_raise(self):
        chunks = [_make_chunk("c1"), _make_chunk("c2")]
        extractions = [_make_extraction("c1"), _make_extraction("c2")]
        ai = _make_ai_service(fail=True)

        indexer = DocumentIndexer(ai)
        # Should not raise
        docs = await indexer.index(chunks, extractions)
        assert len(docs) == 2


# ---------------------------------------------------------------------------
# Tests: _build_payload directly
# ---------------------------------------------------------------------------


class TestBuildPayload:
    def test_payload_without_extraction(self):
        chunk = _make_chunk()
        payload = _build_payload(chunk, None)

        assert payload["chunk_id"] == chunk.chunk_id
        assert payload["hierarchy_level"] == "section"
        assert "scheme_name" not in payload

    def test_payload_with_extraction(self):
        chunk = _make_chunk()
        extraction = _make_extraction()
        payload = _build_payload(chunk, extraction)

        assert payload["scheme_name"] == "PM Kisan"

    def test_payload_excludes_none_values(self):
        chunk = _make_chunk(title=None, section=None)
        extraction = _make_extraction()
        payload = _build_payload(chunk, extraction)

        assert "title" not in payload
        assert "section" not in payload

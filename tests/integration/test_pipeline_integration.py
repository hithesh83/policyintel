"""
Integration tests for the Week 2 document understanding pipeline.

These tests exercise the full pipeline end-to-end:
    ParsedDocument → DocumentChunker → PolicyExtractor → GraphBuilder + DocumentIndexer

All LLM calls are mocked — no real Ollama / AI service required.
These tests validate:
  - Data flows correctly through all four modules
  - Each module's output is the correct input type for the next
  - The full pipeline completes without errors
  - GraphBundle and VectorDocument counts are reasonable
  - No Pydantic validation errors across the full pipeline
  - Mocked parser output (simulating parser.py delivering a ParsedDocument)

Run with:
    pytest tests/integration/test_pipeline_integration.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.llm.models import ExtractionResponse, GenerateResponse
from app.pipeline.chunker import DocumentChunker
from app.pipeline.extractor import PolicyExtractor
from app.pipeline.graph_builder import GraphBuilder
from app.pipeline.indexer import DocumentIndexer
from app.schemas.pipeline import (
    GraphBundle,
    HierarchyLevel,
    NodeType,
    ParsedDocument,
    ParsedPage,
    RelationshipType,
    VectorDocument,
)
from app.services.ai_service import AIService


# ---------------------------------------------------------------------------
# Mock parser output
# ---------------------------------------------------------------------------

PM_KISAN_TEXT = """\
PRADHAN MANTRI KISAN SAMMAN NIDHI (PM-KISAN) SCHEME
Ministry of Agriculture & Farmers Welfare
Government of India

Chapter 1 Background and Objectives

The Pradhan Mantri Kisan Samman Nidhi (PM-KISAN) is a central sector scheme
that provides income support to all landholding farmers' families in the country
to supplement their financial needs.

1.1 Objectives

The scheme aims to supplement the financial needs of the Small and Marginal
Farmers (SMF) in procuring various inputs to ensure proper crop health and
appropriate yields.

Chapter 2 Eligibility Criteria

All land-holding farmers' families in the country, with cultivable land, are
eligible to get the financial benefit under the scheme.

2.1 Income Eligibility

Annual household income must not exceed Rs. 2,00,000 per annum.

2.2 Age and Residency

Applicants must be resident citizens of India and must be between 18 and 70
years of age.

(a) Aadhaar card is mandatory for all beneficiaries.

(b) Land records must be updated in the land registration database.

Chapter 3 Financial Benefits and Amount

3.1 Quantum of Benefit

An amount of Rs. 6,000 per year is provided to each eligible farmer family
as a direct benefit transfer.

The amount is released in three equal instalments of Rs. 2,000 each, every
four months.

3.2 Mode of Payment

The benefit amount is directly credited to the bank account of the beneficiary
farmers through the Direct Benefit Transfer (DBT) mechanism.

Chapter 4 Application Procedure

4.1 Applying Online

Eligible farmers can register at the PM Kisan portal (pmkisan.gov.in) or
through Common Service Centres.

Deadline for applications is 31 March of the relevant financial year.

Chapter 5 Amendments and Updates

This scheme supersedes PM Kisan v1.0 issued in February 2019.

Amendment Circular No. 10/2021 dated 15 June 2021 expanded the coverage
to include all farmers without any land size restriction.
"""


def _make_parsed_document() -> ParsedDocument:
    """Simulate the output of parser.py for PM Kisan scheme document."""
    page1_text = "\n".join(PM_KISAN_TEXT.split("\n")[:50])
    page2_text = "\n".join(PM_KISAN_TEXT.split("\n")[50:])

    return ParsedDocument(
        document_id="pm-kisan-doc-001",
        filename="PM_Kisan_SOP_2024.pdf",
        title="Pradhan Mantri Kisan Samman Nidhi Scheme",
        total_pages=2,
        pages=[
            ParsedPage(page_number=1, text=page1_text),
            ParsedPage(page_number=2, text=page2_text),
        ],
        raw_text=PM_KISAN_TEXT,
        file_metadata={"size_bytes": 204800, "mime_type": "application/pdf"},
    )


# ---------------------------------------------------------------------------
# Mock AIService factory
# ---------------------------------------------------------------------------


def _make_full_ai_service() -> AIService:
    """
    Build a fully mocked AIService covering all pipeline methods.
    """
    ai = MagicMock(spec=AIService)

    # --- generate_chunk_description (chunker enrichment) ---
    ai.generate_chunk_description = AsyncMock(
        return_value=ExtractionResponse(
            data={
                "topic": "PM Kisan policy overview",
                "content_type": "general_information",
                "key_entities": ["PM Kisan", "Ministry of Agriculture"],
                "key_dates": ["2019-02-01"],
                "key_amounts": ["Rs. 6,000"],
                "has_eligibility_criteria": True,
                "has_procedure_steps": False,
                "summary": "PM Kisan provides Rs. 6,000 per year to eligible farmers.",
            },
            raw_output="{}",
            model="qwen2.5:7b",
            request_id="req-enrich",
        )
    )

    # --- extract_policy_metadata (extractor) ---
    ai.extract_policy_metadata = AsyncMock(
        return_value=ExtractionResponse(
            data={
                "document_title": "Pradhan Mantri Kisan Samman Nidhi",
                "issuing_ministry": "Ministry of Agriculture & Farmers Welfare",
                "implementing_organizations": [
                    "State Agriculture Departments",
                    "District Collectors",
                ],
                "supporting_agencies": ["NABARD", "SBI"],
                "departments": [
                    "Department of Agriculture, Cooperation & Farmers Welfare"
                ],
                "stakeholders": ["DBT Mission"],
                "funding_pattern": "100% Central",
                "scheme_code": "PM-KISAN",
                "effective_date": "2019-02-01",
                "issue_date": "2019-01-01",
                "policy_type": "central_scheme",
                "target_beneficiaries": ["farmers", "agricultural households"],
                "geographic_scope": "national",
                "state_name": None,
                "supersedes": ["PM Kisan v1.0"],
                "language": "English",
            },
            raw_output="{}",
            model="qwen2.5:7b",
            request_id="req-meta",
        )
    )

    # --- extract_eligibility (extractor) ---
    ai.extract_eligibility = AsyncMock(
        return_value=ExtractionResponse(
            data={
                "criteria": [
                    {
                        "criterion_type": "income",
                        "description": "Annual household income must not exceed Rs. 2,00,000",
                        "condition": "income <= 200000",
                        "min_value": None,
                        "max_value": 200000,
                        "unit": "INR",
                        "mandatory": True,
                    },
                    {
                        "criterion_type": "age",
                        "description": "Applicant must be between 18 and 70 years",
                        "condition": "18 <= age <= 70",
                        "min_value": 18,
                        "max_value": 70,
                        "unit": "years",
                        "mandatory": True,
                    },
                ],
                "eligible_categories": ["small and marginal farmers", "landholding farmers"],
                "ineligible_categories": ["government employees", "income taxpayers"],
                "income_limit_annual": 200000,
                "age_min": 18,
                "age_max": 70,
            },
            raw_output="{}",
            model="qwen2.5:7b",
            request_id="req-elig",
        )
    )

    # --- extract_benefits (extractor) ---
    ai.extract_benefits = AsyncMock(
        return_value=ExtractionResponse(
            data={
                "benefits": [
                    {
                        "benefit_type": "cash_transfer",
                        "description": "Rs. 6,000 per year via Direct Benefit Transfer",
                        "amount_inr": 6000,
                        "frequency": "annual",
                        "duration_months": None,
                        "conditions": ["Aadhaar mandatory", "bank account required"],
                    }
                ],
                "total_annual_benefit_inr": 6000,
                "is_direct_benefit_transfer": True,
            },
            raw_output="{}",
            model="qwen2.5:7b",
            request_id="req-benefits",
        )
    )

    # --- extract_json (extractor full entity pass) ---
    ai.extract_json = AsyncMock(
        return_value=ExtractionResponse(
            data={
                "scheme_name": "PM Kisan",
                "issuing_ministry": "Ministry of Agriculture & Farmers Welfare",
                "implementing_organizations": ["State Agriculture Departments"],
                "supporting_agencies": ["NABARD"],
                "departments": [
                    "Department of Agriculture, Cooperation & Farmers Welfare"
                ],
                "funding_pattern": "100% Central",
                "stakeholders": [],
                "states": [],
                "deadlines": ["31 March"],
                "key_dates": ["2019-02-01", "2021-06-15"],
                "key_amounts": ["Rs. 6,000", "Rs. 2,000"],
                "beneficiary_categories": ["farmers", "agricultural households"],
                "relationships": ["PM Kisan administered by Ministry of Agriculture"],
                "documents_required": ["Aadhaar card", "Land records", "Bank account"],
                "amendment_references": ["Circular No. 10/2021"],
                "effective_date": "2019-02-01",
                "issue_date": "2019-01-01",
                "geographic_scope": "national",
                "policy_type": "central_scheme",
            },
            raw_output="{}",
            model="qwen2.5:7b",
            request_id="req-full",
        )
    )

    # --- generate (indexer embedding) ---
    ai.generate = AsyncMock(
        return_value=GenerateResponse(
            text="Semantic embedding representation of PM Kisan policy chunk.",
            model="qwen2.5:7b",
            request_id="req-embed",
        )
    )

    return ai


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFullPipelineIntegration:
    """
    Full pipeline integration tests using mocked AIService.

    These tests exercise the entire pipeline from ParsedDocument to
    VectorDocument + GraphBundle without any real network calls.
    """

    @pytest.fixture
    def parsed_doc(self) -> ParsedDocument:
        return _make_parsed_document()

    @pytest.fixture
    def ai_service(self) -> AIService:
        return _make_full_ai_service()

    @pytest.mark.asyncio
    async def test_chunker_produces_chunks(self, parsed_doc, ai_service):
        """Step 1: DocumentChunker produces a non-empty list of DocumentChunks."""
        chunker = DocumentChunker(ai_service, enrich_chunks=True)
        chunks = await chunker.chunk(parsed_doc)

        assert len(chunks) >= 1
        # Should have at minimum the DOCUMENT-level chunk
        doc_chunks = [c for c in chunks if c.hierarchy_level == HierarchyLevel.DOCUMENT]
        assert len(doc_chunks) == 1

    @pytest.mark.asyncio
    async def test_extractor_produces_results(self, parsed_doc, ai_service):
        """Step 2: PolicyExtractor produces an ExtractionResult per chunk."""
        chunker = DocumentChunker(ai_service, enrich_chunks=False)
        chunks = await chunker.chunk(parsed_doc)

        extractor = PolicyExtractor(ai_service)
        results = await extractor.extract(chunks)

        assert len(results) == len(chunks)

    @pytest.mark.asyncio
    async def test_extractor_results_are_pydantic_models(self, parsed_doc, ai_service):
        """All ExtractionResult objects pass Pydantic V2 validation."""
        from app.schemas.pipeline import ExtractionResult

        chunker = DocumentChunker(ai_service, enrich_chunks=False)
        chunks = await chunker.chunk(parsed_doc)

        extractor = PolicyExtractor(ai_service)
        results = await extractor.extract(chunks)

        for result in results:
            assert isinstance(result, ExtractionResult)

    @pytest.mark.asyncio
    async def test_graph_builder_produces_bundle(self, parsed_doc, ai_service):
        """Step 3: GraphBuilder produces a non-empty GraphBundle."""
        chunker = DocumentChunker(ai_service, enrich_chunks=False)
        chunks = await chunker.chunk(parsed_doc)

        extractor = PolicyExtractor(ai_service)
        results = await extractor.extract(chunks)

        builder = GraphBuilder()
        bundle = builder.build(results, document_id=parsed_doc.document_id)

        assert isinstance(bundle, GraphBundle)
        assert bundle.document_id == parsed_doc.document_id
        assert bundle.node_count > 0
        assert bundle.relationship_count > 0

    @pytest.mark.asyncio
    async def test_graph_has_scheme_node(self, parsed_doc, ai_service):
        """The scheme node must be present in the graph."""
        chunker = DocumentChunker(ai_service, enrich_chunks=False)
        chunks = await chunker.chunk(parsed_doc)

        extractor = PolicyExtractor(ai_service)
        results = await extractor.extract(chunks)

        builder = GraphBuilder()
        bundle = builder.build(results, document_id=parsed_doc.document_id)

        scheme_nodes = [n for n in bundle.nodes if n.node_type == NodeType.SCHEME]
        assert len(scheme_nodes) >= 1

    @pytest.mark.asyncio
    async def test_graph_has_ministry_node(self, parsed_doc, ai_service):
        """issuing_ministry must produce a Ministry node in the graph."""
        chunker = DocumentChunker(ai_service, enrich_chunks=False)
        chunks = await chunker.chunk(parsed_doc)

        extractor = PolicyExtractor(ai_service)
        results = await extractor.extract(chunks)

        builder = GraphBuilder()
        bundle = builder.build(results, document_id=parsed_doc.document_id)

        # Ministry nodes include issuing_ministry AND implementing/supporting orgs
        ministry_nodes = [n for n in bundle.nodes if n.node_type == NodeType.MINISTRY]
        assert len(ministry_nodes) >= 1

    @pytest.mark.asyncio
    async def test_graph_has_administered_by_relationship(self, parsed_doc, ai_service):
        """ISSUED_BY relationship must connect Scheme → issuing Ministry."""
        chunker = DocumentChunker(ai_service, enrich_chunks=False)
        chunks = await chunker.chunk(parsed_doc)

        extractor = PolicyExtractor(ai_service)
        results = await extractor.extract(chunks)

        builder = GraphBuilder()
        bundle = builder.build(results, document_id=parsed_doc.document_id)

        # New topology: ISSUED_BY replaces the generic ADMINISTERED_BY
        issued_by_rels = [
            r for r in bundle.relationships
            if r.rel_type == RelationshipType.ISSUED_BY
        ]
        assert len(issued_by_rels) >= 1

    @pytest.mark.asyncio
    async def test_indexer_produces_vector_documents(self, parsed_doc, ai_service):
        """Step 4: DocumentIndexer produces a VectorDocument per chunk."""
        from app.pipeline.indexer import _EMBEDDING_DIM

        chunker = DocumentChunker(ai_service, enrich_chunks=False)
        chunks = await chunker.chunk(parsed_doc)

        extractor = PolicyExtractor(ai_service)
        results = await extractor.extract(chunks)

        indexer = DocumentIndexer(ai_service)
        vector_docs = await indexer.index(chunks, results)

        assert len(vector_docs) == len(chunks)
        for doc in vector_docs:
            assert isinstance(doc, VectorDocument)
            assert len(doc.vector) == _EMBEDDING_DIM

    @pytest.mark.asyncio
    async def test_vector_docs_have_payloads(self, parsed_doc, ai_service):
        """Every VectorDocument should have a non-empty payload."""
        chunker = DocumentChunker(ai_service, enrich_chunks=False)
        chunks = await chunker.chunk(parsed_doc)

        extractor = PolicyExtractor(ai_service)
        results = await extractor.extract(chunks)

        indexer = DocumentIndexer(ai_service)
        vector_docs = await indexer.index(chunks, results)

        for doc in vector_docs:
            assert len(doc.payload) > 0
            assert doc.payload["chunk_id"] == doc.chunk_id
            assert doc.payload["document_id"] == parsed_doc.document_id

    @pytest.mark.asyncio
    async def test_chunk_ids_preserved_across_pipeline(self, parsed_doc, ai_service):
        """chunk_ids must be consistent from chunker → extractor → indexer."""
        chunker = DocumentChunker(ai_service, enrich_chunks=False)
        chunks = await chunker.chunk(parsed_doc)
        chunk_ids = {c.chunk_id for c in chunks}

        extractor = PolicyExtractor(ai_service)
        results = await extractor.extract(chunks)
        result_ids = {r.chunk_id for r in results}

        indexer = DocumentIndexer(ai_service)
        vector_docs = await indexer.index(chunks, results)
        vector_ids = {d.chunk_id for d in vector_docs}

        assert chunk_ids == result_ids == vector_ids

    @pytest.mark.asyncio
    async def test_pipeline_completes_with_enrichment_enabled(self, parsed_doc, ai_service):
        """Full pipeline with AI enrichment enabled should complete without errors."""
        chunker = DocumentChunker(ai_service, enrich_chunks=True)
        chunks = await chunker.chunk(parsed_doc)

        extractor = PolicyExtractor(ai_service)
        results = await extractor.extract(chunks)

        builder = GraphBuilder()
        bundle = builder.build(results, document_id=parsed_doc.document_id)

        indexer = DocumentIndexer(ai_service)
        vector_docs = await indexer.index(chunks, results)

        # All stages completed successfully
        assert len(chunks) >= 1
        assert len(results) == len(chunks)
        assert bundle.node_count >= 1
        assert len(vector_docs) == len(chunks)

    @pytest.mark.asyncio
    async def test_graph_node_deduplication_across_pipeline(self, parsed_doc, ai_service):
        """
        Multiple chunks may produce the same Ministry or Scheme.
        The GraphBuilder must deduplicate them.
        """
        chunker = DocumentChunker(ai_service, enrich_chunks=False)
        chunks = await chunker.chunk(parsed_doc)

        extractor = PolicyExtractor(ai_service)
        results = await extractor.extract(chunks)

        builder = GraphBuilder()
        bundle = builder.build(results, document_id=parsed_doc.document_id)

        # There should only be one Ministry of Agriculture node despite many chunks
        ministry_nodes = [n for n in bundle.nodes if n.node_type == NodeType.MINISTRY]
        ministry_labels = [n.label.lower() for n in ministry_nodes]
        assert ministry_labels.count("ministry of agriculture & farmers welfare") <= 1

    @pytest.mark.asyncio
    async def test_no_pydantic_validation_errors_across_pipeline(self, parsed_doc, ai_service):
        """
        Complete end-to-end validation: no Pydantic errors at any stage.
        This test acts as a type-safety regression test for the full pipeline.
        """
        from pydantic import ValidationError

        try:
            chunker = DocumentChunker(ai_service, enrich_chunks=False)
            chunks = await chunker.chunk(parsed_doc)

            extractor = PolicyExtractor(ai_service)
            results = await extractor.extract(chunks)

            builder = GraphBuilder()
            bundle = builder.build(results, document_id=parsed_doc.document_id)

            indexer = DocumentIndexer(ai_service)
            vector_docs = await indexer.index(chunks, results)

        except ValidationError as exc:
            pytest.fail(f"Pydantic ValidationError in pipeline: {exc}")

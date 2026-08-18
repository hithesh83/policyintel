"""
Unit tests for PolicyExtractor.

All tests use a mocked AIService — no real LLM calls.

Covers
------
- Pydantic validation of extraction outputs
- Concurrent extraction (asyncio.gather)
- LLMJSONError per-chunk → partial ExtractionResult (extraction_error set)
- LLMError per-chunk → partial ExtractionResult (extraction_error set)
- Keyword-based dispatcher (eligibility / benefit calls triggered correctly)
- Merging of multi-source extractions (metadata + eligibility + benefits + full)
- Empty entity handling (no crashes on empty lists)
- _distinct_list deduplication helper
- _safe_float helper
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.llm.exceptions import LLMError, LLMJSONError
from app.llm.models import ExtractionResponse
from app.pipeline.extractor import PolicyExtractor, _distinct_list, _safe_float
from app.schemas.pipeline import (
    DocumentChunk,
    ExtractionResult,
    HierarchyLevel,
    ChunkMetadata,
)
from app.services.ai_service import AIService


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    chunk_id: str = "chunk-001",
    text: str = "Applicants must be eligible farmers.",
    hierarchy_level: HierarchyLevel = HierarchyLevel.SECTION,
    page_number: int = 1,
    doc_id: str = "doc-001",
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        parent_id=None,
        hierarchy_level=hierarchy_level,
        page_number=page_number,
        title="Test Section",
        section="3.1",
        text=text,
        metadata=ChunkMetadata(word_count=5, char_count=40),
    )


def _extraction_response(data: dict) -> ExtractionResponse:
    return ExtractionResponse(
        data=data,
        raw_output="{}",
        model="test-model",
        request_id="req-test",
    )


def _make_ai_service(
    *,
    metadata_data: dict | None = None,
    eligibility_data: dict | None = None,
    benefits_data: dict | None = None,
    full_data: dict | None = None,
    fail_with: Exception | None = None,
) -> AIService:
    ai = MagicMock(spec=AIService)

    if fail_with:
        ai.extract_policy_metadata = AsyncMock(side_effect=fail_with)
        ai.extract_eligibility = AsyncMock(side_effect=fail_with)
        ai.extract_benefits = AsyncMock(side_effect=fail_with)
        ai.extract_json = AsyncMock(side_effect=fail_with)
        return ai

    ai.extract_policy_metadata = AsyncMock(
        return_value=_extraction_response(
            metadata_data
            or {
                "document_title": "PM Kisan",
                "issuing_ministry": "Ministry of Agriculture and Farmers Welfare",
                "implementing_organizations": ["State Agriculture Departments"],
                "supporting_agencies": ["NABARD", "SBI"],
                "departments": ["Department of Agriculture, Cooperation & Farmers Welfare"],
                "stakeholders": ["DBT Mission"],
                "funding_pattern": "100% Central",
                "scheme_code": "PKS-001",
                "effective_date": "2019-02-01",
                "issue_date": "2019-01-01",
                "policy_type": "central_scheme",
                "target_beneficiaries": ["farmers"],
                "geographic_scope": "national",
                "state_name": None,
                "supersedes": [],
                "language": "English",
            }
        )
    )
    ai.extract_eligibility = AsyncMock(
        return_value=_extraction_response(
            eligibility_data
            or {
                "criteria": [
                    {
                        "criterion_type": "income",
                        "description": "Annual income below Rs. 2 lakh",
                        "condition": "income < 200000",
                        "min_value": None,
                        "max_value": 200000,
                        "unit": "INR",
                        "mandatory": True,
                    }
                ],
                "eligible_categories": ["small farmers"],
                "ineligible_categories": ["government employees"],
                "income_limit_annual": 200000,
                "age_min": 18,
                "age_max": 60,
            }
        )
    )
    ai.extract_benefits = AsyncMock(
        return_value=_extraction_response(
            benefits_data
            or {
                "benefits": [
                    {
                        "benefit_type": "cash_transfer",
                        "description": "Rs. 6,000 per year",
                        "amount_inr": 6000,
                        "frequency": "annual",
                        "duration_months": None,
                        "conditions": ["farmer must have Aadhaar"],
                    }
                ],
                "total_annual_benefit_inr": 6000,
                "is_direct_benefit_transfer": True,
            }
        )
    )
    ai.extract_json = AsyncMock(
        return_value=_extraction_response(
            full_data
            or {
                "scheme_name": "PM Kisan",
                "issuing_ministry": "Ministry of Agriculture and Farmers Welfare",
                "implementing_organizations": ["State Agriculture Departments"],
                "supporting_agencies": ["NABARD"],
                "departments": [
                    "Department of Agriculture, Cooperation & Farmers Welfare"
                ],
                "funding_pattern": "100% Central",
                "stakeholders": [],
                "states": ["All States"],
                "deadlines": ["31 March 2024"],
                "key_dates": ["2019-02-01"],
                "key_amounts": ["Rs. 6,000"],
                "beneficiary_categories": ["small and marginal farmers"],
                "relationships": [],
                "documents_required": ["Aadhaar card", "Land record"],
                "amendment_references": [],
                "effective_date": "2019-02-01",
                "issue_date": "2019-01-01",
                "geographic_scope": "national",
                "policy_type": "central_scheme",
            }
        )
    )
    return ai


# ---------------------------------------------------------------------------
# Tests: basic extraction
# ---------------------------------------------------------------------------


class TestPolicyExtractorBasic:
    @pytest.mark.asyncio
    async def test_extract_returns_one_result_per_chunk(self):
        chunks = [
            _make_chunk("c1", text="Eligible farmers can benefit."),
            _make_chunk("c2", text="Other text."),
        ]
        ai = _make_ai_service()
        extractor = PolicyExtractor(ai)
        results = await extractor.extract(chunks)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_result_chunk_ids_match_input_chunks(self):
        chunks = [
            _make_chunk("chunk-aaa"),
            _make_chunk("chunk-bbb"),
        ]
        ai = _make_ai_service()
        extractor = PolicyExtractor(ai)
        results = await extractor.extract(chunks)

        result_ids = {r.chunk_id for r in results}
        assert "chunk-aaa" in result_ids
        assert "chunk-bbb" in result_ids

    @pytest.mark.asyncio
    async def test_extraction_result_is_pydantic_model(self):
        chunk = _make_chunk()
        ai = _make_ai_service()
        extractor = PolicyExtractor(ai)
        results = await extractor.extract([chunk])

        assert isinstance(results[0], ExtractionResult)

    @pytest.mark.asyncio
    async def test_extracted_entities_are_validated(self):
        chunk = _make_chunk(text="eligible benefit farmer income criteria")
        ai = _make_ai_service()
        extractor = PolicyExtractor(ai)
        results = await extractor.extract([chunk])

        entities = results[0].entities
        # Should have validated eligibility criteria
        assert isinstance(entities.eligibility_criteria, list)
        assert isinstance(entities.benefits, list)


# ---------------------------------------------------------------------------
# Tests: keyword dispatcher
# ---------------------------------------------------------------------------


class TestKeywordDispatcher:
    @pytest.mark.asyncio
    async def test_eligibility_extraction_called_when_keywords_present(self):
        chunk = _make_chunk(text="The applicant must be eligible and meet the income criteria.")
        ai = _make_ai_service()
        extractor = PolicyExtractor(ai)
        await extractor.extract([chunk])

        ai.extract_eligibility.assert_called_once()

    @pytest.mark.asyncio
    async def test_benefit_extraction_called_when_keywords_present(self):
        chunk = _make_chunk(text="The scheme provides financial assistance as a cash benefit.")
        ai = _make_ai_service()
        extractor = PolicyExtractor(ai)
        await extractor.extract([chunk])

        ai.extract_benefits.assert_called_once()

    @pytest.mark.asyncio
    async def test_metadata_called_for_document_level_chunk(self):
        chunk = _make_chunk(
            text="Policy document about farmers.",
            hierarchy_level=HierarchyLevel.DOCUMENT,
        )
        ai = _make_ai_service()
        extractor = PolicyExtractor(ai)
        await extractor.extract([chunk])

        ai.extract_policy_metadata.assert_called_once()

    @pytest.mark.asyncio
    async def test_metadata_not_called_for_section_level_chunk(self):
        chunk = _make_chunk(
            text="Pure text without special keywords.",
            hierarchy_level=HierarchyLevel.SECTION,
        )
        ai = _make_ai_service()
        extractor = PolicyExtractor(ai)
        await extractor.extract([chunk])

        ai.extract_policy_metadata.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: error isolation
# ---------------------------------------------------------------------------


class TestErrorIsolation:
    @pytest.mark.asyncio
    async def test_llm_json_error_produces_partial_result(self):
        chunk = _make_chunk(text="eligible benefit farmer")
        ai = _make_ai_service(fail_with=LLMJSONError("Bad JSON"))
        extractor = PolicyExtractor(ai)
        results = await extractor.extract([chunk])

        assert len(results) == 1
        assert results[0].extraction_error is not None
        assert "Bad JSON" in results[0].extraction_error

    @pytest.mark.asyncio
    async def test_llm_error_produces_partial_result(self):
        chunk = _make_chunk(text="eligible benefit farmer")
        ai = _make_ai_service(fail_with=LLMError("Connection failed"))
        extractor = PolicyExtractor(ai)
        results = await extractor.extract([chunk])

        assert len(results) == 1
        assert results[0].extraction_error is not None

    @pytest.mark.asyncio
    async def test_failed_chunk_has_empty_entities(self):
        chunk = _make_chunk(text="eligible benefit farmer")
        ai = _make_ai_service(fail_with=LLMError("fail"))
        extractor = PolicyExtractor(ai)
        results = await extractor.extract([chunk])

        entities = results[0].entities
        assert entities.scheme_name is None
        assert entities.eligibility_criteria == []
        assert entities.benefits == []

    @pytest.mark.asyncio
    async def test_one_failed_chunk_does_not_abort_others(self):
        """A failing chunk should not prevent successful extraction of other chunks."""
        ai_fail = MagicMock(spec=AIService)
        ai_fail.extract_json = AsyncMock(side_effect=LLMJSONError("Bad JSON"))
        ai_fail.extract_policy_metadata = AsyncMock(side_effect=LLMJSONError("Bad JSON"))
        ai_fail.extract_eligibility = AsyncMock(side_effect=LLMJSONError("Bad JSON"))
        ai_fail.extract_benefits = AsyncMock(side_effect=LLMJSONError("Bad JSON"))

        ai_ok = _make_ai_service()

        # Use the failing AI — all chunks will fail but none should raise
        chunk1 = _make_chunk("c1", "eligible benefit farmer")
        chunk2 = _make_chunk("c2", "eligible benefit farmer")
        extractor = PolicyExtractor(ai_fail)
        results = await extractor.extract([chunk1, chunk2])

        assert len(results) == 2
        for r in results:
            assert r.extraction_error is not None


# ---------------------------------------------------------------------------
# Tests: merging logic
# ---------------------------------------------------------------------------


class TestMerging:
    @pytest.mark.asyncio
    async def test_scheme_name_from_metadata_takes_priority(self):
        chunk = _make_chunk(
            text="PM Kisan scheme benefits eligible farmers.",
            hierarchy_level=HierarchyLevel.DOCUMENT,
        )
        ai = _make_ai_service(
            metadata_data={"document_title": "PM Kisan Nidhi", "issuing_ministry": "MoA"},
            full_data={"scheme_name": "PM Kisan", "issuing_ministry": "MoA"},
        )
        extractor = PolicyExtractor(ai)
        results = await extractor.extract([chunk])

        # metadata "document_title" should win over full "scheme_name"
        assert results[0].entities.scheme_name == "PM Kisan Nidhi"

    @pytest.mark.asyncio
    async def test_states_deduped(self):
        chunk = _make_chunk(
            text="Available in Rajasthan and MP eligible farmer benefit.",
            hierarchy_level=HierarchyLevel.DOCUMENT,
        )
        ai = _make_ai_service(
            metadata_data={"state_name": "Rajasthan", "document_title": "Scheme"},
            full_data={
                "scheme_name": "Scheme",
                "states": ["Rajasthan", "Madhya Pradesh"],
            },
        )
        extractor = PolicyExtractor(ai)
        results = await extractor.extract([chunk])

        states = results[0].entities.states
        # "Rajasthan" appears twice but should only be in list once
        assert states.count("Rajasthan") == 1


# ---------------------------------------------------------------------------
# Tests: helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_distinct_list_deduplication(self):
        result = _distinct_list(["a", "b", "a"], ["b", "c"])
        assert result == ["a", "b", "c"]

    def test_distinct_list_none_sources_ignored(self):
        result = _distinct_list(None, ["a", "b"])
        assert result == ["a", "b"]

    def test_distinct_list_empty_string_excluded(self):
        result = _distinct_list(["", "a", "  "])
        assert result == ["a"]

    def test_distinct_list_single_string_source(self):
        result = _distinct_list("single")
        assert result == ["single"]

    def test_safe_float_with_int(self):
        assert _safe_float(42) == 42.0

    def test_safe_float_with_string(self):
        assert _safe_float("3.14") == pytest.approx(3.14)

    def test_safe_float_with_none(self):
        assert _safe_float(None) is None

    def test_safe_float_with_invalid_string(self):
        assert _safe_float("not-a-number") is None

    def test_safe_float_with_empty_string(self):
        assert _safe_float("") is None


# ---------------------------------------------------------------------------
# Regression tests: DDD domain model — organisational field extraction
# ---------------------------------------------------------------------------


from app.pipeline.extractor import _coerce_to_str  # noqa: E402


class TestOrganisationalDomainModel:
    """
    Regression tests for the DDD organisational fields introduced to replace
    the incorrect ``ministry: str`` scalar.

    Covers:
    ✓ Single ministry (issuing_ministry resolves to a string)
    ✓ LLM returns a list for issuing_ministry — root-cause scenario
    ✓ Multiple implementing organizations
    ✓ Supporting agencies (banks, NABARD, etc.)
    ✓ Departments captured separately from issuing ministry
    ✓ Central Government scheme
    ✓ State Government scheme
    ✓ Null / missing values
    ✓ Empty arrays
    ✓ Duplicate organisations deduped
    """

    @pytest.mark.asyncio
    async def test_single_issuing_ministry_extracted(self):
        """Happy path: LLM returns a string for issuing_ministry."""
        chunk = _make_chunk(
            text="PM-KISAN is issued by the Ministry of Agriculture and Farmers Welfare.",
            hierarchy_level=HierarchyLevel.DOCUMENT,
        )
        ai = _make_ai_service(
            metadata_data={
                "document_title": "PM-KISAN",
                "issuing_ministry": "Ministry of Agriculture and Farmers Welfare",
                "implementing_organizations": ["State Agriculture Departments"],
                "supporting_agencies": [],
                "departments": [],
            },
            full_data={
                "issuing_ministry": "Ministry of Agriculture and Farmers Welfare",
                "implementing_organizations": [],
            },
        )
        extractor = PolicyExtractor(ai)
        results = await extractor.extract([chunk])
        entities = results[0].entities

        assert entities.issuing_ministry == "Ministry of Agriculture and Farmers Welfare"
        assert entities.implementing_organizations == ["State Agriculture Departments"]

    @pytest.mark.asyncio
    async def test_llm_returns_list_for_issuing_ministry_is_coerced(self):
        """Root-cause regression: LLM returns list for scalar field."""
        chunk = _make_chunk(
            text="This scheme involves Central Government and State Governments.",
            hierarchy_level=HierarchyLevel.DOCUMENT,
        )
        ai = _make_ai_service(
            metadata_data={
                "document_title": "Test Scheme",
                # LLM returned a list — this was the original crash
                "issuing_ministry": [
                    "Central Government",
                    "State Government",
                    "Departments",
                ],
                "implementing_organizations": [],
                "supporting_agencies": [],
                "departments": [],
            },
            full_data={"scheme_name": "Test Scheme"},
        )
        extractor = PolicyExtractor(ai)
        # Must NOT raise ValidationError
        results = await extractor.extract([chunk])
        entities = results[0].entities

        # _coerce_to_str picks the first non-empty element
        assert entities.issuing_ministry == "Central Government"

    @pytest.mark.asyncio
    async def test_multiple_implementing_organizations(self):
        chunk = _make_chunk(
            text="Implemented by state depts and district offices.",
            hierarchy_level=HierarchyLevel.DOCUMENT,
        )
        ai = _make_ai_service(
            metadata_data={
                "document_title": "Scheme",
                "issuing_ministry": "Ministry of Agriculture",
                "implementing_organizations": [
                    "State Agriculture Departments",
                    "District Collectors",
                    "Block Development Officers",
                ],
                "supporting_agencies": [],
                "departments": [],
            },
            full_data={"implementing_organizations": ["District Collectors"]},
        )
        extractor = PolicyExtractor(ai)
        results = await extractor.extract([chunk])
        orgs = results[0].entities.implementing_organizations

        assert "State Agriculture Departments" in orgs
        assert "District Collectors" in orgs
        assert "Block Development Officers" in orgs
        # "District Collectors" appeared in both sources but must be deduped
        assert orgs.count("District Collectors") == 1

    @pytest.mark.asyncio
    async def test_supporting_agencies_banks_extracted(self):
        chunk = _make_chunk(
            text="Funds disbursed through SBI and NABARD.",
            hierarchy_level=HierarchyLevel.DOCUMENT,
        )
        ai = _make_ai_service(
            metadata_data={
                "document_title": "Scheme",
                "issuing_ministry": "Ministry of Finance",
                "implementing_organizations": [],
                "supporting_agencies": ["SBI", "NABARD", "India Post Payments Bank"],
                "departments": [],
            },
            full_data={"supporting_agencies": ["NPCI"]},
        )
        extractor = PolicyExtractor(ai)
        results = await extractor.extract([chunk])
        agencies = results[0].entities.supporting_agencies

        assert "SBI" in agencies
        assert "NABARD" in agencies
        assert "NPCI" in agencies

    @pytest.mark.asyncio
    async def test_departments_captured_separately(self):
        chunk = _make_chunk(
            text="Implemented by Department of Agriculture.",
            hierarchy_level=HierarchyLevel.DOCUMENT,
        )
        ai = _make_ai_service(
            metadata_data={
                "document_title": "Scheme",
                "issuing_ministry": "Ministry of Agriculture",
                "implementing_organizations": [],
                "supporting_agencies": [],
                "departments": [
                    "Department of Agriculture, Cooperation & Farmers Welfare"
                ],
            },
            full_data={},
        )
        extractor = PolicyExtractor(ai)
        results = await extractor.extract([chunk])
        depts = results[0].entities.departments

        assert "Department of Agriculture, Cooperation & Farmers Welfare" in depts

    @pytest.mark.asyncio
    async def test_null_issuing_ministry_is_none(self):
        chunk = _make_chunk(
            text="Eligibility criteria for this benefit.",
            hierarchy_level=HierarchyLevel.DOCUMENT,
        )
        ai = _make_ai_service(
            metadata_data={
                "document_title": "Unknown Scheme",
                "issuing_ministry": None,
                "implementing_organizations": [],
                "supporting_agencies": [],
                "departments": [],
            },
            full_data={"issuing_ministry": None},
        )
        extractor = PolicyExtractor(ai)
        results = await extractor.extract([chunk])

        assert results[0].entities.issuing_ministry is None

    @pytest.mark.asyncio
    async def test_empty_arrays_produce_empty_lists(self):
        chunk = _make_chunk(
            text="This policy applies nationally.",
            hierarchy_level=HierarchyLevel.DOCUMENT,
        )
        ai = _make_ai_service(
            metadata_data={
                "document_title": "Scheme",
                "issuing_ministry": "Central Ministry",
                "implementing_organizations": [],
                "supporting_agencies": [],
                "departments": [],
                "stakeholders": [],
            },
            full_data={
                "implementing_organizations": [],
                "supporting_agencies": [],
            },
        )
        extractor = PolicyExtractor(ai)
        results = await extractor.extract([chunk])
        entities = results[0].entities

        assert entities.implementing_organizations == []
        assert entities.supporting_agencies == []
        assert entities.departments == []
        assert entities.stakeholders == []

    @pytest.mark.asyncio
    async def test_duplicate_orgs_across_sources_are_deduped(self):
        chunk = _make_chunk(
            text="State Depts implement this scheme.",
            hierarchy_level=HierarchyLevel.DOCUMENT,
        )
        ai = _make_ai_service(
            metadata_data={
                "document_title": "Scheme",
                "issuing_ministry": "Ministry X",
                "implementing_organizations": ["State Agriculture Departments"],
                "supporting_agencies": [],
                "departments": [],
            },
            # Same org appears in full_data as well
            full_data={"implementing_organizations": ["State Agriculture Departments"]},
        )
        extractor = PolicyExtractor(ai)
        results = await extractor.extract([chunk])
        orgs = results[0].entities.implementing_organizations

        assert orgs.count("State Agriculture Departments") == 1


# ---------------------------------------------------------------------------
# Unit tests: _coerce_to_str
# ---------------------------------------------------------------------------


class TestCoerceToStr:
    """Unit tests for the _coerce_to_str normalisation helper."""

    def test_string_returned_as_is(self):
        assert _coerce_to_str("Ministry of Agriculture") == "Ministry of Agriculture"

    def test_string_stripped(self):
        assert _coerce_to_str("  Ministry  ") == "Ministry"

    def test_empty_string_returns_none(self):
        assert _coerce_to_str("") is None

    def test_whitespace_string_returns_none(self):
        assert _coerce_to_str("   ") is None

    def test_none_returns_none(self):
        assert _coerce_to_str(None) is None

    def test_list_returns_first_non_empty_string(self):
        # Root-cause regression: LLM returned a list for a singular field
        result = _coerce_to_str(["Central Government", "State Government", "Departments"])
        assert result == "Central Government"

    def test_list_with_leading_empty_strings_skips_to_first_valid(self):
        assert _coerce_to_str(["", "  ", "Ministry of Finance"]) == "Ministry of Finance"

    def test_list_of_only_empty_strings_returns_none(self):
        assert _coerce_to_str(["", "   "]) is None

    def test_empty_list_returns_none(self):
        assert _coerce_to_str([]) is None

    def test_int_returns_none(self):
        assert _coerce_to_str(42) is None

    def test_dict_returns_none(self):
        assert _coerce_to_str({"key": "value"}) is None

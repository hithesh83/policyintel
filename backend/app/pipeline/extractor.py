"""
Policy Extractor
================

``PolicyExtractor`` processes a list of ``DocumentChunk`` objects and calls
``AIService`` to extract structured policy intelligence from each chunk.

Extraction Strategy
-------------------
For each chunk, the extractor dispatches one or more specialised
``AIService`` calls depending on the chunk's hierarchy level and content:

1. **Document / Chapter chunks** → ``AIService.extract_policy_metadata()``
   to capture top-level scheme metadata (name, ministry, dates, scope).

2. **Any chunk** → ``AIService.extract_eligibility()`` when the chunk is
   heuristically likely to contain eligibility criteria (detected via
   keywords like "eligible", "criteria", "beneficiary").

3. **Any chunk** → ``AIService.extract_benefits()`` when the chunk is
   heuristically likely to contain benefit information (keywords like
   "benefit", "amount", "assistance", "subsidy").

4. **All chunks** → ``AIService.extract_json()`` with the full comprehensive
   entity schema to pick up any remaining entities not covered above.

Concurrency
-----------
All chunk extractions run concurrently via ``asyncio.gather`` with a semaphore
bounding the number of simultaneous LLM calls (default: 3).

Each extraction call within a chunk also runs concurrently (metadata +
eligibility + benefits + entities are all gathered at once per chunk).

Error Isolation
---------------
``LLMJSONError`` and ``LLMError`` per-chunk are caught and logged.  The chunk
produces an ``ExtractionResult`` with ``extraction_error`` set and an empty
``ExtractedEntities`` rather than propagating the exception.  This ensures
one malformed LLM response never aborts the entire document ingestion.

Usage
-----
::

    from app.pipeline.extractor import PolicyExtractor

    extractor = PolicyExtractor(ai_service=ai_service)
    results = await extractor.extract(chunks)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from app.llm.exceptions import LLMError, LLMJSONError
from app.schemas.pipeline import (
    Benefit,
    DocumentChunk,
    EligibilityCriterion,
    ExtractionResult,
    ExtractedEntities,
    HierarchyLevel,
)
from app.services.ai_service import AIService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword heuristics for dispatcher
# ---------------------------------------------------------------------------

_ELIGIBILITY_KEYWORDS = frozenset(
    {
        "eligible", "eligibility", "criteria", "criterion", "beneficiary",
        "beneficiaries", "qualify", "qualification", "entitled", "entitlement",
        "income limit", "age limit", "resident", "citizen", "household",
    }
)

_BENEFIT_KEYWORDS = frozenset(
    {
        "benefit", "benefits", "subsidy", "grant", "allowance", "pension",
        "assistance", "amount", "financial", "cash", "transfer", "loan",
        "insurance", "entitlement", "rupee", "inr", "lakh", "crore",
    }
)

_DOCUMENT_LEVELS = frozenset({HierarchyLevel.DOCUMENT, HierarchyLevel.CHAPTER})


def _contains_keywords(text: str, keywords: frozenset[str]) -> bool:
    """Return True if the lower-cased text contains any of the keywords."""
    lower = text.lower()
    return any(kw in lower for kw in keywords)


# ---------------------------------------------------------------------------
# PolicyExtractor
# ---------------------------------------------------------------------------


class PolicyExtractor:
    """
    Extracts structured policy entities from ``DocumentChunk`` objects using
    ``AIService``.

    Parameters
    ----------
    ai_service :
        The application-level AI abstraction.  Never calls the LLM directly.
    max_concurrent_chunks :
        Maximum number of chunks processed concurrently.  Each chunk may itself
        issue up to 4 concurrent LLM calls.
    """

    def __init__(
        self,
        ai_service: AIService,
        *,
        max_concurrent_chunks: int = 3,
    ) -> None:
        self._ai = ai_service
        self._semaphore = asyncio.Semaphore(max_concurrent_chunks)
        logger.info(
            "PolicyExtractor initialised | max_concurrent_chunks=%d",
            max_concurrent_chunks,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def extract(self, chunks: list[DocumentChunk]) -> list[ExtractionResult]:
        """
        Extract structured entities from all chunks concurrently.

        Parameters
        ----------
        chunks :
            Chunks produced by ``DocumentChunker.chunk()``.

        Returns
        -------
        list[ExtractionResult]
            One result per input chunk, in the same order.
            Results with ``extraction_error`` set indicate partial failures.
        """
        logger.info(
            "PolicyExtractor.extract() START | chunks=%d",
            len(chunks),
        )
        start = time.monotonic()

        results = await asyncio.gather(
            *[self._extract_chunk(chunk) for chunk in chunks],
        )

        elapsed_ms = int((time.monotonic() - start) * 1000)
        errors = sum(1 for r in results if r.extraction_error)
        logger.info(
            "PolicyExtractor.extract() DONE | chunks=%d | errors=%d | latency_ms=%d",
            len(results),
            errors,
            elapsed_ms,
        )
        return list(results)

    # ------------------------------------------------------------------
    # Per-Chunk Extraction
    # ------------------------------------------------------------------

    async def _extract_chunk(self, chunk: DocumentChunk) -> ExtractionResult:
        """
        Extract all entity types from a single chunk concurrently.

        Catches all LLM errors and returns a partial result rather than
        raising.
        """
        async with self._semaphore:
            start = time.monotonic()
            logger.debug(
                "PolicyExtractor._extract_chunk() | chunk_id=%s | level=%s",
                chunk.chunk_id,
                chunk.hierarchy_level,
            )
            try:
                entities = await self._gather_entities(chunk)
                return ExtractionResult(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    hierarchy_level=chunk.hierarchy_level,
                    page_number=chunk.page_number,
                    section=chunk.section,
                    title=chunk.title,
                    entities=entities,
                    raw_text=chunk.text,
                    extraction_error=None,
                    model_used="unknown",
                    latency_ms=int((time.monotonic() - start) * 1000),
                )
            except (LLMJSONError, LLMError) as exc:
                logger.warning(
                    "PolicyExtractor: extraction failed | chunk_id=%s | error=%s",
                    chunk.chunk_id,
                    exc,
                )
                return ExtractionResult(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    hierarchy_level=chunk.hierarchy_level,
                    page_number=chunk.page_number,
                    section=chunk.section,
                    title=chunk.title,
                    entities=ExtractedEntities(),
                    raw_text=chunk.text,
                    extraction_error=str(exc),
                    model_used="unknown",
                    latency_ms=int((time.monotonic() - start) * 1000),
                )

    async def _gather_entities(self, chunk: DocumentChunk) -> ExtractedEntities:
        """
        Dispatch and merge all relevant extraction calls for a single chunk.

        Runs applicable extractions concurrently, then merges the results
        into a single ``ExtractedEntities`` instance.
        """
        text = chunk.text

        # Build the list of coroutines to run for this chunk
        coros = []
        labels = []

        # Always: full entity extraction
        coros.append(self._extract_full_entities(text))
        labels.append("full")

        # Conditional: metadata extraction for top-level chunks
        if chunk.hierarchy_level in _DOCUMENT_LEVELS:
            coros.append(self._extract_metadata(text))
            labels.append("metadata")
        else:
            coros.append(self._noop())
            labels.append("metadata_skip")

        # Conditional: eligibility extraction
        if _contains_keywords(text, _ELIGIBILITY_KEYWORDS):
            coros.append(self._extract_eligibility(text))
            labels.append("eligibility")
        else:
            coros.append(self._noop())
            labels.append("eligibility_skip")

        # Conditional: benefit extraction
        if _contains_keywords(text, _BENEFIT_KEYWORDS):
            coros.append(self._extract_benefits(text))
            labels.append("benefits")
        else:
            coros.append(self._noop())
            labels.append("benefits_skip")

        raw_results = await asyncio.gather(*coros, return_exceptions=True)

        # If ALL non-noop calls raised, propagate the first real exception so
        # that _extract_chunk can set extraction_error correctly.
        real_errors = [
            r for r, lbl in zip(raw_results, labels)
            if isinstance(r, Exception) and not lbl.endswith("_skip")
        ]
        non_error_results = [
            r for r, lbl in zip(raw_results, labels)
            if not isinstance(r, Exception) and not lbl.endswith("_skip")
        ]
        if real_errors and not non_error_results:
            raise real_errors[0]

        return self._merge_extractions(
            full=self._safe_get(raw_results, labels, "full"),
            metadata=self._safe_get(raw_results, labels, "metadata"),
            eligibility=self._safe_get(raw_results, labels, "eligibility"),
            benefits=self._safe_get(raw_results, labels, "benefits"),
            chunk=chunk,
        )

    # ------------------------------------------------------------------
    # Specialised Extraction Calls
    # ------------------------------------------------------------------

    async def _extract_metadata(self, text: str) -> dict[str, Any]:
        """Call AIService.extract_policy_metadata() and return raw data dict."""
        response = await self._ai.extract_policy_metadata(document_text=text)
        return response.data

    async def _extract_eligibility(self, text: str) -> dict[str, Any]:
        """Call AIService.extract_eligibility() and return raw data dict."""
        response = await self._ai.extract_eligibility(section_text=text)
        return response.data

    async def _extract_benefits(self, text: str) -> dict[str, Any]:
        """Call AIService.extract_benefits() and return raw data dict."""
        response = await self._ai.extract_benefits(section_text=text)
        return response.data

    async def _extract_full_entities(self, text: str) -> dict[str, Any]:
        """
        Call AIService.extract_json() with the comprehensive entity schema.

        Extracts dates, amounts, states, beneficiary categories, relationships,
        required documents, deadlines, and amendment references.  Also extracts
        the full organisational graph (issuing_ministry, implementing_organizations,
        supporting_agencies, departments) using the Government Policy Domain Model.
        """
        from app.llm.prompts.extraction import build_json_extraction_prompt

        prompt = build_json_extraction_prompt(
            base_prompt=f"""Extract all structured policy entities from the following \
government policy text.

POLICY TEXT:
{text[:3000]}

Extract every entity you can identify:\
""",
            schema_hint={
                "scheme_name": "string or null",
                # ── Organisational fields: read comments carefully ──
                "issuing_ministry": (
                    "string — the SINGLE ministry that ISSUED this policy. "
                    "MUST be a string, NEVER an array. null if unknown."
                ),
                "implementing_organizations": (
                    "array of strings — bodies that DELIVER the scheme operationally "
                    "(e.g. State Agriculture Departments, District Collectors). "
                    "Empty array [] if none."
                ),
                "supporting_agencies": (
                    "array of strings — banks, NBFCs, insurance companies, technology "
                    "providers that SUPPORT but don't issue (e.g. NABARD, SBI, NPCI). "
                    "Empty array [] if none."
                ),
                "departments": (
                    "array of strings — sub-ministerial departments named explicitly. "
                    "e.g. ['Department of Agriculture, Cooperation & Farmers Welfare']. "
                    "Empty array [] if none."
                ),
                "funding_pattern": (
                    "string — cost-sharing ratio e.g. '60:40 Centre:State'. null if not mentioned."
                ),
                "stakeholders": (
                    "array of strings — other stakeholders (nodal officers, gram panchayats, "
                    "DBT Mission, etc.). Empty array [] if none."
                ),
                # ── Remaining entity fields ──
                "states": ["list of state names mentioned"],
                "deadlines": ["list of deadline dates or periods"],
                "key_dates": ["list of all dates mentioned"],
                "key_amounts": ["list of all monetary amounts"],
                "beneficiary_categories": ["list of beneficiary groups"],
                "relationships": ["list of entity relationships described"],
                "documents_required": ["list of documents required to apply"],
                "amendment_references": ["list of amendment/circular/notification references"],
                "effective_date": "YYYY-MM-DD or null",
                "issue_date": "YYYY-MM-DD or null",
                "geographic_scope": "national | state | district | local | null",
                "policy_type": (
                    "central_scheme | state_scheme | regulation | "
                    "notification | circular | guideline | null"
                ),
            },
        )
        response = await self._ai.extract_json(prompt, temperature=0.0, seed=42)
        return response.data

    @staticmethod
    async def _noop() -> dict[str, Any]:
        """Return an empty dict — placeholder for skipped extraction steps."""
        return {}

    # ------------------------------------------------------------------
    # Merging & Validation
    # ------------------------------------------------------------------

    def _merge_extractions(
        self,
        *,
        full: dict[str, Any],
        metadata: dict[str, Any],
        eligibility: dict[str, Any],
        benefits: dict[str, Any],
        chunk: DocumentChunk,
    ) -> ExtractedEntities:
        """
        Merge extraction results from multiple calls into one
        ``ExtractedEntities`` Pydantic model.

        Priority order (higher priority wins on scalar conflicts):
        metadata > full > eligibility/benefits
        """
        # ---- scalar identity fields (metadata wins over full) ----
        scheme_name = (
            metadata.get("document_title")
            or full.get("scheme_name")
        )
        scheme_code = metadata.get("scheme_code")
        effective_date = (
            metadata.get("effective_date") or full.get("effective_date")
        )
        issue_date = metadata.get("issue_date") or full.get("issue_date")
        policy_type = metadata.get("policy_type") or full.get("policy_type")
        geographic_scope = (
            metadata.get("geographic_scope") or full.get("geographic_scope")
        )

        # ---- organisational fields (DDD domain model) ----
        # issuing_ministry: singular — resolve with metadata priority.
        # Guard against LLM returning a list instead of a string.
        raw_issuing_meta = metadata.get("issuing_ministry")
        raw_issuing_full = full.get("issuing_ministry")
        issuing_ministry = _coerce_to_str(raw_issuing_meta) or _coerce_to_str(raw_issuing_full)

        # Plural organisational lists — merge across metadata + full.
        implementing_organizations = _distinct_list(
            metadata.get("implementing_organizations"),
            full.get("implementing_organizations"),
        )
        supporting_agencies = _distinct_list(
            metadata.get("supporting_agencies"),
            full.get("supporting_agencies"),
        )
        departments = _distinct_list(
            metadata.get("departments"),
            full.get("departments"),
        )
        stakeholders = _distinct_list(
            metadata.get("stakeholders"),
            full.get("stakeholders"),
        )
        funding_pattern = metadata.get("funding_pattern") or full.get("funding_pattern")
        if isinstance(funding_pattern, list):
            # Guard: LLM occasionally returns a list for this scalar field
            funding_pattern = funding_pattern[0] if funding_pattern else None

        # ---- list fields (union across all sources) ----
        states = _distinct_list(
            metadata.get("state_name") and [metadata["state_name"]],
            full.get("states"),
        )
        supersedes = _distinct_list(metadata.get("supersedes"), full.get("supersedes"))
        deadlines = _distinct_list(full.get("deadlines"))
        key_dates = _distinct_list(full.get("key_dates"))
        key_amounts = _distinct_list(full.get("key_amounts"))
        beneficiary_categories = _distinct_list(
            metadata.get("target_beneficiaries"),
            full.get("beneficiary_categories"),
            eligibility.get("eligible_categories"),
        )
        documents_required = _distinct_list(full.get("documents_required"))
        amendment_references = _distinct_list(full.get("amendment_references"))
        relationships = _distinct_list(full.get("relationships"))

        # ---- eligibility criteria (validated via Pydantic) ----
        eligibility_criteria = self._parse_eligibility_criteria(
            eligibility.get("criteria") or []
        )
        eligible_categories = _distinct_list(
            eligibility.get("eligible_categories"),
            full.get("beneficiary_categories"),
        )
        ineligible_categories = _distinct_list(eligibility.get("ineligible_categories"))
        income_limit_annual = _safe_float(eligibility.get("income_limit_annual"))
        age_min = _safe_float(eligibility.get("age_min"))
        age_max = _safe_float(eligibility.get("age_max"))

        # ---- benefits (validated via Pydantic) ----
        benefit_objects = self._parse_benefits(benefits.get("benefits") or [])
        total_annual_benefit_inr = _safe_float(benefits.get("total_annual_benefit_inr"))
        is_direct_benefit_transfer: bool | None = benefits.get("is_direct_benefit_transfer")
        if isinstance(is_direct_benefit_transfer, str):
            is_direct_benefit_transfer = is_direct_benefit_transfer.lower() in ("true", "yes", "1")

        return ExtractedEntities(
            scheme_name=scheme_name,
            issuing_ministry=issuing_ministry,
            implementing_organizations=implementing_organizations,
            supporting_agencies=supporting_agencies,
            departments=departments,
            stakeholders=stakeholders,
            funding_pattern=funding_pattern or None,
            scheme_code=scheme_code,
            effective_date=effective_date,
            issue_date=issue_date,
            policy_type=policy_type,
            geographic_scope=geographic_scope,
            states=states,
            supersedes=supersedes,
            eligibility_criteria=eligibility_criteria,
            eligible_categories=eligible_categories,
            ineligible_categories=ineligible_categories,
            income_limit_annual=income_limit_annual,
            age_min=age_min,
            age_max=age_max,
            beneficiary_categories=beneficiary_categories,
            benefits=benefit_objects,
            total_annual_benefit_inr=total_annual_benefit_inr,
            is_direct_benefit_transfer=is_direct_benefit_transfer,
            deadlines=deadlines,
            key_dates=key_dates,
            key_amounts=key_amounts,
            relationships=relationships,
            documents_required=documents_required,
            amendment_references=amendment_references,
        )

    @staticmethod
    def _parse_eligibility_criteria(raw: list[Any]) -> list[EligibilityCriterion]:
        """
        Validate raw eligibility criterion dicts through the Pydantic model.

        Invalid items are skipped with a warning.
        """
        result: list[EligibilityCriterion] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                result.append(EligibilityCriterion(**item))
            except Exception as exc:
                logger.debug("Skipping invalid eligibility criterion: %s | %s", item, exc)
        return result

    @staticmethod
    def _parse_benefits(raw: list[Any]) -> list[Benefit]:
        """
        Validate raw benefit dicts through the Pydantic model.

        Invalid items are skipped with a warning.
        """
        result: list[Benefit] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                result.append(Benefit(**item))
            except Exception as exc:
                logger.debug("Skipping invalid benefit: %s | %s", item, exc)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_get(
        results: tuple[Any, ...] | list[Any],
        labels: list[str],
        label: str,
    ) -> dict[str, Any]:
        """
        Retrieve the result at the index of *label* from *labels*.

        Returns an empty dict if the result was an exception or the label
        does not map to a dict.
        """
        try:
            idx = labels.index(label)
        except ValueError:
            return {}
        value = results[idx]
        if isinstance(value, Exception):
            logger.debug("Extraction '%s' failed: %s", label, value)
            return {}
        if isinstance(value, dict):
            return value
        return {}


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _distinct_list(*sources: Any) -> list[str]:
    """
    Merge multiple source lists into a deduplicated list of non-empty strings.

    Sources can be None, a list, or a single string.
    """
    seen: set[str] = set()
    result: list[str] = []
    for src in sources:
        if src is None:
            continue
        if isinstance(src, str):
            items = [src]
        elif isinstance(src, list):
            items = src
        else:
            continue
        for item in items:
            if isinstance(item, str) and item.strip() and item not in seen:
                seen.add(item)
                result.append(item)
    return result


def _safe_float(value: Any) -> float | None:
    """Safely coerce a value to float, returning None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_to_str(value: Any) -> str | None:
    """
    Safely coerce an AI-returned value to ``str | None``.

    Government LLMs sometimes return a list for a field declared as a string
    (e.g. ``"issuing_ministry": ["Central Government", "State Government"]``).
    This helper normalises that gracefully:

    * ``str``  → stripped value, or ``None`` if empty/whitespace.
    * ``list`` → the first non-empty string element, or ``None``.
    * ``None`` → ``None``.
    * anything else → ``None``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    return stripped
        return None
    return None


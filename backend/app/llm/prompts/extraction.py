"""
Extraction Prompts
==================

Prompts for extracting structured data from unstructured policy documents.

These prompts are used by:
  - AIService.extract_json()               (Week 1)
  - Week 2: pipeline/extractor.py          (PDF content extraction)
  - Week 2: pipeline/chunker.py            (chunk metadata generation)

JSON Instruction Patterns
--------------------------
Every extraction prompt MUST end with an explicit instruction to return
valid JSON with no additional text.  This is the most effective way to
get reliable JSON output from instruction-tuned models.

Schema Hints
------------
Callers can pass ``schema_hint`` to ``AIService.extract_json()`` which
is then serialised into the prompt.  This significantly improves extraction
accuracy for complex nested schemas.
"""

from __future__ import annotations

import json
from typing import Any


def build_json_extraction_prompt(
    base_prompt: str,
    schema_hint: dict[str, Any] | None = None,
) -> str:
    """
    Append JSON output instructions to a base extraction prompt.

    This is called by ``OllamaClient.extract_json()`` to ensure the model
    always returns a JSON object regardless of how the caller phrased the
    base prompt.

    Parameters
    ----------
    base_prompt :
        The task description / extraction instructions.
    schema_hint :
        Optional dict describing the expected JSON schema.
        When provided, it is serialised and embedded in the prompt.

    Returns
    -------
    str
        The base prompt with JSON output instructions appended.

    Example
    -------
    >>> prompt = build_json_extraction_prompt(
    ...     "Extract the scheme name and eligibility criteria from this text:\n...",
    ...     schema_hint={"scheme_name": "str", "criteria": ["str"]}
    ... )
    """
    schema_section = ""
    if schema_hint:
        schema_section = f"""
Expected JSON schema:
{json.dumps(schema_hint, indent=2, ensure_ascii=False)}

"""

    return f"""{base_prompt}
{schema_section}
CRITICAL INSTRUCTIONS:
- Return ONLY a valid JSON object.
- Do NOT include markdown code fences (``` or ```json).
- Do NOT include any explanation, prose, or commentary.
- Do NOT include trailing commas.
- Use null for missing values, not None or "N/A".
- All string values must use double quotes.

JSON OUTPUT:"""


def build_policy_metadata_extraction_prompt(document_text: str) -> str:
    """
    Extract top-level metadata from a full policy document.

    Used at the beginning of the ingestion pipeline to identify what
    policy document has been received before chunking.

    Parameters
    ----------
    document_text :
        First 3000 characters of the policy document (header + first few sections).

    Returns
    -------
    str
        Extraction prompt for document-level metadata.
    """
    return build_json_extraction_prompt(
        base_prompt=f"""Extract metadata from the following government policy document header.

DOCUMENT TEXT (first section):
{document_text[:3000]}

Extract the following information:""",
        schema_hint={
            "document_title": "string — official name of the policy/scheme",
            "ministry": "string — issuing ministry or department, or null",
            "scheme_code": "string — official scheme code/number, or null",
            "effective_date": "string — effective date in YYYY-MM-DD format, or null",
            "issue_date": "string — date of issue in YYYY-MM-DD format, or null",
            "policy_type": "enum: central_scheme | state_scheme | regulation | notification | circular | guideline",
            "target_beneficiaries": "array of strings — who this policy targets",
            "geographic_scope": "enum: national | state | district | local | null",
            "state_name": "string — state name if state-specific, or null",
            "supersedes": "array of strings — documents this supersedes, or empty array",
            "language": "string — document language, e.g. 'English', 'Hindi'",
        },
    )


def build_eligibility_extraction_prompt(section_text: str) -> str:
    """
    Extract structured eligibility criteria from a policy section.

    Used by the ingestion pipeline to populate the eligibility engine's
    knowledge base in PostgreSQL.

    Parameters
    ----------
    section_text :
        The text of the eligibility/beneficiary section.

    Returns
    -------
    str
        Extraction prompt for eligibility criteria.
    """
    return build_json_extraction_prompt(
        base_prompt=f"""Extract all eligibility criteria from this government policy section.

SECTION TEXT:
{section_text}

Extract every eligibility condition mentioned:""",
        schema_hint={
            "criteria": [
                {
                    "criterion_type": "enum: age | income | occupation | land_holding | residence | caste_category | gender | disability | marital_status | other",
                    "description": "string — plain English description of the criterion",
                    "condition": "string — exact condition (e.g., 'age >= 18 AND age <= 60')",
                    "min_value": "number or null",
                    "max_value": "number or null",
                    "unit": "string — unit of measure (years, rupees, acres) or null",
                    "mandatory": "boolean — whether this criterion is mandatory",
                }
            ],
            "eligible_categories": "array of strings — explicitly listed eligible groups",
            "ineligible_categories": "array of strings — explicitly listed ineligible groups",
            "income_limit_annual": "number — annual income limit in INR, or null",
            "age_min": "number or null",
            "age_max": "number or null",
        },
    )


def build_benefit_extraction_prompt(section_text: str) -> str:
    """
    Extract benefit/entitlement details from a policy section.

    Parameters
    ----------
    section_text :
        The text of the benefits/entitlement section.

    Returns
    -------
    str
        Extraction prompt for benefit details.
    """
    return build_json_extraction_prompt(
        base_prompt=f"""Extract all benefits, entitlements, and financial assistance details \
from this government policy section.

SECTION TEXT:
{section_text}

Extract every benefit mentioned:""",
        schema_hint={
            "benefits": [
                {
                    "benefit_type": "enum: cash_transfer | subsidy | loan | insurance | pension | training | equipment | land | housing | other",
                    "description": "string — plain English description",
                    "amount_inr": "number — amount in Indian Rupees, or null",
                    "frequency": "enum: one_time | monthly | quarterly | annual | on_demand | null",
                    "duration_months": "number — duration in months, or null",
                    "conditions": "array of strings — conditions attached to this benefit",
                }
            ],
            "total_annual_benefit_inr": "number or null",
            "is_direct_benefit_transfer": "boolean",
        },
    )

"""
Policy Comparison Prompts
==========================

Prompts for comparing two or more government policies, identifying
overlaps, gaps, contradictions, and improvements.

Used by:
  - AIService.compare()                       (Week 1 — planned)
  - Week 3: agents/orchestrator.py            (comparison pipeline)
  - Frontend: Policy Comparison view          (Week 4+)

Comparison Dimensions
---------------------
The comparison engine analyses policies along these dimensions:
  1. Scope and coverage (who does each policy serve?)
  2. Financial benefits (amounts, frequency, caps)
  3. Eligibility criteria (are they overlapping? contradictory?)
  4. Geographic coverage (national vs. state-specific)
  5. Application procedure (complexity, required documents)
  6. Temporal validity (active vs. expired vs. superseded)
"""

from __future__ import annotations


def build_policy_comparison_prompt(
    policy_a_text: str,
    policy_b_text: str,
    *,
    policy_a_name: str = "Policy A",
    policy_b_name: str = "Policy B",
    comparison_focus: str | None = None,
) -> str:
    """
    Build a prompt comparing two government policies side-by-side.

    Parameters
    ----------
    policy_a_text :
        Summary or full text of the first policy.
    policy_b_text :
        Summary or full text of the second policy.
    policy_a_name :
        Human-readable name for policy A.
    policy_b_name :
        Human-readable name for policy B.
    comparison_focus :
        Optional focus area (e.g., "eligibility", "financial benefits",
        "application procedure").  Narrows the comparison scope.

    Returns
    -------
    str
        A structured comparison prompt returning JSON.
    """
    focus_instruction = (
        f"\nFocus specifically on comparing: {comparison_focus}\n"
        if comparison_focus
        else ""
    )

    return f"""Compare the following two government policies and provide a structured analysis.
{focus_instruction}
POLICY A — {policy_a_name}:
{policy_a_text}

POLICY B — {policy_b_name}:
{policy_b_text}

Provide a comprehensive comparison. Return a JSON object:
{{
    "summary": "<2-3 sentence high-level comparison>",
    "similarities": [
        {{
            "aspect": "<what aspect is similar>",
            "description": "<how they are similar>"
        }}
    ],
    "differences": [
        {{
            "aspect": "<what aspect differs>",
            "policy_a_position": "<what Policy A says>",
            "policy_b_position": "<what Policy B says>",
            "significance": "HIGH | MEDIUM | LOW"
        }}
    ],
    "policy_a_advantages": ["<advantages of Policy A over B>"],
    "policy_b_advantages": ["<advantages of Policy B over A>"],
    "overlap_risk": "HIGH | MEDIUM | LOW | NONE",
    "overlap_description": "<description of potential double-dipping or gaps, or null>",
    "recommendation": "<which policy is more comprehensive and why, or that both serve different purposes>"
}}

Return ONLY the JSON object."""


def build_eligibility_overlap_prompt(
    criteria_a: list[dict],
    criteria_b: list[dict],
    *,
    scheme_a_name: str = "Scheme A",
    scheme_b_name: str = "Scheme B",
) -> str:
    """
    Build a prompt to analyse eligibility overlap between two schemes.

    Helps determine if a person eligible for scheme A is automatically
    eligible for scheme B, or if they are mutually exclusive.

    Parameters
    ----------
    criteria_a :
        List of eligibility criterion dicts for scheme A (from extraction).
    criteria_b :
        List of eligibility criterion dicts for scheme B (from extraction).
    scheme_a_name :
        Human-readable name for scheme A.
    scheme_b_name :
        Human-readable name for scheme B.

    Returns
    -------
    str
        A structured eligibility overlap analysis prompt.
    """
    import json

    return f"""Analyse the eligibility criteria overlap between two government schemes.

{scheme_a_name} eligibility criteria:
{json.dumps(criteria_a, indent=2, ensure_ascii=False)}

{scheme_b_name} eligibility criteria:
{json.dumps(criteria_b, indent=2, ensure_ascii=False)}

Determine:
1. Can a beneficiary be eligible for BOTH schemes simultaneously?
2. Are any criteria mutually exclusive?
3. What is the overlap population (who qualifies for both)?

Return a JSON object:
{{
    "dual_eligibility_possible": <true | false>,
    "overlap_population_description": "<description of who qualifies for both>",
    "conflicting_criteria": [
        {{
            "criterion_type": "<type>",
            "scheme_a_requirement": "<requirement>",
            "scheme_b_requirement": "<requirement>",
            "conflict_type": "MUTUALLY_EXCLUSIVE | CONTRADICTORY | OVERLAPPING"
        }}
    ],
    "stricter_scheme": "{scheme_a_name} | {scheme_b_name} | EQUAL",
    "notes": "<any important observations about the eligibility relationship>"
}}

Return ONLY the JSON object."""


def build_temporal_comparison_prompt(
    old_policy_text: str,
    new_policy_text: str,
    *,
    old_effective_date: str | None = None,
    new_effective_date: str | None = None,
    policy_name: str = "Policy",
) -> str:
    """
    Build a prompt to identify changes between two versions of the same policy.

    Used when tracking policy amendments over time — a core feature of
    the Temporal Engine (Week 3).

    Parameters
    ----------
    old_policy_text :
        Text of the older policy version.
    new_policy_text :
        Text of the newer policy version.
    old_effective_date :
        Effective date of the old version (ISO 8601).
    new_effective_date :
        Effective date of the new version (ISO 8601).
    policy_name :
        Name of the policy being compared.

    Returns
    -------
    str
        A temporal change analysis prompt returning structured JSON.
    """
    date_context = ""
    if old_effective_date or new_effective_date:
        date_context = (
            f"\nOld version effective: {old_effective_date or 'unknown'}"
            f"\nNew version effective: {new_effective_date or 'unknown'}\n"
        )

    return f"""Identify all changes between these two versions of {policy_name}.
{date_context}
OLDER VERSION:
{old_policy_text}

NEWER VERSION:
{new_policy_text}

Categorise every change. Return a JSON object:
{{
    "total_changes": <integer>,
    "change_summary": "<1-2 sentence summary of what changed overall>",
    "changes": [
        {{
            "change_type": "ADDED | REMOVED | MODIFIED | CLARIFIED",
            "aspect": "<what aspect of the policy changed>",
            "old_value": "<old text or value, or null if added>",
            "new_value": "<new text or value, or null if removed>",
            "impact_assessment": "INCREASES_BENEFITS | REDUCES_BENEFITS | TIGHTENS_ELIGIBILITY | RELAXES_ELIGIBILITY | ADMINISTRATIVE | NEUTRAL",
            "affected_beneficiaries": "<which beneficiary groups are affected>"
        }}
    ],
    "net_beneficiary_impact": "POSITIVE | NEGATIVE | MIXED | NEUTRAL",
    "key_changes_summary": ["<top 3 most impactful changes>"]
}}

Return ONLY the JSON object."""

"""
Query Understanding Prompts
============================

Prompts for classifying user intent, extracting query entities, and
reformulating queries for better retrieval.

Used by:
  - Week 2: HybridRetriever (query reformulation)
  - Week 3: AgentaticRAG orchestrator (intent routing)
"""

from __future__ import annotations

from enum import Enum


class QueryIntent(str, Enum):
    """
    Taxonomy of user query intents in PolicyIntel AI.

    Used to route queries to the appropriate retrieval + generation pipeline.
    """

    ELIGIBILITY = "eligibility"         # "Am I eligible for PM Kisan?"
    PROCEDURE = "procedure"             # "How do I apply for Aadhaar?"
    COMPARISON = "comparison"           # "What is the difference between X and Y?"
    TEMPORAL = "temporal"               # "When did this policy change?"
    DEFINITION = "definition"           # "What is the Pradhan Mantri Awas Yojana?"
    AMOUNT = "amount"                   # "How much subsidy does PM Kisan provide?"
    DOCUMENT_LIST = "document_list"     # "What documents do I need for...?"
    UNKNOWN = "unknown"                 # Does not fit any known intent


def build_intent_classification_prompt(user_query: str) -> str:
    """
    Build a prompt to classify the intent of a user's policy question.

    The model should return a JSON object with the classified intent and
    extracted entities.

    Parameters
    ----------
    user_query :
        The raw user question in natural language.

    Returns
    -------
    str
        A complete prompt including the query and the JSON schema to fill.

    Example output (JSON from model)
    ---------------------------------
    {
        "intent": "eligibility",
        "confidence": 0.92,
        "entities": {
            "scheme_name": "PM Kisan",
            "beneficiary_type": "farmer",
            "location": null
        },
        "reformulated_query": "eligibility criteria for PM Kisan scheme for farmers"
    }
    """
    valid_intents = " | ".join(intent.value for intent in QueryIntent)
    return f"""Classify the intent of the following government policy query.

USER QUERY:
{user_query}

Valid intents: {valid_intents}

Return a JSON object with these exact fields:
{{
    "intent": "<one of the valid intents>",
    "confidence": <float between 0.0 and 1.0>,
    "entities": {{
        "scheme_name": "<scheme name or null>",
        "ministry": "<ministry name or null>",
        "beneficiary_type": "<e.g. farmer, student, widow or null>",
        "location": "<state or district or null>",
        "time_period": "<year or date range or null>"
    }},
    "reformulated_query": "<rephrased query optimised for document retrieval>"
}}

Return ONLY the JSON object. No explanation."""


def build_query_expansion_prompt(
    original_query: str,
    n_variants: int = 3,
) -> str:
    """
    Build a prompt that generates ``n_variants`` alternative phrasings of the query.

    Alternative phrasings improve recall in hybrid retrieval by covering
    different vocabulary that may appear in policy documents.

    Parameters
    ----------
    original_query :
        The original user query.
    n_variants :
        Number of alternative queries to generate.

    Returns
    -------
    str
        Prompt that asks the model to return a JSON array of query strings.

    Example output
    --------------
    {
        "variants": [
            "PM Kisan Samman Nidhi eligibility for small farmers",
            "who qualifies for Pradhan Mantri Kisan scheme",
            "PM Kisan 6000 rupee annual income support criteria"
        ]
    }
    """
    return f"""Generate {n_variants} alternative phrasings of the following government policy query.
The variants should:
- Use different vocabulary that might appear in official government documents
- Cover synonyms, acronyms, and formal/informal names for the same scheme
- Maintain the original meaning and intent

ORIGINAL QUERY: {original_query}

Return a JSON object with this exact structure:
{{
    "variants": [<list of {n_variants} string variants>]
}}

Return ONLY the JSON object."""


def build_entity_extraction_prompt(user_query: str) -> str:
    """
    Extract named entities from a policy query for use in graph traversal.

    Week 2 will use these entities to query the Neo4j knowledge graph.

    Parameters
    ----------
    user_query :
        The raw user question.

    Returns
    -------
    str
        A prompt for extracting policy-domain entities as structured JSON.
    """
    return f"""Extract all named entities from this government policy query.

QUERY: {user_query}

Entity types to extract:
- SCHEME: Government scheme or programme names (e.g., "PM Kisan", "MGNREGA")
- MINISTRY: Government ministry or department (e.g., "Ministry of Agriculture")
- PERSON_TYPE: Type of beneficiary (e.g., "farmer", "BPL family", "widow")
- LOCATION: Geographic entity (state, district, village)
- DATE: Any mentioned dates or time periods
- AMOUNT: Any monetary amounts or limits mentioned
- DOCUMENT: Any mentioned documents (e.g., "Aadhaar", "ration card")

Return ONLY a JSON object:
{{
    "entities": [
        {{
            "text": "<entity text as appears in query>",
            "type": "<entity type from the list above>",
            "canonical_form": "<normalised/official name if known, else same as text>"
        }}
    ]
}}"""

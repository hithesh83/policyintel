"""
Verification Prompts
====================

Prompts for fact-checking, answer verification, and claim validation
against retrieved policy documents.

Used by:
  - AIService.verify()                    (Week 1)
  - Week 3: agents/verifier.py            (answer verification agent)
  - Week 3: Grounded Answer pipeline      (post-generation verification)

Architecture Note
-----------------
The verifier agent (Week 3) will call AIService.verify() exclusively.
It will NEVER call OllamaClient directly.  Prompts here must be designed
to work with the verifier's output schema.
"""

from __future__ import annotations


def build_verification_prompt(
    claim_and_context: str,
    *,
    claim: str | None = None,
    context: str | None = None,
) -> str:
    """
    Build a fact-verification prompt.

    Can be called in two ways:

    1. Pre-formatted (simple): Pass ``claim_and_context`` with the full prompt text.
       Used by ``OllamaClient.verify()`` which receives a pre-built string.

    2. Structured: Pass ``claim`` and ``context`` separately for a well-structured
       verification prompt.  Used by Week 3 verifier agent.

    Parameters
    ----------
    claim_and_context :
        Pre-formatted string combining claim and context.
        Ignored when both ``claim`` and ``context`` are provided.
    claim :
        The specific claim or statement to verify.
    context :
        The policy text to verify the claim against.

    Returns
    -------
    str
        Complete verification prompt.
    """
    if claim is not None and context is not None:
        return _build_structured_verification(claim=claim, context=context)
    return claim_and_context.strip()


def _build_structured_verification(claim: str, context: str) -> str:
    """
    Build a structured verification prompt with explicit claim and context.

    Expected model output:
        VERDICT: SUPPORTED | REFUTED | INSUFFICIENT_EVIDENCE
        CONFIDENCE: HIGH | MEDIUM | LOW
        EXPLANATION: <1-3 sentence explanation>
        EVIDENCE: <direct quote from context>
    """
    return f"""Verify the following claim against the provided policy document context.

CLAIM TO VERIFY:
{claim}

POLICY CONTEXT:
{context}

VERIFICATION INSTRUCTIONS:
1. Compare the claim against the context carefully, word by word.
2. A claim is SUPPORTED only if the context explicitly states the same information.
3. A claim is REFUTED if the context explicitly contradicts it.
4. A claim is INSUFFICIENT_EVIDENCE if the context does not address it directly.
5. Do NOT use your general knowledge — base your verdict solely on the context.

Respond using EXACTLY this format:
VERDICT: [SUPPORTED | REFUTED | INSUFFICIENT_EVIDENCE]
CONFIDENCE: [HIGH | MEDIUM | LOW]
EXPLANATION: [1-3 sentences explaining your verdict based on the context]
EVIDENCE: [Quote the exact phrase from the context that supports your verdict, or "No direct evidence found"]"""


def build_answer_grounding_prompt(
    question: str,
    generated_answer: str,
    source_chunks: list[str],
) -> str:
    """
    Build a prompt to verify whether a generated answer is grounded in the source chunks.

    This is a post-generation quality check: after generating an answer via RAG,
    the verifier checks that each factual claim in the answer has a source in the
    retrieved chunks.

    Parameters
    ----------
    question :
        The original user question.
    generated_answer :
        The AI-generated answer to verify.
    source_chunks :
        The retrieved context chunks used to generate the answer.

    Returns
    -------
    str
        A prompt that asks the model to produce a JSON grounding report.
    """
    formatted_chunks = "\n\n".join(
        f"[SOURCE {i + 1}]: {chunk}" for i, chunk in enumerate(source_chunks)
    )

    return f"""You are a fact-checking specialist. Verify whether the following AI-generated \
answer is fully supported by the provided source documents.

ORIGINAL QUESTION:
{question}

AI-GENERATED ANSWER:
{generated_answer}

SOURCE DOCUMENTS:
{formatted_chunks}

TASK:
For each factual claim in the answer, determine if it is:
- SUPPORTED: Directly supported by text in the source documents
- UNSUPPORTED: Not found in or contradicted by the source documents

Return a JSON object:
{{
    "overall_verdict": "GROUNDED | PARTIALLY_GROUNDED | UNGROUNDED",
    "overall_confidence": "HIGH | MEDIUM | LOW",
    "claims": [
        {{
            "claim": "<specific factual claim from the answer>",
            "verdict": "SUPPORTED | UNSUPPORTED",
            "source_reference": "<source number and quote, or null if unsupported>"
        }}
    ],
    "unsupported_claims_count": <integer>,
    "recommendation": "PUBLISH | REVISE | REJECT"
}}

Return ONLY the JSON object."""


def build_contradiction_detection_prompt(
    text_a: str,
    text_b: str,
    *,
    doc_a_name: str = "Document A",
    doc_b_name: str = "Document B",
) -> str:
    """
    Build a prompt to detect contradictions between two policy texts.

    Used when the knowledge graph identifies that two policies govern
    the same domain, to flag potential conflicts for human review.

    Parameters
    ----------
    text_a :
        Text of the first policy document / section.
    text_b :
        Text of the second policy document / section.
    doc_a_name :
        Human-readable name for document A.
    doc_b_name :
        Human-readable name for document B.

    Returns
    -------
    str
        A contradiction detection prompt returning structured JSON.
    """
    return f"""Analyse the following two government policy texts and identify any contradictions, \
inconsistencies, or conflicts between them.

{doc_a_name}:
{text_a}

{doc_b_name}:
{text_b}

Return a JSON object:
{{
    "has_contradictions": <true | false>,
    "contradictions": [
        {{
            "aspect": "<what aspect of policy they conflict on>",
            "statement_a": "<what {doc_a_name} says>",
            "statement_b": "<what {doc_b_name} says>",
            "severity": "CRITICAL | SIGNIFICANT | MINOR"
        }}
    ],
    "compatible_aspects": ["<aspects where both documents agree>"],
    "resolution_suggestion": "<brief suggestion on how to resolve conflicts, or null>"
}}

Return ONLY the JSON object."""

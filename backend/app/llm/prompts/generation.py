"""
Generation Prompts
==================

Prompts for the core answer generation and summarisation use cases.

Used by:
  - AIService.generate_answer()       (Week 1)
  - AIService.summarize()             (Week 1)
  - Week 3: AgentaticRAG orchestrator (answer synthesis)
"""

from __future__ import annotations


def build_rag_answer_prompt(
    question: str,
    context_chunks: list[str],
    *,
    include_confidence: bool = True,
    max_answer_words: int = 400,
) -> str:
    """
    Build a Retrieval-Augmented Generation (RAG) prompt for answering
    a policy question given retrieved context chunks.

    Parameters
    ----------
    question :
        The user's original policy question.
    context_chunks :
        List of retrieved text chunks from policy documents.
        Each chunk should ideally include its source metadata (document name + page).
    include_confidence :
        If True, asks the model to include a confidence score in its response.
    max_answer_words :
        Soft cap on answer length.

    Returns
    -------
    str
        A complete RAG prompt ready to be sent to ``LLMClient.generate()``.
    """
    formatted_context = "\n\n---\n\n".join(
        f"CONTEXT [{i + 1}]:\n{chunk}" for i, chunk in enumerate(context_chunks)
    )

    confidence_instruction = (
        "\nAt the end of your answer, add:\n"
        "CONFIDENCE: [HIGH | MEDIUM | LOW] — with a one-sentence justification."
        if include_confidence
        else ""
    )

    return f"""You are PolicyIntel AI. Answer the following government policy question \
using ONLY the information provided in the context sections below.

QUESTION:
{question}

CONTEXT:
{formatted_context}

INSTRUCTIONS:
1. Answer directly and completely based solely on the provided context.
2. If the context does not contain enough information to answer, say: \
"The provided documents do not contain sufficient information to answer this question."
3. Cite the context number (e.g., [CONTEXT 1]) when referencing specific information.
4. Use bullet points for eligibility criteria, steps, or lists.
5. Keep your answer under {max_answer_words} words.
{confidence_instruction}

ANSWER:"""


def build_summarize_prompt(
    text: str,
    *,
    max_words: int = 150,
    focus: str | None = None,
) -> str:
    """
    Build a summarisation prompt for a policy document or text excerpt.

    Parameters
    ----------
    text :
        The text to summarise. Should be pre-chunked if very long.
    max_words :
        Approximate target word count for the summary.
    focus :
        Optional focus area to guide the summary (e.g., "eligibility criteria",
        "application procedure", "financial benefits").

    Returns
    -------
    str
        A complete summarisation prompt.
    """
    focus_instruction = (
        f"\nFocus specifically on: {focus}"
        if focus
        else ""
    )

    return f"""Summarise the following government policy text in approximately {max_words} words.{focus_instruction}

The summary should:
- Capture the key purpose and scope of the policy
- Mention specific beneficiaries, amounts, and dates if present
- Use clear, plain English accessible to a general audience
- Preserve important technical or legal terms with brief explanations

TEXT TO SUMMARISE:
{text}

SUMMARY:"""


def build_chunk_description_prompt(chunk_text: str, document_name: str) -> str:
    """
    Build a prompt to generate a semantic description of a policy document chunk.

    Used by the ingestion pipeline to generate rich metadata for each chunk
    before indexing into Qdrant.  Better descriptions → better retrieval.

    Parameters
    ----------
    chunk_text :
        The raw text of the chunk.
    document_name :
        Name of the source document (e.g., "PM Kisan SOP 2023.pdf").

    Returns
    -------
    str
        A prompt requesting a structured JSON description of the chunk.
    """
    return f"""Analyse the following excerpt from a government policy document and provide \
a structured description of its content.

SOURCE DOCUMENT: {document_name}

CHUNK TEXT:
{chunk_text}

Return a JSON object with these fields:
{{
    "topic": "<main topic of this chunk in 5-10 words>",
    "content_type": "<one of: eligibility_criteria | procedure | definition | benefit | \
penalty | general_information | schedule | amendment>",
    "key_entities": ["<list of key scheme names, ministries, or person types mentioned>"],
    "key_dates": ["<list of dates or time periods mentioned, ISO format if possible>"],
    "key_amounts": ["<list of monetary amounts or limits mentioned>"],
    "has_eligibility_criteria": <true | false>,
    "has_procedure_steps": <true | false>,
    "summary": "<1-2 sentence summary of this specific chunk>"
}}

Return ONLY the JSON object."""

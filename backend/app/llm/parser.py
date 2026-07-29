"""
Robust JSON Parser / Repair Utility
=====================================

LLMs often return output that *almost* parses as valid JSON but fails due to:

  1. Markdown code fences  (```json ... ```)
  2. Trailing commas       ({"a": 1,})
  3. Single quotes         ({'key': 'value'})
  4. Unquoted keys         ({key: "value"})
  5. Explanatory prose     before or after the JSON block
  6. Ellipsis / comments   (// ... or # ...)
  7. Python None/True/False instead of null/true/false

The repair pipeline is applied in order — each stage is lightweight and safe.
If the output still cannot be parsed after all stages, ``LLMJSONError`` is raised.

Architecture Note
-----------------
This module is intentionally framework-agnostic.  It has no FastAPI, no Ollama,
no HTTP dependencies.  It can be tested purely with string inputs.

The single public entry-point is ``parse_llm_json``.

Usage
-----
    from app.llm.parser import parse_llm_json, strip_markdown_fences

    data = parse_llm_json(raw_model_output, request_id="abc-123")
    # Returns: dict[str, Any]

    text = strip_markdown_fences(raw)
    # Returns: str (fences stripped, prose trimmed)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.llm.exceptions import LLMJSONError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns compiled once at module load for performance
# ---------------------------------------------------------------------------

# Matches ```json ... ``` or ``` ... ``` fences (multiline, non-greedy)
_MARKDOWN_FENCE_RE = re.compile(
    r"```(?:json|python|text|plain|JSON)?\s*(.*?)\s*```",
    re.DOTALL | re.IGNORECASE,
)

# Matches trailing commas before ] or }
# e.g.  [1, 2, 3,]  →  [1, 2, 3]
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")

# Matches Python-style single-line comments
_PYTHON_COMMENT_RE = re.compile(r"#[^\n]*")

# Matches JavaScript-style single-line comments  (// ...)
# Must not match URLs like https://
_JS_COMMENT_RE = re.compile(r"(?<!:)//[^\n]*")

# Matches /* ... */ block comments
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)

# Replace Python None/True/False with JSON null/true/false
# Word-boundary assertions prevent replacing substrings (e.g. "Trueness")
_PYTHON_LITERALS_RE = re.compile(r"\bNone\b|\bTrue\b|\bFalse\b")

_PYTHON_LITERAL_MAP = {
    "None": "null",
    "True": "true",
    "False": "false",
}

# Matches an unquoted JSON key followed by a colon
# e.g.  {key: "value"}  →  {"key": "value"}
# Note: This is intentionally conservative — only fixes simple identifiers.
_UNQUOTED_KEY_RE = re.compile(r'(?<=[{,])\s*([a-zA-Z_]\w*)\s*:')

# Find the first { or [ that starts a JSON structure in a prose string
_JSON_START_RE = re.compile(r"[{\[]")

# Find the last } or ] that ends a JSON structure
_JSON_END_RE = re.compile(r"[}\]]")

# ---------------------------------------------------------------------------
# Stage 1 — Strip markdown fences
# ---------------------------------------------------------------------------


def strip_markdown_fences(text: str) -> str:
    """
    Remove markdown code fences from a model response.

    If fences are found, returns the text INSIDE them.
    Otherwise returns the original text with leading/trailing whitespace removed.

    Examples
    --------
    >>> strip_markdown_fences('```json\\n{"a": 1}\\n```')
    '{"a": 1}'
    >>> strip_markdown_fences('{"a": 1}')
    '{"a": 1}'
    """
    match = _MARKDOWN_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Stage 2 — Extract JSON substring from prose
# ---------------------------------------------------------------------------


def extract_json_substring(text: str) -> str:
    """
    Attempt to locate the first JSON object or array within a larger string.

    Handles models that prepend explanatory prose:

        "Here is the extracted data: {"name": "Alice"}"

    Strategy: find the first ``{`` or ``[``, then find its matching close
    bracket using a simple depth counter.  This handles nested structures
    correctly.

    Returns the extracted substring, or the original text if no JSON
    structure is found.
    """
    start_match = _JSON_START_RE.search(text)
    if not start_match:
        return text

    start = start_match.start()
    opening = text[start]
    closing = "}" if opening == "{" else "]"
    depth = 0
    end = start

    for i, ch in enumerate(text[start:], start=start):
        if ch == opening:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                end = i
                break

    return text[start : end + 1] if depth == 0 else text


# ---------------------------------------------------------------------------
# Stage 3 — Comment removal
# ---------------------------------------------------------------------------


def remove_comments(text: str) -> str:
    """
    Strip JavaScript-style (``//``) and block (``/* */``) comments from JSON.

    Also strips Python ``#`` comments that some models incorrectly produce.
    """
    text = _BLOCK_COMMENT_RE.sub("", text)
    text = _JS_COMMENT_RE.sub("", text)
    text = _PYTHON_COMMENT_RE.sub("", text)
    return text


# ---------------------------------------------------------------------------
# Stage 4 — Python literal normalisation
# ---------------------------------------------------------------------------


def normalize_python_literals(text: str) -> str:
    """
    Replace Python-style ``None``, ``True``, ``False`` with JSON equivalents.

    The model was likely instructed to produce Python dicts and followed
    Python conventions rather than JSON conventions.
    """
    return _PYTHON_LITERALS_RE.sub(
        lambda m: _PYTHON_LITERAL_MAP[m.group(0)], text
    )


# ---------------------------------------------------------------------------
# Stage 5 — Single-quote normalisation
# ---------------------------------------------------------------------------


def normalize_quotes(text: str) -> str:
    """
    Convert single-quoted strings to double-quoted strings.

    This is deliberately simple and handles the most common case where the
    model produces ``{'key': 'value'}`` instead of ``{"key": "value"}``.

    Limitation: Does not handle escaped single quotes inside single-quoted
    strings.  For pathological output, the caller should use ``LLMJSONError``
    and rephrase the prompt.
    """
    # Only apply if the text appears to use single quotes
    if "'" in text and '"' not in text:
        return text.replace("'", '"')
    return text


# ---------------------------------------------------------------------------
# Stage 6 — Trailing comma removal
# ---------------------------------------------------------------------------


def remove_trailing_commas(text: str) -> str:
    """
    Remove trailing commas that are valid Python but invalid JSON.

    Example: ``{"a": 1,}``  →  ``{"a": 1}``
    """
    return _TRAILING_COMMA_RE.sub(r"\1", text)


# ---------------------------------------------------------------------------
# Stage 7 — Unquoted key quoting
# ---------------------------------------------------------------------------


def quote_unquoted_keys(text: str) -> str:
    """
    Wrap unquoted JSON keys in double quotes.

    Handles:  ``{key: "value"}``  →  ``{"key": "value"}``

    Conservative — only wraps simple alphanumeric identifiers.
    """
    return _UNQUOTED_KEY_RE.sub(r'"\1":', text)


# ---------------------------------------------------------------------------
# Public API — full repair pipeline
# ---------------------------------------------------------------------------


def parse_llm_json(
    raw: str,
    *,
    request_id: str | None = None,
) -> dict[str, Any]:
    """
    Parse LLM output into a Python dict, applying a multi-stage repair pipeline.

    Repair stages (in order)
    ------------------------
    1. Strip markdown fences
    2. Extract JSON substring from surrounding prose
    3. Remove JavaScript / Python comments
    4. Normalise Python literals  (None → null, True → true, False → false)
    5. Normalise single quotes    (' → ")
    6. Remove trailing commas     (,] → ], ,} → })
    7. Quote unquoted keys        ({key: v} → {"key": v})
    8. Final ``json.loads`` parse

    Parameters
    ----------
    raw :
        The raw text output from the LLM.
    request_id :
        Correlation ID passed into ``LLMJSONError`` for traceability.

    Returns
    -------
    dict[str, Any]
        The parsed JSON object.

    Raises
    ------
    LLMJSONError
        If the output cannot be parsed after all repair stages.
    ValueError
        If the parsed JSON is not a dict (e.g. the model returned a bare list).
        Callers that expect a list should use ``json.loads`` on ``raw_output`` instead.
    """
    if not raw or not raw.strip():
        raise LLMJSONError(
            "Model returned an empty response; cannot extract JSON.",
            raw_output=raw,
            request_id=request_id,
        )

    # Stage 1
    text = strip_markdown_fences(raw)
    logger.debug("After fence strip: %.120s", text)

    # Stage 2
    text = extract_json_substring(text)
    logger.debug("After JSON extraction: %.120s", text)

    # Stage 3
    text = remove_comments(text)

    # Stage 4
    text = normalize_python_literals(text)

    # Stage 5
    text = normalize_quotes(text)

    # Stage 6
    text = remove_trailing_commas(text)

    # Stage 7
    text = quote_unquoted_keys(text)

    # Stage 8 — final parse
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning(
            "JSON repair failed | request_id=%s | error=%s",
            request_id,
            exc,
        )
        raise LLMJSONError(
            f"JSON parse failed after repair: {exc}",
            raw_output=raw,
            request_id=request_id,
        ) from exc

    if not isinstance(parsed, dict):
        # The model returned a JSON array or primitive at the top level.
        logger.warning(
            "Model returned non-dict JSON (%s) | request_id=%s",
            type(parsed).__name__,
            request_id,
        )
        raise ValueError(f"Expected a JSON object (dict), but got {type(parsed).__name__}")

    return parsed

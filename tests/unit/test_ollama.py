"""
Unit Tests for OllamaClient
============================

Tests mock the ``httpx.AsyncClient`` — no real Ollama server required.

Coverage:
  - health()           : healthy server, unreachable server
  - generate()         : normal success, timeout, connection error
  - chat()             : normal success, HTTP 500 error
  - extract_json()     : clean JSON, markdown fences, trailing commas, LLMJSONError
  - verify()           : normal success
  - summarize()        : normal success
  - retry behaviour    : connection error retries, JSON error does NOT retry
  - payload builder    : correct Ollama payload structure
  - request_id         : unique per call

Run:
    pytest tests/unit/test_ollama.py -v
    pytest tests/unit/test_ollama.py -v --tb=short
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import the modules under test
# ---------------------------------------------------------------------------

from app.llm.exceptions import (
    LLMConnectionError,
    LLMJSONError,
    LLMResponseError,
    LLMTimeoutError,
)
from app.llm.models import ChatMessage, MessageRole
from app.llm.ollama import LLMSettings, OllamaClient
from app.llm.parser import (
    extract_json_substring,
    normalize_python_literals,
    normalize_quotes,
    parse_llm_json,
    remove_trailing_commas,
    strip_markdown_fences,
)


# ---------------------------------------------------------------------------
# Test Settings Fixture
# ---------------------------------------------------------------------------


def make_settings(**kwargs) -> LLMSettings:
    """Create a test LLMSettings with fast retry defaults."""
    defaults = {
        "url": "http://localhost:11434",
        "model": "qwen2.5:7b",
        "timeout": 5.0,
        "temperature": 0.7,
        "top_p": 0.9,
        "num_predict": -1,
        "max_retries": 2,
        "retry_min_wait": 0.01,
        "retry_max_wait": 0.05,
    }
    defaults.update(kwargs)
    return LLMSettings(**defaults)


def make_generate_response(text: str = "Test response") -> dict[str, Any]:
    """Build a mock Ollama /api/generate response body."""
    return {
        "model": "qwen2.5:7b",
        "response": text,
        "done": True,
        "prompt_eval_count": 10,
        "eval_count": 20,
    }


def make_chat_response(content: str = "Assistant reply") -> dict[str, Any]:
    """Build a mock Ollama /api/chat response body."""
    return {
        "model": "qwen2.5:7b",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "prompt_eval_count": 15,
        "eval_count": 25,
    }


def make_tags_response(models: list[str] | None = None) -> dict[str, Any]:
    """Build a mock Ollama /api/tags response body."""
    model_list = models or ["qwen2.5:7b"]
    return {
        "models": [{"name": m} for m in model_list]
    }


def make_mock_http_response(
    json_data: dict,
    status_code: int = 200,
) -> MagicMock:
    """Create a mock httpx Response object."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.json.return_value = json_data
    mock_response.text = json.dumps(json_data)
    if status_code >= 400:
        import httpx
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status_code}",
            request=MagicMock(),
            response=mock_response,
        )
    else:
        mock_response.raise_for_status.return_value = None
    return mock_response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def settings() -> LLMSettings:
    return make_settings()


@pytest.fixture
def client(settings: LLMSettings) -> OllamaClient:
    return OllamaClient(settings)


# ===========================================================================
# SECTION 1: health()
# ===========================================================================


class TestHealth:
    """Tests for OllamaClient.health()"""

    @pytest.mark.asyncio
    async def test_health_success(self, client: OllamaClient):
        """Health check returns 'healthy' when Ollama is running and model is present."""
        mock_resp = make_mock_http_response(make_tags_response(["qwen2.5:7b", "llama3:8b"]))

        with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_resp

            result = await client.health()

        assert result.status == "healthy"
        assert result.model == "qwen2.5:7b"
        assert result.backend == "ollama"
        assert result.latency_ms >= 0
        assert result.details["target_model_found"] is True
        assert "qwen2.5:7b" in result.details["available_models"]

    @pytest.mark.asyncio
    async def test_health_model_not_found(self, client: OllamaClient):
        """Health returns 'healthy' but target_model_found=False when model is missing."""
        mock_resp = make_mock_http_response(make_tags_response(["llama3:8b"]))

        with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_resp

            result = await client.health()

        assert result.status == "healthy"
        assert result.details["target_model_found"] is False

    @pytest.mark.asyncio
    async def test_health_connection_error(self, client: OllamaClient):
        """Health returns 'unhealthy' when the server is not reachable."""
        import httpx

        with patch.object(
            client._http, "get", new_callable=AsyncMock,
            side_effect=httpx.ConnectError("Connection refused"),
        ):
            result = await client.health()

        assert result.status == "unhealthy"
        assert "error" in result.details


# ===========================================================================
# SECTION 2: generate()
# ===========================================================================


class TestGenerate:
    """Tests for OllamaClient.generate()"""

    @pytest.mark.asyncio
    async def test_generate_success(self, client: OllamaClient):
        """generate() returns a GenerateResponse with text and metadata."""
        mock_resp = make_mock_http_response(
            make_generate_response("This is a test policy answer.")
        )

        with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp

            result = await client.generate("What is the PM Kisan scheme?")

        assert result.text == "This is a test policy answer."
        assert result.model == "qwen2.5:7b"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 20
        assert result.latency_ms >= 0
        assert result.request_id  # non-empty UUID

    @pytest.mark.asyncio
    async def test_generate_with_temperature_override(self, client: OllamaClient):
        """generate() passes temperature override to Ollama payload."""
        mock_resp = make_mock_http_response(make_generate_response("response"))

        with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp

            await client.generate("Test prompt", temperature=0.1)

        call_kwargs = mock_post.call_args
        payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs[0][1]
        assert payload["options"]["temperature"] == 0.1

    @pytest.mark.asyncio
    async def test_generate_unique_request_ids(self, client: OllamaClient):
        """Each generate() call produces a unique request_id."""
        mock_resp = make_mock_http_response(make_generate_response("ok"))

        with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp

            r1 = await client.generate("Prompt 1")
            r2 = await client.generate("Prompt 2")

        assert r1.request_id != r2.request_id

    @pytest.mark.asyncio
    async def test_generate_timeout_raises_llm_timeout_error(self, client: OllamaClient):
        """generate() raises LLMTimeoutError on httpx.TimeoutException."""
        import httpx

        with patch.object(
            client._http, "post", new_callable=AsyncMock,
            side_effect=httpx.TimeoutException("Request timed out"),
        ):
            with pytest.raises(LLMTimeoutError) as exc_info:
                await client.generate("A long prompt")

        assert exc_info.value.timeout_seconds == 5.0

    @pytest.mark.asyncio
    async def test_generate_connection_error_raises_llm_connection_error(
        self, client: OllamaClient
    ):
        """generate() raises LLMConnectionError on httpx.ConnectError."""
        import httpx

        with patch.object(
            client._http, "post", new_callable=AsyncMock,
            side_effect=httpx.ConnectError("Connection refused"),
        ):
            with pytest.raises(LLMConnectionError):
                await client.generate("Test prompt")

    @pytest.mark.asyncio
    async def test_generate_http_500_raises_llm_response_error(self, client: OllamaClient):
        """generate() raises LLMResponseError on HTTP 500."""
        error_resp = make_mock_http_response({"error": "internal server error"}, status_code=500)

        with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = error_resp

            with pytest.raises(LLMResponseError) as exc_info:
                await client.generate("Test")

        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_generate_retries_on_connection_error(self, client: OllamaClient):
        """generate() retries on LLMConnectionError and succeeds on 2nd attempt."""
        import httpx

        mock_success = make_mock_http_response(make_generate_response("success"))
        call_count = 0

        async def flaky_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ConnectError("Temporary connection failure")
            return mock_success

        with patch.object(client._http, "post", side_effect=flaky_post):
            result = await client.generate("Test retry")

        assert result.text == "success"
        assert call_count == 2  # Failed once, succeeded on retry

    @pytest.mark.asyncio
    async def test_generate_exhausts_retries_raises_connection_error(
        self, client: OllamaClient
    ):
        """generate() raises LLMConnectionError after exhausting all retries."""
        import httpx

        with patch.object(
            client._http, "post", new_callable=AsyncMock,
            side_effect=httpx.ConnectError("Server down"),
        ):
            with pytest.raises(LLMConnectionError):
                await client.generate("Test")

    @pytest.mark.asyncio
    async def test_generate_payload_structure(self, client: OllamaClient):
        """generate() sends the correct payload structure to Ollama."""
        mock_resp = make_mock_http_response(make_generate_response("ok"))

        with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp

            await client.generate("Hello Ollama", seed=42)

        call_args = mock_post.call_args
        endpoint = call_args[0][0]
        payload = call_args[1]["json"]

        assert endpoint == "/api/generate"
        assert payload["model"] == "qwen2.5:7b"
        assert payload["prompt"] == "Hello Ollama"
        assert payload["stream"] is False
        assert payload["options"]["seed"] == 42


# ===========================================================================
# SECTION 3: chat()
# ===========================================================================


class TestChat:
    """Tests for OllamaClient.chat()"""

    @pytest.mark.asyncio
    async def test_chat_success(self, client: OllamaClient):
        """chat() returns a ChatResponse with the assistant message."""
        mock_resp = make_mock_http_response(
            make_chat_response("I can help you with that policy question.")
        )
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are a policy expert."),
            ChatMessage(role=MessageRole.USER, content="What is PM Kisan?"),
        ]

        with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp

            result = await client.chat(messages)

        assert result.message.role == MessageRole.ASSISTANT
        assert result.message.content == "I can help you with that policy question."
        assert result.prompt_tokens == 15
        assert result.completion_tokens == 25
        assert result.request_id

    @pytest.mark.asyncio
    async def test_chat_payload_structure(self, client: OllamaClient):
        """chat() sends the correct message format to Ollama."""
        mock_resp = make_mock_http_response(make_chat_response("ok"))
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="System msg"),
            ChatMessage(role=MessageRole.USER, content="User msg"),
        ]

        with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp

            await client.chat(messages)

        payload = mock_post.call_args[1]["json"]
        assert payload["model"] == "qwen2.5:7b"
        assert payload["stream"] is False
        assert len(payload["messages"]) == 2
        assert payload["messages"][0] == {"role": "system", "content": "System msg"}
        assert payload["messages"][1] == {"role": "user", "content": "User msg"}

    @pytest.mark.asyncio
    async def test_chat_http_error_raises_response_error(self, client: OllamaClient):
        """chat() raises LLMResponseError on HTTP 503."""
        error_resp = make_mock_http_response({"error": "service unavailable"}, status_code=503)
        messages = [ChatMessage(role=MessageRole.USER, content="Test")]

        with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = error_resp

            with pytest.raises(LLMResponseError) as exc_info:
                await client.chat(messages)

        assert exc_info.value.status_code == 503


# ===========================================================================
# SECTION 4: extract_json()
# ===========================================================================


class TestExtractJson:
    """Tests for OllamaClient.extract_json()"""

    @pytest.mark.asyncio
    async def test_extract_json_clean_output(self, client: OllamaClient):
        """extract_json() succeeds when model returns clean JSON."""
        clean_json = '{"scheme_name": "PM Kisan", "amount_inr": 6000}'
        mock_resp = make_mock_http_response(make_generate_response(clean_json))

        with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp

            result = await client.extract_json("Extract scheme info from: PM Kisan 6000 INR")

        assert result.data["scheme_name"] == "PM Kisan"
        assert result.data["amount_inr"] == 6000
        assert result.raw_output == clean_json

    @pytest.mark.asyncio
    async def test_extract_json_strips_markdown_fences(self, client: OllamaClient):
        """extract_json() handles ```json ... ``` code fences."""
        fenced_json = '```json\n{"eligibility": "farmers only"}\n```'
        mock_resp = make_mock_http_response(make_generate_response(fenced_json))

        with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp

            result = await client.extract_json("Extract eligibility info")

        assert result.data["eligibility"] == "farmers only"

    @pytest.mark.asyncio
    async def test_extract_json_handles_trailing_commas(self, client: OllamaClient):
        """extract_json() repairs trailing commas."""
        bad_json = '{"scheme": "PM Kisan", "amount": 6000,}'
        mock_resp = make_mock_http_response(make_generate_response(bad_json))

        with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp

            result = await client.extract_json("Extract info")

        assert result.data["scheme"] == "PM Kisan"
        assert result.data["amount"] == 6000

    @pytest.mark.asyncio
    async def test_extract_json_handles_python_literals(self, client: OllamaClient):
        """extract_json() converts None/True/False to null/true/false."""
        python_json = '{"active": True, "amount": None, "archived": False}'
        mock_resp = make_mock_http_response(make_generate_response(python_json))

        with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp

            result = await client.extract_json("Extract status")

        assert result.data["active"] is True
        assert result.data["amount"] is None
        assert result.data["archived"] is False

    @pytest.mark.asyncio
    async def test_extract_json_raises_llm_json_error_on_invalid(self, client: OllamaClient):
        """extract_json() raises LLMJSONError when output cannot be parsed."""
        garbage = "I cannot provide that information in JSON format. Please try again."
        mock_resp = make_mock_http_response(make_generate_response(garbage))

        with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp

            with pytest.raises(LLMJSONError) as exc_info:
                await client.extract_json("Extract info")

        assert exc_info.value.raw_output == garbage

    @pytest.mark.asyncio
    async def test_extract_json_does_not_retry_on_json_error(self, client: OllamaClient):
        """LLMJSONError is NOT retried — the HTTP post is called only once."""
        garbage = "Sorry, I cannot produce JSON."
        mock_resp = make_mock_http_response(make_generate_response(garbage))

        with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp

            with pytest.raises(LLMJSONError):
                await client.extract_json("Extract")

        # Exactly 1 HTTP call — no retries for JSON failures
        assert mock_post.call_count == 1

    @pytest.mark.asyncio
    async def test_extract_json_uses_zero_temperature_by_default(self, client: OllamaClient):
        """extract_json() defaults to temperature=0.0 for deterministic output."""
        mock_resp = make_mock_http_response(make_generate_response('{"key": "val"}'))

        with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp

            await client.extract_json("Extract")

        payload = mock_post.call_args[1]["json"]
        assert payload["options"]["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_extract_json_prose_before_json(self, client: OllamaClient):
        """extract_json() extracts JSON embedded in surrounding prose."""
        prose_with_json = (
            'Here is the extracted information: {"scheme": "PM Awas"} '
            'Please let me know if you need anything else.'
        )
        mock_resp = make_mock_http_response(make_generate_response(prose_with_json))

        with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp

            result = await client.extract_json("Extract scheme info")

        assert result.data["scheme"] == "PM Awas"


# ===========================================================================
# SECTION 5: verify() and summarize()
# ===========================================================================


class TestVerifyAndSummarize:
    """Tests for OllamaClient.verify() and OllamaClient.summarize()"""

    @pytest.mark.asyncio
    async def test_verify_success(self, client: OllamaClient):
        """verify() returns a GenerateResponse with a verification verdict."""
        verdict = "VERDICT: SUPPORTED\nCONFIDENCE: HIGH\nEXPLANATION: The context confirms it."
        mock_resp = make_mock_http_response(make_generate_response(verdict))

        with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp

            result = await client.verify(
                "PM Kisan provides 6000 INR annual support to farmers."
            )

        assert "SUPPORTED" in result.text
        assert result.model == "qwen2.5:7b"

    @pytest.mark.asyncio
    async def test_verify_uses_low_temperature(self, client: OllamaClient):
        """verify() defaults to temperature=0.1 for conservative output."""
        mock_resp = make_mock_http_response(make_generate_response("VERDICT: SUPPORTED"))

        with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp

            await client.verify("Some claim")

        payload = mock_post.call_args[1]["json"]
        assert payload["options"]["temperature"] == 0.1

    @pytest.mark.asyncio
    async def test_summarize_success(self, client: OllamaClient):
        """summarize() returns a concise summary."""
        summary = "PM Kisan provides financial aid to small and marginal farmers."
        mock_resp = make_mock_http_response(make_generate_response(summary))

        with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp

            result = await client.summarize(
                "Pradhan Mantri Kisan Samman Nidhi (PM-KISAN) is a central sector scheme...",
                max_words=30,
            )

        assert result.text == summary

    @pytest.mark.asyncio
    async def test_summarize_uses_moderate_temperature(self, client: OllamaClient):
        """summarize() defaults to temperature=0.3."""
        mock_resp = make_mock_http_response(make_generate_response("Summary text"))

        with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp

            await client.summarize("Long policy document text here...")

        payload = mock_post.call_args[1]["json"]
        assert payload["options"]["temperature"] == 0.3


# ===========================================================================
# SECTION 6: JSON Parser Unit Tests (no HTTP mocking needed)
# ===========================================================================


class TestParser:
    """Unit tests for parser.py — pure string manipulation, no HTTP."""

    def test_strip_markdown_json_fence(self):
        """strip_markdown_fences removes ```json ... ``` fences."""
        raw = '```json\n{"key": "value"}\n```'
        assert strip_markdown_fences(raw) == '{"key": "value"}'

    def test_strip_markdown_plain_fence(self):
        """strip_markdown_fences removes ``` ... ``` fences."""
        raw = '```\n{"key": "value"}\n```'
        assert strip_markdown_fences(raw) == '{"key": "value"}'

    def test_strip_markdown_no_fence(self):
        """strip_markdown_fences returns stripped text unchanged when no fence."""
        raw = '  {"key": "value"}  '
        assert strip_markdown_fences(raw) == '{"key": "value"}'

    def test_extract_json_substring_with_prose(self):
        """extract_json_substring finds JSON embedded in prose."""
        prose = 'Here is the data: {"name": "Alice", "age": 30} — extracted successfully.'
        result = extract_json_substring(prose)
        assert result == '{"name": "Alice", "age": 30}'

    def test_extract_json_substring_nested(self):
        """extract_json_substring handles nested objects."""
        nested = 'Result: {"outer": {"inner": "value"}}'
        result = extract_json_substring(nested)
        assert result == '{"outer": {"inner": "value"}}'

    def test_extract_json_substring_array(self):
        """extract_json_substring handles top-level arrays."""
        array_str = 'Data: [1, 2, 3]'
        result = extract_json_substring(array_str)
        assert result == '[1, 2, 3]'

    def test_normalize_python_literals(self):
        """normalize_python_literals converts None/True/False correctly."""
        raw = '{"a": None, "b": True, "c": False}'
        result = normalize_python_literals(raw)
        assert '"a": null' in result
        assert '"b": true' in result
        assert '"c": false' in result

    def test_normalize_quotes_single_to_double(self):
        """normalize_quotes converts single-quoted JSON to double-quoted."""
        single_quoted = "{'key': 'value'}"
        result = normalize_quotes(single_quoted)
        assert result == '{"key": "value"}'

    def test_normalize_quotes_no_change_when_mixed(self):
        """normalize_quotes does not change text that already has double quotes."""
        double_quoted = '{"key": "value"}'
        result = normalize_quotes(double_quoted)
        assert result == double_quoted

    def test_remove_trailing_commas_object(self):
        """remove_trailing_commas fixes trailing commas in objects."""
        bad = '{"a": 1, "b": 2,}'
        result = remove_trailing_commas(bad)
        parsed = json.loads(result)
        assert parsed == {"a": 1, "b": 2}

    def test_remove_trailing_commas_array(self):
        """remove_trailing_commas fixes trailing commas in arrays."""
        bad = '[1, 2, 3,]'
        result = remove_trailing_commas(bad)
        parsed = json.loads(result)
        assert parsed == [1, 2, 3]

    def test_parse_llm_json_clean(self):
        """parse_llm_json parses clean JSON correctly."""
        raw = '{"scheme": "PM Kisan", "amount": 6000}'
        result = parse_llm_json(raw)
        assert result == {"scheme": "PM Kisan", "amount": 6000}

    def test_parse_llm_json_with_fences(self):
        """parse_llm_json handles markdown fences."""
        raw = '```json\n{"scheme": "PM Awas", "type": "housing"}\n```'
        result = parse_llm_json(raw)
        assert result["scheme"] == "PM Awas"

    def test_parse_llm_json_with_trailing_comma(self):
        """parse_llm_json handles trailing commas."""
        raw = '{"scheme": "MGNREGA", "days": 100,}'
        result = parse_llm_json(raw)
        assert result["days"] == 100

    def test_parse_llm_json_with_python_literals(self):
        """parse_llm_json handles Python None/True/False."""
        raw = '{"active": True, "superseded": None}'
        result = parse_llm_json(raw)
        assert result["active"] is True
        assert result["superseded"] is None

    def test_parse_llm_json_raises_on_garbage(self):
        """parse_llm_json raises LLMJSONError on unparseable output."""
        garbage = "I cannot provide a JSON response for this query."
        with pytest.raises(LLMJSONError) as exc_info:
            parse_llm_json(garbage, request_id="test-123")
        assert exc_info.value.raw_output == garbage
        assert exc_info.value.request_id == "test-123"

    def test_parse_llm_json_raises_on_empty_string(self):
        """parse_llm_json raises LLMJSONError on empty input."""
        with pytest.raises(LLMJSONError):
            parse_llm_json("")

    def test_parse_llm_json_raises_on_array(self):
        """parse_llm_json raises ValueError on top-level arrays."""
        raw = '[{"scheme": "PM Kisan"}, {"scheme": "PM Awas"}]'
        import pytest
        with pytest.raises(ValueError, match="Expected a JSON object"):
            parse_llm_json(raw)

    def test_parse_llm_json_nested_complex(self):
        """parse_llm_json handles complex nested JSON with repairs."""
        raw = '''```json
{
    "scheme": "PM Kisan",
    "eligibility": {
        "min_age": 18,
        "is_farmer": True,
        "income_limit": None,
    },
    "benefits": [6000, 2000, 2000,]
}
```'''
        result = parse_llm_json(raw)
        assert result["scheme"] == "PM Kisan"
        assert result["eligibility"]["is_farmer"] is True
        assert result["eligibility"]["income_limit"] is None
        assert result["benefits"] == [6000, 2000, 2000]


# ===========================================================================
# SECTION 7: LLMSettings
# ===========================================================================


class TestLLMSettings:
    """Tests for LLMSettings configuration."""

    def test_default_settings_from_env(self, monkeypatch):
        """LLMSettings reads from environment variables."""
        monkeypatch.setenv("OLLAMA_URL", "http://test-server:11434")
        monkeypatch.setenv("OLLAMA_MODEL", "llama3:8b")
        monkeypatch.setenv("OLLAMA_TIMEOUT", "60")
        monkeypatch.setenv("OLLAMA_TEMPERATURE", "0.5")

        settings = LLMSettings()
        assert settings.url == "http://test-server:11434"
        assert settings.model == "llama3:8b"
        assert settings.timeout == 60.0
        assert settings.temperature == 0.5

    def test_explicit_settings_override_env(self, monkeypatch):
        """Explicit constructor args override environment variables."""
        monkeypatch.setenv("OLLAMA_MODEL", "from-env")

        settings = LLMSettings(model="explicit-model")
        assert settings.model == "explicit-model"

    def test_settings_repr(self):
        """LLMSettings has a readable __repr__."""
        settings = make_settings()
        r = repr(settings)
        assert "LLMSettings" in r
        assert "qwen2.5:7b" in r


# ===========================================================================
# SECTION 8: Integration — OllamaClient + Parser together
# ===========================================================================


class TestIntegration:
    """End-to-end tests that combine HTTP mocking with parser."""

    @pytest.mark.asyncio
    async def test_full_extraction_pipeline(self, client: OllamaClient):
        """
        Simulates a complete extraction call where the model returns
        markdown-fenced JSON with trailing commas and Python literals.
        """
        model_output = '''Here is the extracted policy metadata:

```json
{
    "document_title": "PM Kisan Samman Nidhi",
    "ministry": "Ministry of Agriculture",
    "effective_date": "2019-12-01",
    "is_active": True,
    "supersedes": None,
    "target_beneficiaries": ["small farmers", "marginal farmers",]
}
```

This extraction is complete.'''

        mock_resp = make_mock_http_response(make_generate_response(model_output))

        with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp

            result = await client.extract_json(
                "Extract policy metadata from the document",
                schema_hint={"document_title": "str", "ministry": "str"},
            )

        assert result.data["document_title"] == "PM Kisan Samman Nidhi"
        assert result.data["ministry"] == "Ministry of Agriculture"
        assert result.data["is_active"] is True
        assert result.data["supersedes"] is None
        assert "small farmers" in result.data["target_beneficiaries"]

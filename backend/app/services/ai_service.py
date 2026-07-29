"""
AIService — Application-Level AI Abstraction Layer
====================================================

``AIService`` is the **single interface** through which all application
modules interact with the LLM layer.

Architecture
------------

    Application Code (pipeline, agents, API routes)
         │
         ▼
    AIService              ← You are here
         │
         ▼
    LLMClient (interface)
         │
         ▼
    OllamaClient           ← Concrete implementation (Week 1)
    (future: OpenAIClient, ClaudeClient, GeminiClient)
         │
         ▼
    Ollama REST API

Design Principles
-----------------
1. **Single Responsibility**: AIService translates high-level *business operations*
   (answer_policy_question, extract_policy_metadata, verify_answer) into LLMClient
   method calls.  It does NOT contain retrieval, database, or HTTP logic.

2. **Provider Agnostic**: AIService only knows about ``LLMClient``, never about
   ``OllamaClient``.  Switching providers requires zero changes here.

3. **Stateless**: AIService holds no mutable state (no conversation history, no
   per-user context).  It is safe to use as a singleton.

4. **Structured Logging**: Every method logs request_id, latency, and outcome.
   Errors are logged with full context before being re-raised.

Future Modules That Must Use AIService
---------------------------------------
- ``pipeline/extractor.py``      — policy metadata extraction
- ``pipeline/chunker.py``        — chunk description generation
- ``agents/orchestrator.py``     — query understanding + answer generation
- ``agents/verifier.py``         — answer grounding verification
- ``engine/eligibility.py``      — eligibility explanation generation
- ``engine/temporal.py``         — policy change summarisation

None of the above should import ``OllamaClient`` or any other LLM implementation.

Usage
-----
::

    # Via FastAPI Depends (preferred in routes):
    from app.llm.dependency import get_ai_service
    ai: AIService = Depends(get_ai_service)

    # Direct construction (scripts, tests):
    from app.llm.dependency import get_ai_service_instance
    ai = get_ai_service_instance()

    # Example call:
    result = await ai.answer_policy_question(
        question="Am I eligible for PM Kisan?",
        context_chunks=["PM Kisan eligibility: ...", "Annual income limit: ..."],
    )
    print(result.text)
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from app.core.logging import request_id_var
from app.llm.base import LLMClient
from app.llm.exceptions import LLMError, LLMJSONError
from app.llm.models import (
    ChatMessage,
    ChatResponse,
    ExtractionResponse,
    GenerateResponse,
    HealthResponse,
    MessageRole,
)
from app.llm.prompts.comparison import build_policy_comparison_prompt
from app.llm.prompts.extraction import (
    build_benefit_extraction_prompt,
    build_eligibility_extraction_prompt,
    build_policy_metadata_extraction_prompt,
)
from app.llm.prompts.generation import (
    build_chunk_description_prompt,
    build_rag_answer_prompt,
    build_summarize_prompt,
)
from app.llm.prompts.query import (
    build_entity_extraction_prompt,
    build_intent_classification_prompt,
    build_query_expansion_prompt,
)
from app.llm.prompts.system import build_system_prompt, build_verification_system_prompt
from app.llm.prompts.verification import (
    build_answer_grounding_prompt,
    build_verification_prompt,
)

logger = logging.getLogger(__name__)


class AIService:
    """
    High-level AI operations for PolicyIntel AI.

    All methods are async and return typed Pydantic models.  No raw strings
    are returned — every response has request_id, latency_ms, and model.

    Parameters
    ----------
    client :
        An ``LLMClient`` implementation.  Injected by the DI layer.
        Tests pass a mock client here.

    Attributes
    ----------
    client :
        The underlying LLM provider client (read-only after construction).
    """

    def __init__(self, client: LLMClient) -> None:
        self._client = client
        logger.info("AIService initialised with client: %s", type(client).__name__)

    @property
    def client(self) -> LLMClient:
        """The underlying LLM provider client."""
        return self._client

    # ------------------------------------------------------------------
    # Infrastructure
    # ------------------------------------------------------------------

    async def health(self) -> HealthResponse:
        """
        Check whether the LLM backend is healthy.

        Returns
        -------
        HealthResponse
            ``status`` is either 'healthy' or 'unhealthy'.
        """
        logger.info("AIService.health() called")
        return await self._client.health()

    # ------------------------------------------------------------------
    # Core Text Generation
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        seed: int | None = None,
        max_tokens: int | None = None,
    ) -> GenerateResponse:
        """
        Generate free-form text from a prompt.

        This is a low-level pass-through for use cases that need full control
        over the prompt.  Most callers should use higher-level methods like
        ``answer_policy_question`` or ``summarize_document``.

        Parameters
        ----------
        prompt :
            The complete prompt string.
        temperature :
            Sampling temperature override.
        seed :
            Seed for deterministic output.
        num_predict :
            Max tokens to generate.

        Returns
        -------
        GenerateResponse
        """
        request_id = request_id_var.get("") or str(uuid.uuid4())
        self._log_start("generate", request_id, prompt_size=len(prompt))
        start = time.monotonic()
        try:
            response = await self._client.generate(
                prompt,
                temperature=temperature,
                seed=seed,
                max_tokens=max_tokens,
                request_id=request_id,
            )
            self._log_ok("generate", request_id, start, response_size=len(response.text))
            return response
        except LLMError:
            self._log_error("generate", request_id, start)
            raise

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        seed: int | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        """
        Conduct a multi-turn conversation with the LLM.

        Parameters
        ----------
        messages :
            Ordered conversation history.
        temperature, seed, num_predict :
            Generation parameters.

        Returns
        -------
        ChatResponse
        """
        request_id = request_id_var.get("") or str(uuid.uuid4())
        total_chars = sum(len(m.content) for m in messages)
        self._log_start("chat", request_id, prompt_size=total_chars, extra={"n_messages": len(messages)})
        start = time.monotonic()
        try:
            response = await self._client.chat(
                messages,
                temperature=temperature,
                seed=seed,
                max_tokens=max_tokens,
                request_id=request_id,
            )
            self._log_ok("chat", request_id, start, response_size=len(response.message.content))
            return response
        except LLMError:
            self._log_error("chat", request_id, start)
            raise

    # ------------------------------------------------------------------
    # Structured Extraction
    # ------------------------------------------------------------------

    async def extract_json(
        self,
        prompt: str,
        *,
        schema_hint: dict[str, Any] | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> ExtractionResponse:
        """
        Extract structured JSON from a prompt.

        Applies the full JSON repair pipeline (see ``parser.py``).
        Raises ``LLMJSONError`` if parsing fails after all repair attempts.

        Parameters
        ----------
        prompt :
            Task description for the extraction.
        schema_hint :
            Optional JSON schema hint embedded in the prompt.
        temperature :
            Defaults to 0.0 for deterministic extraction.
        seed :
            Seed for reproducibility.

        Returns
        -------
        ExtractionResponse
            Contains ``data`` (parsed dict) and ``raw_output``.

        Raises
        ------
        LLMJSONError
            When JSON cannot be parsed.  NOT retried automatically.
        """
        request_id = request_id_var.get("") or str(uuid.uuid4())
        self._log_start("extract_json", request_id, prompt_size=len(prompt))
        start = time.monotonic()
        try:
            response = await self._client.extract_json(
                prompt,
                schema_hint=schema_hint,
                temperature=temperature,
                seed=seed,
                request_id=request_id,
            )
            self._log_ok("extract_json", request_id, start, extra={"keys": list(response.data.keys())})
            return response
        except LLMJSONError:
            self._log_error("extract_json", request_id, start, error_type="LLMJSONError")
            raise
        except LLMError:
            self._log_error("extract_json", request_id, start)
            raise

    # ------------------------------------------------------------------
    # High-Level Business Operations
    # ------------------------------------------------------------------

    async def answer_policy_question(
        self,
        question: str,
        context_chunks: list[str],
        *,
        include_confidence: bool = True,
        max_answer_words: int = 400,
    ) -> GenerateResponse:
        """
        Answer a user's policy question using retrieved context chunks (RAG).

        This is the primary generation method for the Q&A pipeline.
        Called by agents/orchestrator.py in Week 3.

        Parameters
        ----------
        question :
            The user's natural language question.
        context_chunks :
            Retrieved policy text chunks (from Qdrant + Neo4j hybrid search).
        include_confidence :
            If True, the answer will include a confidence assessment.
        max_answer_words :
            Soft cap on answer length.

        Returns
        -------
        GenerateResponse
            The model's answer with citation instructions followed.
        """
        request_id = request_id_var.get("") or str(uuid.uuid4())
        self._log_start(
            "answer_policy_question",
            request_id,
            prompt_size=len(question),
            extra={"n_chunks": len(context_chunks)},
        )
        start = time.monotonic()

        prompt = build_rag_answer_prompt(
            question=question,
            context_chunks=context_chunks,
            include_confidence=include_confidence,
            max_answer_words=max_answer_words,
        )

        try:
            response = await self._client.generate(prompt, temperature=0.3, request_id=request_id)
            self._log_ok("answer_policy_question", request_id, start, response_size=len(response.text))
            return response
        except LLMError:
            self._log_error("answer_policy_question", request_id, start)
            raise

    async def understand_query(
        self,
        user_query: str,
    ) -> ExtractionResponse:
        """
        Classify the intent and extract entities from a user's query.

        Used by the orchestrator to route the query to the correct retrieval
        and generation pipeline.

        Parameters
        ----------
        user_query :
            The raw user question.

        Returns
        -------
        ExtractionResponse
            ``data`` contains intent, confidence, entities, and reformulated_query.
        """
        request_id = request_id_var.get("") or str(uuid.uuid4())
        self._log_start("understand_query", request_id, prompt_size=len(user_query))
        start = time.monotonic()

        prompt = build_intent_classification_prompt(user_query)

        try:
            response = await self._client.extract_json(prompt, temperature=0.0, seed=42, request_id=request_id)
            self._log_ok("understand_query", request_id, start)
            return response
        except LLMError:
            self._log_error("understand_query", request_id, start)
            raise

    async def expand_query(
        self,
        user_query: str,
        n_variants: int = 3,
    ) -> ExtractionResponse:
        """
        Generate alternative phrasings of a user query to improve recall.

        Parameters
        ----------
        user_query :
            The original user question.
        n_variants :
            Number of alternative queries to generate.

        Returns
        -------
        ExtractionResponse
            ``data["variants"]`` contains the list of alternative queries.
        """
        request_id = request_id_var.get("") or str(uuid.uuid4())
        self._log_start("expand_query", request_id, prompt_size=len(user_query))
        start = time.monotonic()

        prompt = build_query_expansion_prompt(user_query, n_variants=n_variants)

        try:
            response = await self._client.extract_json(prompt, temperature=0.5, request_id=request_id)
            self._log_ok("expand_query", request_id, start)
            return response
        except LLMError:
            self._log_error("expand_query", request_id, start)
            raise

    async def extract_entities(self, user_query: str) -> ExtractionResponse:
        """
        Extract named entities from a user query for graph traversal.

        Parameters
        ----------
        user_query :
            The raw user question.

        Returns
        -------
        ExtractionResponse
            ``data["entities"]`` contains a list of entity dicts.
        """
        request_id = request_id_var.get("") or str(uuid.uuid4())
        prompt = build_entity_extraction_prompt(user_query)
        return await self.extract_json(prompt, temperature=0.0)

    async def summarize_document(
        self,
        text: str,
        *,
        max_words: int = 150,
        focus: str | None = None,
    ) -> GenerateResponse:
        """
        Summarise a policy document or text excerpt.

        Parameters
        ----------
        text :
            The text to summarise.  Pre-chunk very long texts.
        max_words :
            Approximate word count target.
        focus :
            Optional focus area (e.g., "eligibility criteria").

        Returns
        -------
        GenerateResponse
            ``text`` field contains the summary.
        """
        request_id = request_id_var.get("") or str(uuid.uuid4())
        self._log_start("summarize_document", request_id, prompt_size=len(text))
        start = time.monotonic()

        prompt = build_summarize_prompt(text, max_words=max_words, focus=focus)

        try:
            response = await self._client.generate(prompt, temperature=0.3, request_id=request_id)
            self._log_ok("summarize_document", request_id, start, response_size=len(response.text))
            return response
        except LLMError:
            self._log_error("summarize_document", request_id, start)
            raise

    async def verify_answer(
        self,
        claim: str,
        context: str,
    ) -> GenerateResponse:
        """
        Verify whether a claim is supported by the given policy context.

        Used by the verifier agent (Week 3) to check generated answers
        before returning them to the user.

        Parameters
        ----------
        claim :
            The factual statement to verify.
        context :
            The policy text to verify against.

        Returns
        -------
        GenerateResponse
            Text contains VERDICT, CONFIDENCE, EXPLANATION, and EVIDENCE.
        """
        request_id = request_id_var.get("") or str(uuid.uuid4())
        self._log_start("verify_answer", request_id, prompt_size=len(claim) + len(context))
        start = time.monotonic()

        prompt = build_verification_prompt(
            claim_and_context="",
            claim=claim,
            context=context,
        )

        try:
            response = await self._client.generate(prompt, temperature=0.1, request_id=request_id)
            self._log_ok("verify_answer", request_id, start, response_size=len(response.text))
            return response
        except LLMError:
            self._log_error("verify_answer", request_id, start)
            raise

    async def ground_answer(
        self,
        question: str,
        generated_answer: str,
        source_chunks: list[str],
    ) -> ExtractionResponse:
        """
        Verify that a generated answer is grounded in source documents.

        Returns a JSON grounding report with per-claim verdicts.

        Parameters
        ----------
        question :
            The original user question.
        generated_answer :
            The AI answer to check.
        source_chunks :
            The source documents used to generate the answer.

        Returns
        -------
        ExtractionResponse
            ``data`` contains overall_verdict, claims list, and recommendation.
        """
        prompt = build_answer_grounding_prompt(
            question=question,
            generated_answer=generated_answer,
            source_chunks=source_chunks,
        )
        return await self.extract_json(prompt, temperature=0.0)

    async def extract_policy_metadata(
        self,
        document_text: str,
    ) -> ExtractionResponse:
        """
        Extract top-level metadata from a policy document.

        Called by ``pipeline/extractor.py`` during PDF ingestion (Week 2).

        Parameters
        ----------
        document_text :
            The first few thousand characters of the policy document.

        Returns
        -------
        ExtractionResponse
            ``data`` contains title, ministry, dates, policy_type, etc.
        """
        prompt = build_policy_metadata_extraction_prompt(document_text)
        return await self.extract_json(prompt, temperature=0.0, seed=0)

    async def extract_eligibility(
        self,
        section_text: str,
    ) -> ExtractionResponse:
        """
        Extract structured eligibility criteria from a policy section.

        Called by ``pipeline/extractor.py`` when processing eligibility
        sections during PDF ingestion (Week 2).

        Parameters
        ----------
        section_text :
            The text of the eligibility section.

        Returns
        -------
        ExtractionResponse
            ``data["criteria"]`` contains structured eligibility conditions.
        """
        prompt = build_eligibility_extraction_prompt(section_text)
        return await self.extract_json(prompt, temperature=0.0, seed=0)

    async def extract_benefits(
        self,
        section_text: str,
    ) -> ExtractionResponse:
        """
        Extract structured benefit details from a policy section.

        Called by ``pipeline/extractor.py`` during PDF ingestion (Week 2).

        Parameters
        ----------
        section_text :
            The text of the benefits/entitlement section.

        Returns
        -------
        ExtractionResponse
            ``data["benefits"]`` contains structured benefit information.
        """
        prompt = build_benefit_extraction_prompt(section_text)
        return await self.extract_json(prompt, temperature=0.0, seed=0)

    async def generate_chunk_description(
        self,
        chunk_text: str,
        document_name: str,
    ) -> ExtractionResponse:
        """
        Generate a rich semantic description for a policy chunk.

        Used by the ingestion pipeline to create metadata for Qdrant indexing.
        Better descriptions improve retrieval quality.

        Parameters
        ----------
        chunk_text :
            The raw text of a document chunk.
        document_name :
            The name of the source document.

        Returns
        -------
        ExtractionResponse
            ``data`` contains topic, content_type, key entities, dates, amounts, summary.
        """
        prompt = build_chunk_description_prompt(
            chunk_text=chunk_text,
            document_name=document_name,
        )
        return await self.extract_json(prompt, temperature=0.0)

    async def compare_policies(
        self,
        policy_a_text: str,
        policy_b_text: str,
        *,
        policy_a_name: str = "Policy A",
        policy_b_name: str = "Policy B",
        comparison_focus: str | None = None,
    ) -> ExtractionResponse:
        """
        Compare two government policies and return a structured analysis.

        Parameters
        ----------
        policy_a_text :
            Text of the first policy.
        policy_b_text :
            Text of the second policy.
        policy_a_name :
            Name of first policy.
        policy_b_name :
            Name of second policy.
        comparison_focus :
            Optional specific aspect to focus on.

        Returns
        -------
        ExtractionResponse
            ``data`` contains similarities, differences, and recommendations.
        """
        prompt = build_policy_comparison_prompt(
            policy_a_text=policy_a_text,
            policy_b_text=policy_b_text,
            policy_a_name=policy_a_name,
            policy_b_name=policy_b_name,
            comparison_focus=comparison_focus,
        )
        return await self.extract_json(prompt, temperature=0.1)

    async def policy_chat(
        self,
        user_message: str,
        *,
        conversation_history: list[ChatMessage] | None = None,
        system_prompt: str | None = None,
    ) -> ChatResponse:
        """
        Conduct a multi-turn policy Q&A conversation.

        Assembles the message list with a system prompt and conversation
        history before calling the chat endpoint.

        Parameters
        ----------
        user_message :
            The latest user message.
        conversation_history :
            Previous turns in the conversation (alternating user/assistant).
            Should NOT include the system message.
        system_prompt :
            Custom system prompt.  Defaults to the PolicyIntel AI analyst persona.

        Returns
        -------
        ChatResponse
            The assistant's reply with role=assistant.
        """
        effective_system = system_prompt or build_system_prompt()

        messages: list[ChatMessage] = [
            ChatMessage(role=MessageRole.SYSTEM, content=effective_system),
            *(conversation_history or []),
            ChatMessage(role=MessageRole.USER, content=user_message),
        ]

        return await self.chat(messages, temperature=0.5)

    # ------------------------------------------------------------------
    # Private Logging Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _elapsed_ms(start: float) -> int:
        return int((time.monotonic() - start) * 1000)

    def _log_start(
        self,
        method: str,
        request_id: str,
        *,
        prompt_size: int = 0,
        extra: dict | None = None,
    ) -> None:
        extra_str = ""
        if extra:
            extra_str = " | " + " | ".join(f"{k}={v}" for k, v in extra.items())
        logger.info(
            "AIService.%s() START | request_id=%s | prompt_chars=%d%s",
            method,
            request_id,
            prompt_size,
            extra_str,
        )

    def _log_ok(
        self,
        method: str,
        request_id: str,
        start: float,
        *,
        response_size: int = 0,
        extra: dict | None = None,
    ) -> None:
        latency_ms = self._elapsed_ms(start)
        extra_str = ""
        if extra:
            extra_str = " | " + " | ".join(f"{k}={v}" for k, v in extra.items())
        logger.info(
            "AIService.%s() OK | request_id=%s | latency_ms=%d | response_chars=%d%s",
            method,
            request_id,
            latency_ms,
            response_size,
            extra_str,
        )

    def _log_error(
        self,
        method: str,
        request_id: str,
        start: float,
        *,
        error_type: str = "LLMError",
    ) -> None:
        latency_ms = self._elapsed_ms(start)
        logger.error(
            "AIService.%s() FAILED | request_id=%s | latency_ms=%d | error_type=%s",
            method,
            request_id,
            latency_ms,
            error_type,
        )

# ==============================================================================
# PolicyIntel AI — LLM Layer
# ==============================================================================
# Week 1: AI Foundation
# Architecture: Application → AIService → LLMClient → OllamaClient → Ollama API
#
# This README documents the LLM subsystem architecture, usage, and extensibility.
# ==============================================================================

# Architecture Overview

```
Application Code (API routes, agents, pipeline)
    │
    ▼
AIService                          ← Single interface for ALL AI operations
    │                                 Never instantiated directly; use Depends(get_ai_service)
    │
    ▼
LLMClient (Abstract Interface)     ← Contract that all providers must implement
    │                                 base.py → abstract methods: health, generate, chat,
    │                                 extract_json, verify, summarize
    │
    ▼
OllamaClient                       ← Concrete Ollama implementation
    │                                 ollama.py → httpx.AsyncClient
    │                                 Retry: tenacity exponential backoff
    │                                 JSON repair: parser.py
    │
    ▼
Ollama REST API                    ← http://localhost:11434
    POST /api/generate             ← single-turn text completion
    POST /api/chat                 ← multi-turn conversation
    GET  /api/tags                 ← health check / model list
```

## Dependency Graph

```
app/llm/__init__.py
    ├── base.py           ← LLMClient (ABC)
    ├── exceptions.py     ← LLMError, LLMConnectionError, LLMTimeoutError,
    │                        LLMJSONError, LLMResponseError
    ├── models.py         ← Pydantic V2: GenerateRequest, ChatMessage, etc.
    ├── parser.py         ← parse_llm_json (7-stage repair pipeline)
    ├── ollama.py         ← OllamaClient(LLMClient), LLMSettings
    ├── dependency.py     ← get_llm(), get_ai_service() FastAPI Depends
    └── prompts/
        ├── system.py     ← System persona prompts
        ├── query.py      ← Intent classification, query expansion, entity extraction
        ├── generation.py ← RAG answer, summarisation, chunk description
        ├── verification.py ← Claim verification, grounding, contradiction detection
        ├── extraction.py ← Policy metadata, eligibility, benefits extraction
        └── comparison.py ← Policy comparison, temporal change analysis

app/services/
    └── ai_service.py     ← AIService (uses LLMClient, never OllamaClient directly)
```

## Sequence Diagram — Q&A Request

```
User Request
    │
    ▼
FastAPI Route Handler
    │  Depends(get_ai_service)
    ▼
AIService.answer_policy_question(question, context_chunks)
    │  build_rag_answer_prompt(question, context_chunks)
    ▼
LLMClient.generate(prompt)          ← AIService calls the interface, not OllamaClient
    │
    ▼
OllamaClient._post_with_retry()     ← tenacity retries on ConnectionError/TimeoutError
    │
    ▼
POST http://localhost:11434/api/generate
    │
    ▼
Ollama → qwen2.5:7b inference
    │
    ▼
GenerateResponse(text, tokens, latency_ms, request_id)
    │
    ▼
User Response
```

## Sequence Diagram — JSON Extraction

```
AIService.extract_policy_metadata(document_text)
    │  build_policy_metadata_extraction_prompt(text)
    ▼
LLMClient.extract_json(prompt, schema_hint)
    │
    ▼
OllamaClient.extract_json()
    │  POST /api/generate (temperature=0.0)
    ▼
parser.parse_llm_json(raw_output)
    │  Stage 1: strip_markdown_fences()
    │  Stage 2: extract_json_substring()
    │  Stage 3: remove_comments()
    │  Stage 4: normalize_python_literals()
    │  Stage 5: normalize_quotes()
    │  Stage 6: remove_trailing_commas()
    │  Stage 7: quote_unquoted_keys()
    │  Stage 8: json.loads()
    ▼
ExtractionResponse(data: dict, raw_output, model, latency_ms, request_id)
```

## How Ollama Works

Ollama runs a local inference server at `http://localhost:11434`.

Key endpoints used:

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/generate` | POST | Single-prompt text completion |
| `/api/chat` | POST | Multi-turn conversation with message history |
| `/api/tags` | GET | List available models (used for health check) |

### Starting Ollama

```bash
# Install Ollama (macOS)
brew install ollama

# Pull the model
ollama pull qwen2.5:7b

# Start the server (runs on :11434 by default)
ollama serve
```

### Configuration

All Ollama settings are read from `.env`:

```env
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_TIMEOUT=120
OLLAMA_TEMPERATURE=0.7
OLLAMA_TOP_P=0.9
OLLAMA_NUM_PREDICT=-1
OLLAMA_MAX_RETRIES=3
OLLAMA_RETRY_MIN_WAIT=1.0
OLLAMA_RETRY_MAX_WAIT=30.0
```

## How AIService Works

`AIService` is the **only** class that application code uses for LLM operations.

### High-Level Methods

| Method | Purpose | Used By |
|---|---|---|
| `answer_policy_question()` | RAG answer generation | agents/orchestrator.py |
| `understand_query()` | Intent classification + entity extraction | agents/orchestrator.py |
| `expand_query()` | Generate alternative query phrasings | retrieval/hybrid.py |
| `extract_policy_metadata()` | Extract document-level metadata from PDFs | pipeline/extractor.py |
| `extract_eligibility()` | Extract structured eligibility criteria | pipeline/extractor.py |
| `extract_benefits()` | Extract structured benefit details | pipeline/extractor.py |
| `generate_chunk_description()` | Semantic chunk description for Qdrant indexing | pipeline/chunker.py |
| `verify_answer()` | Fact-check a claim against context | agents/verifier.py |
| `ground_answer()` | Check if answer is grounded in sources | agents/verifier.py |
| `compare_policies()` | Side-by-side policy comparison | engine/ |
| `summarize_document()` | Summarise a policy text | pipeline/ |
| `policy_chat()` | Multi-turn conversation with history | API routes |

### Low-Level Methods

| Method | Purpose |
|---|---|
| `health()` | LLM backend health check |
| `generate()` | Raw text generation (use high-level methods instead) |
| `chat()` | Raw multi-turn chat (use `policy_chat()` instead) |
| `extract_json()` | Raw JSON extraction (use semantic methods instead) |

## How to Add a New Provider

To swap Ollama for OpenAI, Claude, Gemini, or Azure OpenAI:

### Step 1 — Create the client

```python
# app/llm/openai_client.py

from app.llm.base import LLMClient
from app.llm.models import GenerateResponse, ChatResponse, ...

class OpenAIClient(LLMClient):
    """OpenAI GPT implementation of LLMClient."""

    def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def health(self) -> HealthResponse: ...
    async def generate(self, prompt: str, ...) -> GenerateResponse: ...
    async def chat(self, messages: list[ChatMessage], ...) -> ChatResponse: ...
    async def extract_json(self, prompt: str, ...) -> ExtractionResponse: ...
    async def verify(self, prompt: str, ...) -> GenerateResponse: ...
    async def summarize(self, text: str, ...) -> GenerateResponse: ...
```

### Step 2 — Update dependency.py

```python
# app/llm/dependency.py

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

async def initialise_llm_client(settings=None):
    if LLM_PROVIDER == "openai":
        client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))
    elif LLM_PROVIDER == "claude":
        client = ClaudeClient(api_key=os.getenv("ANTHROPIC_API_KEY"))
    else:
        client = OllamaClient(settings or LLMSettings())
    ...
```

### Step 3 — Set environment variable

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

**Zero changes to AIService or any business logic.** ✓

## Example Usage

### In a FastAPI route

```python
from fastapi import APIRouter, Depends
from app.llm.dependency import get_ai_service
from app.services.ai_service import AIService

router = APIRouter()

@router.post("/answer")
async def answer(
    question: str,
    ai: AIService = Depends(get_ai_service),
):
    result = await ai.answer_policy_question(
        question=question,
        context_chunks=["PM Kisan provides 6000 INR annually to farmers..."],
    )
    return {"answer": result.text, "latency_ms": result.latency_ms}
```

### In a pipeline script

```python
import asyncio
from app.llm.dependency import get_ai_service_instance

async def main():
    ai = get_ai_service_instance()
    try:
        result = await ai.extract_policy_metadata(
            document_text="PM Kisan Samman Nidhi — Ministry of Agriculture — 2019..."
        )
        print(result.data)
    finally:
        await ai.client.aclose()

asyncio.run(main())
```

### JSON Extraction with schema hint

```python
result = await ai.extract_json(
    prompt="Extract the scheme name and annual benefit amount from: PM Kisan provides 6000 INR per year.",
    schema_hint={"scheme_name": "string", "annual_benefit_inr": "number"},
)
print(result.data)
# → {"scheme_name": "PM Kisan", "annual_benefit_inr": 6000}
```

## Running Tests

```bash
# Install dependencies
pip install -r backend/requirements.txt

# Run all unit tests (no Ollama required)
cd /path/to/policyintel
pytest tests/unit/test_ollama.py -v

# Run with coverage
pytest tests/unit/test_ollama.py -v --cov=app/llm --cov=app/services --cov-report=term-missing

# Run all tests
pytest tests/ -v
```

## Clean Architecture Explanation

The LLM layer follows **Clean Architecture** (Robert C. Martin):

```
┌────────────────────────────────────────────────┐
│            Frameworks & Drivers                │
│   httpx, FastAPI, Ollama REST API              │
│   OllamaClient (adapts Ollama to LLMClient)   │
├────────────────────────────────────────────────┤
│          Interface Adapters                    │
│   dependency.py (DI wiring)                   │
│   parser.py (data transformation)             │
│   prompts/ (data formatting)                  │
├────────────────────────────────────────────────┤
│          Application Business Rules           │
│   AIService (use cases)                       │
│   e.g. answer_policy_question = RAG use case  │
├────────────────────────────────────────────────┤
│          Enterprise Business Rules            │
│   LLMClient (interface/contract)              │
│   models.py (domain data shapes)              │
│   exceptions.py (domain errors)               │
└────────────────────────────────────────────────┘
```

The **Dependency Rule**: dependencies point inward only.
- `OllamaClient` knows about `LLMClient` → ✓
- `LLMClient` knows nothing about `OllamaClient` → ✓
- `AIService` knows about `LLMClient` → ✓
- `AIService` knows nothing about `OllamaClient` → ✓
- `FastAPI routes` know about `AIService` → ✓
- `AIService` knows nothing about FastAPI → ✓

This means you can test AIService without Ollama, test OllamaClient without FastAPI,
and swap providers without touching AIService.

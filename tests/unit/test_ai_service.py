"""
Unit tests for AIService using a mocked LLMClient.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from app.llm.base import LLMClient
from app.llm.models import GenerateResponse, ChatResponse, ChatMessage, MessageRole
from app.services.ai_service import AIService


class MockLLMClient(LLMClient):
    def __init__(self):
        self.health_mock = AsyncMock()
        self.generate_mock = AsyncMock()
        self.chat_mock = AsyncMock()
        self.extract_json_mock = AsyncMock()
        self.verify_mock = AsyncMock()
        self.summarize_mock = AsyncMock()
        
    async def health(self, *, request_id=None):
        return await self.health_mock(request_id=request_id)
        
    async def generate(self, prompt, *, temperature=None, seed=None, max_tokens=None, request_id=None):
        return await self.generate_mock(prompt=prompt, temperature=temperature, seed=seed, max_tokens=max_tokens, request_id=request_id)
        
    async def chat(self, messages, *, temperature=None, seed=None, max_tokens=None, request_id=None):
        return await self.chat_mock(messages=messages, temperature=temperature, seed=seed, max_tokens=max_tokens, request_id=request_id)
        
    async def extract_json(self, prompt, *, schema_hint=None, temperature=None, seed=None, request_id=None):
        return await self.extract_json_mock(prompt=prompt, schema_hint=schema_hint, temperature=temperature, seed=seed, request_id=request_id)
        
    async def verify(self, prompt, *, temperature=None, request_id=None):
        return await self.verify_mock(prompt=prompt, temperature=temperature, request_id=request_id)
        
    async def summarize(self, text, *, max_words=150, temperature=None, request_id=None):
        return await self.summarize_mock(text=text, max_words=max_words, temperature=temperature, request_id=request_id)


@pytest.fixture
def mock_client():
    return MockLLMClient()


@pytest.fixture
def ai_service(mock_client):
    return AIService(client=mock_client)


@pytest.mark.asyncio
async def test_generate_passes_request_id(ai_service, mock_client):
    mock_client.generate_mock.return_value = GenerateResponse(
        text="Mock response",
        model="test",
        request_id="req-123"
    )
    
    with patch("app.services.ai_service.request_id_var") as mock_var:
        mock_var.get.return_value = "custom-req-id"
        response = await ai_service.generate("Test prompt")
        
    assert response.text == "Mock response"
    mock_client.generate_mock.assert_called_once()
    _, kwargs = mock_client.generate_mock.call_args
    assert kwargs["request_id"] == "custom-req-id"
    assert kwargs["prompt"] == "Test prompt"


@pytest.mark.asyncio
async def test_chat_passes_request_id(ai_service, mock_client):
    mock_client.chat_mock.return_value = ChatResponse(
        message=ChatMessage(role=MessageRole.ASSISTANT, content="Hi"),
        model="test",
        request_id="req-123"
    )
    
    messages = [ChatMessage(role=MessageRole.USER, content="Hello")]
    
    with patch("app.services.ai_service.request_id_var") as mock_var:
        mock_var.get.return_value = "custom-req-id-chat"
        response = await ai_service.chat(messages)
        
    assert response.message.content == "Hi"
    mock_client.chat_mock.assert_called_once()
    _, kwargs = mock_client.chat_mock.call_args
    assert kwargs["request_id"] == "custom-req-id-chat"
    assert kwargs["messages"] == messages


@pytest.mark.asyncio
async def test_high_level_method_answer_policy_question(ai_service, mock_client):
    mock_client.generate_mock.return_value = GenerateResponse(
        text="Yes, you are eligible.",
        model="test",
        request_id="123"
    )
    
    response = await ai_service.answer_policy_question(
        question="Am I eligible?",
        context_chunks=["chunk1"]
    )
    
    assert response.text == "Yes, you are eligible."
    mock_client.generate_mock.assert_called_once()
    args, kwargs = mock_client.generate_mock.call_args
    assert "Am I eligible?" in kwargs["prompt"]
    assert "chunk1" in kwargs["prompt"]
    assert kwargs["temperature"] == 0.3

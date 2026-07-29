"""
Root conftest.py — shared test fixtures for the entire test suite.

Pytest configuration and shared fixtures used by:
  - tests/unit/
  - tests/integration/
"""

from __future__ import annotations

import os

import pytest

# ---------------------------------------------------------------------------
# Set test environment variables BEFORE any imports of app code
# ---------------------------------------------------------------------------

os.environ.setdefault("OLLAMA_URL", "http://localhost:11434")
os.environ.setdefault("OLLAMA_MODEL", "qwen2.5:7b")
os.environ.setdefault("OLLAMA_TIMEOUT", "10")
os.environ.setdefault("OLLAMA_TEMPERATURE", "0.7")
os.environ.setdefault("OLLAMA_TOP_P", "0.9")
os.environ.setdefault("OLLAMA_NUM_PREDICT", "-1")
os.environ.setdefault("OLLAMA_MAX_RETRIES", "2")
os.environ.setdefault("OLLAMA_RETRY_MIN_WAIT", "0.01")
os.environ.setdefault("OLLAMA_RETRY_MAX_WAIT", "0.1")


# ---------------------------------------------------------------------------
# Asyncio mode configuration (pytest-asyncio)
# ---------------------------------------------------------------------------

# This tells pytest-asyncio to handle async test functions automatically.
# Placed here so all test files benefit without needing @pytest.mark.asyncio
# on every test class (though we add it explicitly for clarity).


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires running services)"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test (no external services required)"
    )

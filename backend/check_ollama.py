#!/usr/bin/env python3
"""
Ollama Environment Verification Script
======================================

Validates that the local LLM environment is correctly set up for PolicyIntel AI.
It verifies that the Ollama server is reachable, the required model is installed,
and the REST API can successfully generate completions.

Usage:
    python backend/check_ollama.py

Configuration:
    OLLAMA_URL: Base URL for the Ollama server (default: http://localhost:11434)
    OLLAMA_MODEL: The required model to check (default: qwen2.5:7b)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"
TEST_PROMPT = "Explain the importance of public policy in one short sentence."
TIMEOUT_SECONDS = 30.0

# ---------------------------------------------------------------------------
# ANSI Colors for Terminal Output
# ---------------------------------------------------------------------------

class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def print_step(message: str) -> None:
    """Print a step indicator."""
    print(f"{Colors.OKCYAN}▶ {message}{Colors.ENDC}")


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"{Colors.OKGREEN}✔ {message}{Colors.ENDC}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"{Colors.FAIL}✖ {message}{Colors.ENDC}")


def print_info(message: str) -> None:
    """Print an informational message."""
    print(f"  {message}")

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class OllamaCheckError(Exception):
    """Base exception for Ollama verification failures."""
    pass


class OllamaConnectionError(OllamaCheckError):
    """Raised when the Ollama server is unreachable."""
    pass


class OllamaModelNotFoundError(OllamaCheckError):
    """Raised when the required model is not installed."""
    pass

# ---------------------------------------------------------------------------
# Core Logic
# ---------------------------------------------------------------------------

async def check_ollama_server(client: httpx.AsyncClient) -> list[str]:
    """
    Check if the Ollama server is running and return available models.
    """
    print_step("Checking Ollama server connectivity...")
    try:
        response = await client.get("/api/tags")
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        models = [m.get("name") for m in data.get("models", [])]
        
        print_success("Ollama server is running.")
        if models:
            print_info(f"Installed models: {', '.join(models)}")
        else:
            print_info("No models currently installed.")
        
        return models
    except httpx.ConnectError as exc:
        raise OllamaConnectionError(
            "Could not connect to Ollama server. Is it running?"
        ) from exc
    except httpx.HTTPError as exc:
        raise OllamaCheckError(f"HTTP error communicating with Ollama: {exc}") from exc


async def test_generation(client: httpx.AsyncClient, model: str) -> None:
    """
    Test the generation API endpoint and measure response time.
    """
    print_step(f"Testing text generation with '{model}'...")
    print_info(f"Prompt: \"{TEST_PROMPT}\"")
    
    payload = {
        "model": model,
        "prompt": TEST_PROMPT,
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }
    
    start_time = time.monotonic()
    try:
        response = await client.post("/api/generate", json=payload)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        
        elapsed_time = time.monotonic() - start_time
        generated_text = data.get("response", "").strip()
        
        print_success(f"Generation successful (Response time: {elapsed_time:.2f}s)")
        print_info(f"Model response: \"{generated_text}\"")
        
    except httpx.TimeoutException as exc:
        raise OllamaCheckError(
            f"Generation timed out after {TIMEOUT_SECONDS}s"
        ) from exc
    except httpx.HTTPError as exc:
        raise OllamaCheckError(f"API request failed: {exc}") from exc


async def run_checks(url: str, required_model: str) -> bool:
    """
    Run the full suite of verification checks.
    
    Returns True if all checks pass, False otherwise.
    """
    print(f"\n{Colors.BOLD}=== PolicyIntel AI - Ollama Environment Check ==={Colors.ENDC}\n")
    print_info(f"Server URL:    {url}")
    print_info(f"Target Model:  {required_model}\n")
    
    async with httpx.AsyncClient(
        base_url=url, 
        timeout=httpx.Timeout(TIMEOUT_SECONDS)
    ) as client:
        try:
            # 1. Check server & get models
            models = await check_ollama_server(client)
            
            # 2. Verify required model exists
            print_step(f"Verifying '{required_model}' is installed...")
            if required_model not in models:
                raise OllamaModelNotFoundError(
                    f"Model '{required_model}' is not installed.\n"
                    f"  Please run: `ollama pull {required_model}`"
                )
            print_success(f"Model '{required_model}' is ready.")
            
            # 3. Test REST API generation
            await test_generation(client, required_model)
            
            print(f"\n{Colors.OKGREEN}{Colors.BOLD}✔ All checks passed! The local LLM environment is ready.{Colors.ENDC}\n")
            return True
            
        except OllamaCheckError as exc:
            print_error(str(exc))
            print(f"\n{Colors.FAIL}{Colors.BOLD}✖ Environment check failed. Please resolve the issues above.{Colors.ENDC}\n")
            return False
        except Exception as exc:
            print_error(f"An unexpected error occurred: {exc}")
            return False


def main() -> None:
    """Parse arguments and execute the async checks."""
    parser = argparse.ArgumentParser(description="Verify Ollama LLM setup.")
    parser.add_argument(
        "--url", 
        default=os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL),
        help="Base URL for Ollama server"
    )
    parser.add_argument(
        "--model", 
        default=os.getenv("OLLAMA_MODEL", DEFAULT_MODEL),
        help="Required model name"
    )
    
    args = parser.parse_args()
    
    try:
        success = asyncio.run(run_checks(url=args.url, required_model=args.model))
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nVerification cancelled by user.")
        sys.exit(1)


if __name__ == "__main__":
    main()

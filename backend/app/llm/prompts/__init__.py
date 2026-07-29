"""
PolicyIntel AI - Prompt Library
================================

This package exposes reusable prompt-builder functions for every use case
in PolicyIntel AI.  Business logic must NEVER hardcode prompt strings.

All functions return plain strings ready to be passed to the LLM client.

Modules
-------
system.py      : System-level persona and instruction templates
query.py       : Query understanding / intent classification prompts
generation.py  : Answer generation and summarisation prompts
verification.py: Fact-checking and answer verification prompts
extraction.py  : Structured data extraction prompts
comparison.py  : Policy comparison and diff prompts
"""

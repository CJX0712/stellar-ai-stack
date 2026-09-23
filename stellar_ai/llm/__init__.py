"""LLM layer.

Default: a deterministic, offline mock provider built on the extractive reader.
Production: an OpenAI-compatible provider that also talks to Ollama / vLLM /
llama-cpp servers. Selecting a real provider requires only environment
variables; the default install stays dependency-free.
"""

from .provider import MockLLM, OpenAIProvider, get_llm_provider

__all__ = ["MockLLM", "OpenAIProvider", "get_llm_provider"]

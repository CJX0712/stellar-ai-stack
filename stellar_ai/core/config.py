"""Configuration model with environment-variable overrides.

All tunables for the system live here. ``Config.from_env`` reads ``STELLAR_*``
environment variables so the same binary can run in mock mode (zero
dependency, offline) or wired to production backends without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    """System configuration.

    Attributes:
        embedding_dim: dimensionality of the default hash embedding.
        embedding_provider: ``hash`` (default) or a real model name.
        vectorstore_provider: ``memory`` (default) or ``faiss``.
        llm_provider: ``mock`` (default) or ``openai`` / ``ollama`` / ``llamacpp``.
        llm_model: model identifier passed to the LLM provider.
        llm_base_url: base URL for OpenAI-compatible / Ollama endpoints.
        llm_api_key: API key (never logged).
        retrieval_top_k: candidates pulled from each branch before fusion.
        rerank_top_n: final chunks returned after reranking.
        agent_max_steps: hard upper bound on ReAct iterations.
        log_level: logging verbosity.
    """

    embedding_dim: int = 256
    embedding_provider: str = "hash"
    vectorstore_provider: str = "memory"
    llm_provider: str = "mock"
    llm_model: str = "mock-1"
    llm_base_url: str = ""
    llm_api_key: str = ""
    retrieval_top_k: int = 20
    rerank_top_n: int = 5
    agent_max_steps: int = 6
    log_level: str = "INFO"

    @staticmethod
    def _int(value: str | None, default: int) -> int:
        if not value:
            return default
        try:
            return int(value)
        except ValueError:
            return default

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "Config":
        env = environ if environ is not None else os.environ
        return cls(
            embedding_dim=cls._int(env.get("STELLAR_EMBEDDING_DIM"), 256),
            embedding_provider=env.get("STELLAR_EMBEDDING_PROVIDER", "hash"),
            vectorstore_provider=env.get("STELLAR_VECTORSTORE_PROVIDER", "memory"),
            llm_provider=env.get("STELLAR_LLM_PROVIDER", "mock"),
            llm_model=env.get("STELLAR_LLM_MODEL", "mock-1"),
            llm_base_url=env.get("STELLAR_LLM_BASE_URL", ""),
            llm_api_key=env.get("STELLAR_LLM_API_KEY", ""),
            retrieval_top_k=cls._int(env.get("STELLAR_RETRIEVAL_TOP_K"), 20),
            rerank_top_n=cls._int(env.get("STELLAR_RERANK_TOP_N"), 5),
            agent_max_steps=cls._int(env.get("STELLAR_AGENT_MAX_STEPS"), 6),
            log_level=env.get("STELLAR_LOG_LEVEL", "INFO"),
        )

    def as_dict(self) -> dict[str, object]:
        redacted = dict(self.__dict__)
        if redacted.get("llm_api_key"):
            redacted["llm_api_key"] = "***"
        return redacted

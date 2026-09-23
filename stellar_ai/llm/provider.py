"""LLM providers: offline mock and OpenAI-compatible client.

The mock provider is a real generator (extractive), not a stub: given a RAG
prompt containing a ``<kb-context>`` block it returns the most relevant
sentence as a ``Final Answer:``. With no context it returns a neutral final
answer, which lets the agent fall back to a deterministic search step.
"""

from __future__ import annotations

import re

from ..core.config import Config
from ..core.errors import LLMError
from ..generation.reader import extractive_answer

_CONTEXT_RE = re.compile(r"<kb-context>(.*?)</kb-context>", re.S)
_QUESTION_RE = re.compile(r"Question:\s*(.*?)(?:\n|$)", re.I | re.S)


def _extract_context(prompt: str) -> str:
    m = _CONTEXT_RE.search(prompt)
    return m.group(1).strip() if m else ""


def _extract_question(prompt: str) -> str:
    stripped = _CONTEXT_RE.sub("", prompt)
    m = _QUESTION_RE.search(stripped)
    if m:
        return m.group(1).strip()
    tail = stripped.strip()
    return tail[-200:]


class MockLLM:
    """Deterministic, offline LLM used by default.

    When the prompt is a ReAct/agent prompt (no ``<kb-context>`` but tool
    instructions present) it returns a deterministic ``search`` action so the
    agent loop is exercised without any real model. When a context block is
    present it answers extractively. Otherwise it returns a neutral final answer.
    """

    def __init__(self, model: str = "mock-1") -> None:
        self.model = model

    def complete(self, prompt: str) -> str:
        context = _extract_context(prompt)
        question = _extract_question(prompt)
        if context:
            return "Final Answer: " + extractive_answer(question, context)
        if "Action Input:" in prompt or "Tools:" in prompt or "可用工具" in prompt:
            return f"Action: search\nAction Input: {question}"
        return "Final Answer: 暂无可用知识库信息，请先摄入相关文档。"


def _is_local(url: str) -> bool:
    return any(h in url for h in ("localhost", "127.0.0.1", "0.0.0.0"))


class OpenAIProvider:
    """OpenAI-compatible chat completion client (also works with Ollama/vLLM)."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "",
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def complete(self, prompt: str) -> str:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise LLMError("httpx is required for the OpenAI provider") from exc
        url = self.base_url
        if not url.endswith("/chat/completions"):
            url = url + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        # Local endpoints (e.g. Ollama) must bypass the system SOCKS proxy.
        trust_env = not _is_local(self.base_url)
        try:
            with httpx.Client(timeout=self.timeout, trust_env=trust_env) as client:
                resp = client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"openai completion failed: {exc}") from exc


def get_llm_provider(config: Config):
    """Return the LLM provider selected by configuration."""
    provider = config.llm_provider.lower()
    if provider in ("mock", "default"):
        return MockLLM(model=config.llm_model)
    if provider in ("openai", "ollama", "llamacpp", "vllm"):
        base_url = config.llm_base_url or "http://localhost:11434/v1"
        return OpenAIProvider(
            model=config.llm_model,
            base_url=base_url,
            api_key=config.llm_api_key,
        )
    raise LLMError(f"unknown llm provider: {provider}")

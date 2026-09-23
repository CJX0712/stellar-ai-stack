"""Composition root.

``build_system`` is the single place where concrete implementations are chosen
from configuration and injected into one another. Entry points (API, CLI) depend
only on this function, never on concrete classes - which is what keeps the
business surface tiny and the modules independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass

from .agent.agent import Agent
from .core.config import Config
from .core.logging import configure_logging
from .generation.rag import RAGPipeline
from .knowledge import KnowledgeBase
from .llm.provider import get_llm_provider


@dataclass
class System:
    config: Config
    knowledge: KnowledgeBase
    llm: object
    rag: RAGPipeline
    agent: Agent


def build_system(config: Config | None = None) -> System:
    config = config or Config.from_env()
    configure_logging(config.log_level)
    knowledge = KnowledgeBase(config)
    llm = get_llm_provider(config)
    rag = RAGPipeline(knowledge.retriever, llm, top_n=config.rerank_top_n)
    agent = Agent(knowledge.retriever, llm, max_steps=config.agent_max_steps)
    return System(
        config=config,
        knowledge=knowledge,
        llm=llm,
        rag=rag,
        agent=agent,
    )

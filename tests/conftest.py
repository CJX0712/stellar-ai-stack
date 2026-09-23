"""Shared pytest fixtures: a fresh System plus sample corpus and eval cases."""

from __future__ import annotations

import pytest

from stellar_ai.composition import build_system
from stellar_ai.core.config import Config

SAMPLE_STELLAR = (
    "# 恒星\n"
    "恒星是由引力凝聚在一起的球型发光等离子体。\n"
    "太阳是最接近地球的恒星，为地球提供光和热。\n"
    "恒星的主要成分是氢和氦，通过核聚变释放能量。\n\n"
    "# 生命周期\n"
    "恒星的生命周期取决于它的质量。\n"
    "质量越大的恒星寿命越短，最终可能以超新星爆发结束。\n"
)

SAMPLE_AI = (
    "# 人工智能\n"
    "人工智能是研究如何让机器模拟人类智能的科学。\n"
    "机器学习是人工智能的一个分支，依赖数据驱动的方法。\n"
    "深度学习使用多层神经网络，近年来推动了大模型的发展。\n"
)


@pytest.fixture
def config():
    return Config(
        embedding_dim=128,
        llm_provider="mock",
        embedding_provider="hash",
        vectorstore_provider="memory",
        retrieval_top_k=20,
        rerank_top_n=5,
        agent_max_steps=6,
    )


@pytest.fixture
def system(config):
    sys = build_system(config)
    sys.knowledge.ingest_text(SAMPLE_STELLAR, title="恒星", doc_id="d1")
    sys.knowledge.ingest_text(SAMPLE_AI, title="人工智能", doc_id="d2")
    return sys


@pytest.fixture
def knowledge(system):
    return system.knowledge


@pytest.fixture
def rag(system):
    return system.rag


@pytest.fixture
def agent(system):
    return system.agent


@pytest.fixture
def eval_cases():
    return [
        {
            "question": "恒星的主要成分是什么？",
            "relevant_doc_ids": ["d1"],
            "relevant_chunk_ids": [],
        },
        {
            "question": "什么是机器学习？",
            "relevant_doc_ids": ["d2"],
            "relevant_chunk_ids": [],
        },
        {
            "question": "深度学习使用了什么技术？",
            "relevant_doc_ids": ["d2"],
            "relevant_chunk_ids": [],
        },
    ]

"""Retrieval-augmented generation pipeline.

Builds a prompt that isolates the knowledge context behind a globally-unique
``<kb-context>`` delimiter (never used in the instructions, so the mock LLM can
parse it unambiguously), calls the LLM, and returns an Answer carrying
provenance citations and the raw contexts for downstream evaluation.
"""

from __future__ import annotations

from ..core.interfaces import LLMProvider, RAGPipeline as RAGPipelineProto, Retriever
from ..core.types import Answer, Citation


class RAGPipeline:
    def __init__(self, retriever: Retriever, llm: LLMProvider, top_n: int = 5) -> None:
        self.retriever = retriever
        self.llm = llm
        self.top_n = top_n

    @staticmethod
    def _render_context(retrieved) -> str:
        lines = []
        for i, rc in enumerate(retrieved):
            one_line = rc.chunk.text.replace("\n", " ")
            source = rc.chunk.heading or rc.chunk.doc_id
            lines.append(f"[id#{i}] (source: {source}) {one_line}")
        return "\n".join(lines)

    def _build_prompt(self, question: str, retrieved) -> str:
        context = self._render_context(retrieved)
        return (
            "你是一个严谨的问答助手。只能依据下方知识库内容作答，不得编造。\n"
            f"User question: {question}\n"
            "<kb-context>\n"
            f"{context}\n"
            "</kb-context>\n"
            "请直接给出最能回答问题的句子作为最终答案。\n"
            "Final Answer:"
        )

    @staticmethod
    def _parse(raw: str) -> str:
        if "Final Answer:" in raw:
            return raw.split("Final Answer:", 1)[1].strip()
        return raw.strip()

    def answer(self, question: str, top_n: int | None = None) -> Answer:
        top_n = top_n if top_n is not None else self.top_n
        retrieved = self.retriever.retrieve(question, top_n)
        prompt = self._build_prompt(question, retrieved)
        raw = self.llm.complete(prompt)
        answer_text = self._parse(raw)
        citations = [
            Citation(
                doc_id=rc.chunk.doc_id,
                chunk_id=rc.chunk.id,
                snippet=rc.chunk.text[:160],
                score=round(rc.score, 6),
            )
            for rc in retrieved
        ]
        # Contexts are collapsed to a single line so downstream consumers
        # (and the evaluation harness) never see split lines.
        contexts = [
            rc.chunk.text.replace("\n", " ").replace("\r", " ").strip()
            for rc in retrieved
        ]
        return Answer(
            question=question,
            answer=answer_text,
            citations=citations,
            contexts=contexts,
        )

"""Evaluation harness.

Runs a fixed set of query/ground-truth cases through a RAG pipeline and reports
a scorecard. It reads results EXPLICITLY from each returned Answer (never from a
pipeline cache), so the measured retrieval is the retrieval that actually served
the answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.interfaces import RAGPipeline as RAGPipelineProto
from .metrics import doc_hit_rate, doc_mrr, grounding_overlap, token_recall


@dataclass
class Scorecard:
    cases: int = 0
    doc_hit_rate: float = 0.0
    doc_mrr: float = 0.0
    mean_grounding: float = 0.0
    chunk_recall: float = 0.0
    details: list[dict] = field(default_factory=list)


class EvalHarness:
    def __init__(self, rag: RAGPipelineProto, top_n: int = 5) -> None:
        self.rag = rag
        self.top_n = top_n

    def run(self, cases: list[dict]) -> Scorecard:
        rows: list[dict] = []
        hit_sum = 0.0
        mrr_sum = 0.0
        ground_sum = 0.0
        recall_sum = 0.0
        retrieval_cases = 0
        for case in cases:
            question = case["question"]
            relevant_docs = set(case["relevant_doc_ids"])
            relevant_chunks = set(case.get("relevant_chunk_ids", []))
            answer = self.rag.answer(question, top_n=self.top_n)

            cited_doc_ids = [c.doc_id for c in answer.citations]
            cited_chunk_ids = {c.chunk_id for c in answer.citations}

            hit = doc_hit_rate(cited_doc_ids, relevant_docs)
            mrr = doc_mrr(cited_doc_ids, relevant_docs)
            recall = token_recall(cited_chunk_ids, relevant_chunks)
            ground = (
                max((grounding_overlap(answer.answer, c.snippet) for c in answer.citations), default=0.0)
                if answer.citations
                else 0.0
            )
            # Retrieval-only cases (e.g. arithmetic routing) are excluded from
            # retrieval aggregates so they don't drag the average down.
            if case.get("counts_as_retrieval", True):
                hit_sum += hit
                mrr_sum += mrr
                ground_sum += ground
                recall_sum += recall
                retrieval_cases += 1

            rows.append(
                {
                    "question": question,
                    "cited_doc_ids": cited_doc_ids,
                    "hit": hit,
                    "mrr": round(mrr, 4),
                    "chunk_recall": round(recall, 4),
                    "grounding": round(ground, 4),
                    "answer": answer.answer,
                }
            )

        n = max(1, retrieval_cases)
        return Scorecard(
            cases=len(cases),
            doc_hit_rate=round(hit_sum / n, 4),
            doc_mrr=round(mrr_sum / n, 4),
            mean_grounding=round(ground_sum / n, 4),
            chunk_recall=round(recall_sum / n, 4),
            details=rows,
        )

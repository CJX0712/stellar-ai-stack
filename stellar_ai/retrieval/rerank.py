"""TF-IDF cosine reranker.

Takes the fused candidate set and re-scores it with a precise TF-IDF cosine
between the query and each chunk. This is the step that fixes ranking quality
after the coarse RRF fusion, and it is what the evaluation harness measures.
"""

from __future__ import annotations

import math

from ..embedding.provider import tokenize
from .lexical import LexicalIndex


class Reranker:
    def __init__(self, index: LexicalIndex) -> None:
        self.index = index

    def rerank(self, query: str, chunk_ids: list[str], top_n: int) -> list[tuple[str, float]]:
        q_tf = {}
        for term in tokenize(query):
            idf = self.index.idf(term)
            if idf <= 0.0:
                continue
            q_tf[term] = q_tf.get(term, 0.0) + idf
        q_norm = math.sqrt(sum(v * v for v in q_tf.values())) or 1.0

        results: list[tuple[str, float]] = []
        for cid in chunk_ids:
            c_tf = self.index.chunk_tf.get(cid)
            if not c_tf:
                continue
            dot = 0.0
            c_sq = 0.0
            for term, tf in c_tf.items():
                w = tf * self.index.idf(term)
                c_sq += w * w
                if term in q_tf:
                    dot += q_tf[term] * w
            c_norm = math.sqrt(c_sq) or 1.0
            denom = q_norm * c_norm
            score = dot / denom if denom > 0.0 else 0.0
            results.append((cid, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]

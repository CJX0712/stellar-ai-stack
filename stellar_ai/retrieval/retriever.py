"""Hybrid retriever and chunk registry.

The retriever fuses a dense (vector) branch and a sparse (lexical) branch with
Reciprocal Rank Fusion, then re-ranks the fused candidates with TF-IDF cosine.
The ChunkRegistry maps chunk ids back to chunk objects for the final answer.
"""

from __future__ import annotations

from ..core.interfaces import EmbeddingProvider, Retriever, VectorStore
from ..core.types import Chunk, RetrievedChunk
from .lexical import LexicalIndex
from .rerank import Reranker


class ChunkRegistry:
    """Maps chunk ids to Chunk objects and tracks document membership."""

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}

    def put(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            self._chunks[c.id] = c

    def drop(self, doc_id: str) -> None:
        stale = [cid for cid, c in self._chunks.items() if c.doc_id == doc_id]
        for cid in stale:
            self._chunks.pop(cid, None)

    def get(self, ids: list[str]) -> list[Chunk]:
        return [self._chunks[i] for i in ids if i in self._chunks]

    def all(self) -> list[Chunk]:
        return list(self._chunks.values())

    def count(self) -> int:
        return len(self._chunks)


def reciprocal_rank_fusion(ranked_lists: list[list[tuple[str, float]]], k: int = 60) -> dict[str, float]:
    """Fuse multiple ranked lists into a single score map."""
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (cid, _score) in enumerate(ranked):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return fused


class HybridRetriever:
    """Dense + sparse hybrid retrieval with RRF fusion and TF-IDF rerank."""

    def __init__(
        self,
        registry: ChunkRegistry,
        vector_store: VectorStore,
        embedding: EmbeddingProvider,
        lexical_index: LexicalIndex,
        reranker: Reranker,
        top_k: int = 20,
        top_n: int = 5,
    ) -> None:
        self.registry = registry
        self.vector_store = vector_store
        self.embedding = embedding
        self.lexical_index = lexical_index
        self.reranker = reranker
        self.top_k = top_k
        self.top_n = top_n

    def retrieve(self, query: str, top_n: int | None = None) -> list[RetrievedChunk]:
        top_n = top_n if top_n is not None else self.top_n
        qvec = self.embedding.embed([query])[0]
        vec_hits = self.vector_store.query(qvec, self.top_k)
        lex_hits = self.lexical_index.search(query, self.top_k)
        fused = reciprocal_rank_fusion([vec_hits, lex_hits])
        cand_ids = [cid for cid, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)]
        reranked = self.reranker.rerank(query, cand_ids, top_n)
        chunks = self.registry.get([cid for cid, _ in reranked])
        return [
            RetrievedChunk(chunk=chunk, score=score)
            for (_, score), chunk in zip(reranked, chunks)
        ]

"""Retrieval layer.

Combines two complementary signals: a lexical (sparse, IDF-weighted) branch and
a dense (vector) branch, fused with Reciprocal Rank Fusion, then re-ranked with a
precise TF-IDF cosine. The hybrid design is what keeps retrieval working for both
keyword and semantic queries without any GPU or external service.
"""

from .lexical import LexicalIndex
from .rerank import Reranker
from .retriever import ChunkRegistry, HybridRetriever

__all__ = ["LexicalIndex", "Reranker", "ChunkRegistry", "HybridRetriever"]

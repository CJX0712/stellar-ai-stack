"""In-memory exact-cosine vector store and provider selection.

Vectors are L2-normalised on write, so cosine similarity is a dot product.
``query`` returns the top-``k`` (chunk_id, score) pairs sorted by descending
score. ``drop_document`` removes every chunk belonging to a document, which is
what makes re-ingestion of a (possibly shorter) document safe: old orphan
chunks cannot linger and be retrieved.
"""

from __future__ import annotations

import math

from ..core.config import Config
from ..core.errors import VectorStoreError
from ..core.types import Chunk


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise VectorStoreError("dimension mismatch in cosine")
    s = 0.0
    for x, y in zip(a, b):
        s += x * y
    return s


class InMemoryVectorStore:
    """Exact nearest-neighbour store backed by a dict."""

    def __init__(self) -> None:
        self._vecs: dict[str, list[float]] = {}
        self._doc_of: dict[str, str] = {}
        self._dim: int | None = None

    def put_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise VectorStoreError("chunks/embeddings length mismatch")
        for chunk, vec in zip(chunks, embeddings):
            if self._dim is None:
                self._dim = len(vec)
            if len(vec) != self._dim:
                raise VectorStoreError(
                    f"dimension mismatch: expected {self._dim}, got {len(vec)}"
                )
            mag = math.sqrt(sum(v * v for v in vec))
            norm = [v / mag for v in vec] if mag > 0.0 else [0.0] * len(vec)
            self._vecs[chunk.id] = norm
            self._doc_of[chunk.id] = chunk.doc_id

    def drop_document(self, doc_id: str) -> None:
        stale = [cid for cid, d in self._doc_of.items() if d == doc_id]
        for cid in stale:
            self._vecs.pop(cid, None)
            self._doc_of.pop(cid, None)

    def query(self, vector: list[float], k: int) -> list[tuple[str, float]]:
        if self._dim is not None and len(vector) != self._dim:
            raise VectorStoreError(
                f"query dimension {len(vector)} != stored {self._dim}"
            )
        scored = [(cid, cosine(vector, vec)) for cid, vec in self._vecs.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def count(self) -> int:
        return len(self._vecs)


def get_vector_store(config: Config):
    """Return the vector store selected by configuration."""
    provider = config.vectorstore_provider.lower()
    if provider in ("memory", "default"):
        return InMemoryVectorStore()
    if provider == "faiss":
        try:
            from .faiss_store import FaissVectorStore  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise VectorStoreError(
                "faiss backend requires faiss-cpu. "
                "Run: pip install -r optional-requirements.txt"
            ) from exc
        return FaissVectorStore(dim=config.embedding_dim)
    raise VectorStoreError(f"unknown vector store provider: {provider}")

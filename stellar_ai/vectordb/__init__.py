"""Vector store layer.

Default: an exact-cosine in-memory store (pure Python, deterministic). An
optional FAISS backend is selectable via configuration for large corpora.
"""

from .store import InMemoryVectorStore, get_vector_store

__all__ = ["InMemoryVectorStore", "get_vector_store"]

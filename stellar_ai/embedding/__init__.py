"""Embedding layer.

The default provider is a deterministic, zero-dependency hash embedding based on
BLAKE2b signed hashing. It is dimension-stable and reproducibility-safe: the same
text always maps to the same unit vector, which is exactly what makes the
offline test suite deterministic. Production embeddings (fastembed /
sentence-transformers) are selectable via configuration.
"""

from .provider import HashEmbedding, get_embedding_provider

__all__ = ["HashEmbedding", "get_embedding_provider"]

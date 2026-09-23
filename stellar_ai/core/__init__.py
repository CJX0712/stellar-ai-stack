"""Core layer: configuration, logging, types, errors and interface contracts.

Every external port of Stellar AI is defined here as a Protocol so that modules
depend on abstractions, not implementations. Concrete implementations are
injected at composition time, which makes each module independently testable.
"""

from .config import Config
from .errors import (
    AgentError,
    APIError,
    ConfigError,
    EmbeddingError,
    GenerationError,
    IngestionError,
    LLMError,
    RetrievalError,
    StellarError,
    VectorStoreError,
)
from .interfaces import (
    Agent,
    Chunker,
    DocumentLoader,
    EmbeddingProvider,
    LLMProvider,
    RAGPipeline,
    Retriever,
    VectorStore,
)
from .types import Answer, Chunk, Citation, Document, RetrievedChunk

__all__ = [
    "Config",
    "StellarError",
    "ConfigError",
    "IngestionError",
    "EmbeddingError",
    "VectorStoreError",
    "RetrievalError",
    "LLMError",
    "GenerationError",
    "AgentError",
    "APIError",
    "Document",
    "Chunk",
    "RetrievedChunk",
    "Citation",
    "Answer",
    "DocumentLoader",
    "Chunker",
    "EmbeddingProvider",
    "VectorStore",
    "Retriever",
    "LLMProvider",
    "RAGPipeline",
    "Agent",
]

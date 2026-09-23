"""Exception hierarchy for Stellar AI.

Each layer raises its own subtype of StellarError so callers (API, CLI, agent)
can map failures to the correct response code or recovery strategy.
"""


class StellarError(Exception):
    """Base class for all Stellar AI errors."""


class ConfigError(StellarError):
    """Raised when configuration is missing or invalid."""


class IngestionError(StellarError):
    """Raised when a document cannot be loaded or chunked."""


class EmbeddingError(StellarError):
    """Raised when embedding generation fails."""


class VectorStoreError(StellarError):
    """Raised when the vector store cannot store or query."""


class RetrievalError(StellarError):
    """Raised when retrieval fails."""


class LLMError(StellarError):
    """Raised when the LLM provider cannot complete a prompt."""


class GenerationError(StellarError):
    """Raised when the RAG generation pipeline fails."""


class AgentError(StellarError):
    """Raised when the agent loop fails or exceeds its step budget."""


class APIError(StellarError):
    """Raised by the API layer for user-facing errors."""

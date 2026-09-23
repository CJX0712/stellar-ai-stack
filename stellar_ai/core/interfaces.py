"""Interface contracts (Protocols) for every Stellar AI module.

Modules depend only on these abstractions. At composition time (see
``stellar_ai.composition.build_system``) concrete implementations are selected
from configuration and injected. This is what lets each module be unit-tested
in isolation with a fake implementation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import Answer, Chunk, Document, RetrievedChunk


@runtime_checkable
class DocumentLoader(Protocol):
    """Loads a source path into one or more Documents."""

    def load(self, source: str) -> list[Document]:
        ...


@runtime_checkable
class Chunker(Protocol):
    """Splits a Document into self-contained Chunks."""

    def chunk(self, doc: Document) -> list[Chunk]:
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns texts into fixed-dimensional vectors."""

    @property
    def dim(self) -> int:
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Stores chunk vectors and answers nearest-neighbour queries."""

    def put_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        ...

    def drop_document(self, doc_id: str) -> None:
        ...

    def query(self, vector: list[float], k: int) -> list[tuple[str, float]]:
        """Return up to ``k`` (chunk_id, cosine_similarity) pairs, best first."""
        ...

    def count(self) -> int:
        ...


@runtime_checkable
class Retriever(Protocol):
    """Retrieves the most relevant chunks for a natural-language query."""

    def retrieve(self, query: str, top_n: int | None = None) -> list[RetrievedChunk]:
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """Completes a prompt into text."""

    def complete(self, prompt: str) -> str:
        ...


@runtime_checkable
class RAGPipeline(Protocol):
    """Answers a question using retrieval-augmented generation."""

    def answer(self, question: str, top_n: int | None = None) -> Answer:
        ...


@runtime_checkable
class Agent(Protocol):
    """Runs a multi-step ReAct loop to accomplish a task."""

    def run(self, task: str) -> str:
        ...

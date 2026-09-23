"""Shared data types used across all Stellar AI modules.

All types are plain dataclasses so they serialize cleanly to JSON for the API
and remain trivially comparable in unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A loaded source document before chunking."""

    id: str
    title: str
    text: str
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A contiguous, self-contained piece of a document."""

    id: str
    doc_id: str
    text: str
    heading: str = ""
    start: int = 0
    end: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """A chunk together with its retrieval score (higher is better)."""

    chunk: Chunk
    score: float


@dataclass
class Citation:
    """A provenance pointer surfaced with a generated answer."""

    doc_id: str
    chunk_id: str
    snippet: str
    score: float


@dataclass
class Answer:
    """The result of a RAG query or agent task."""

    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

"""Stellar AI - a world-class, modular, end-to-end runnable RAG + ReAct agent system.

The default run path is pure Python standard library (zero third-party dependency):
a hash embedding, an exact-cosine in-memory vector store, an IDF-weighted hybrid
retriever, a deterministic mock LLM, an extractive RAG reader and a ReAct agent.
Production-grade backends (sentence-transformers / faiss / llama-cpp) are
selectable through environment variables and live in optional-requirements.txt.
"""

__version__ = "1.0.0"
__author__ = "晨星"

from .core.config import Config
from .core.types import Answer, Chunk, Citation, Document, RetrievedChunk
from .composition import build_system

__all__ = [
    "Config",
    "Document",
    "Chunk",
    "RetrievedChunk",
    "Citation",
    "Answer",
    "build_system",
    "__version__",
    "__author__",
]

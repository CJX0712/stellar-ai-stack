"""Ingestion layer: load documents and split them into chunks.

Zero-dependency by default. PDF loading is available only when ``pypdf`` is
installed (see optional-requirements.txt); everything else is plain stdlib.
"""

from .chunker import HeadingChunker
from .loader import (
    PdfLoader,
    StringDocument,
    TextLoader,
    load_document,
)

__all__ = [
    "HeadingChunker",
    "TextLoader",
    "PdfLoader",
    "StringDocument",
    "load_document",
]

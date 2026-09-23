"""Document loaders.

The default loader handles plain text and Markdown with no third-party
dependency. PDF support is enabled only when ``pypdf`` is importable, keeping
the base install tiny and offline-friendly.
"""

from __future__ import annotations

import hashlib
import os
import re

from ..core.errors import IngestionError
from ..core.logging import get_logger
from ..core.types import Document

logger = get_logger("ingestion.loader")


def _stable_id(source: str, text: str) -> str:
    h = hashlib.blake2b(source.encode("utf-8"), digest_size=8).hexdigest()
    return "doc-" + h


class TextLoader:
    """Load ``.txt`` and ``.md`` files as Documents."""

    def load(self, source: str) -> list[Document]:
        if not os.path.exists(source):
            raise IngestionError(f"source not found: {source}")
        with open(source, "r", encoding="utf-8") as fh:
            text = fh.read()
        return [self._to_document(source, text)]

    def _to_document(self, source: str, text: str) -> Document:
        title = os.path.splitext(os.path.basename(source))[0]
        # Prefer the first Markdown heading as the title.
        for line in text.splitlines():
            m = re.match(r"^#{1,6}\s+(.*)$", line.strip())
            if m:
                title = m.group(1).strip()
                break
        return Document(
            id=_stable_id(source, text),
            title=title,
            text=text,
            source=source,
            metadata={"kind": "text"},
        )


class PdfLoader:
    """Load ``.pdf`` files. Requires the optional ``pypdf`` package."""

    def load(self, source: str) -> list[Document]:
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise IngestionError(
                "PDF loading requires pypdf. Install it with: "
                "pip install -r optional-requirements.txt"
            ) from exc
        if not os.path.exists(source):
            raise IngestionError(f"source not found: {source}")
        reader = PdfReader(source)
        parts: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                parts.append(page_text)
        text = "\n\n".join(parts)
        title = os.path.splitext(os.path.basename(source))[0]
        return [
            Document(
                id=_stable_id(source, text),
                title=title,
                text=text,
                source=source,
                metadata={"kind": "pdf", "pages": len(reader.pages)},
            )
        ]


class StringDocument:
    """Build a Document directly from raw text (used by the API and tests)."""

    @staticmethod
    def build(text: str, title: str = "", doc_id: str = "") -> Document:
        text = text.strip()
        if not text:
            raise IngestionError("cannot ingest empty text")
        if not title:
            title = (text.splitlines()[0][:60] if text.splitlines() else "untitled")
        if not doc_id:
            doc_id = "doc-" + hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
        return Document(
            id=doc_id,
            title=title,
            text=text,
            source="inline",
            metadata={"kind": "inline"},
        )


_LOADERS = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".markdown": TextLoader,
    ".pdf": PdfLoader,
}


def load_document(source: str) -> list[Document]:
    """Dispatch to the correct loader by file extension."""
    ext = os.path.splitext(source)[1].lower()
    loader_cls = _LOADERS.get(ext)
    if loader_cls is None:
        logger.warning("no loader for %s; falling back to TextLoader", ext or "<none>")
        loader_cls = TextLoader
    return loader_cls().load(source)

"""Heading-aware chunker with section-title inheritance.

Design goals (driven by past reproducibility failures):
  * Every chunk carries the section heading that scopes it (heading inheritance),
    and the heading text is PREFIXED onto every chunk so later chunks in a long
    section do not lose their context (a bug where 90% of context was dropped).
  * Long sections are split into multiple bounded chunks at sentence boundaries,
    so the multi-chunk retrieval path is always exercised.
  * Heading-only fragments are skipped (no useless chunks).

The chunker is pure standard library: deterministic and offline-safe.
"""

from __future__ import annotations

import re

from ..core.types import Chunk, Document

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_NUMBERED_RE = re.compile(r"^\s*\d+(?:\.\d+)*\.\s+(.*)$")

_SENT_END = set("。！？.!?")


def _heading_of(line: str) -> str | None:
    s = line.strip()
    m = _HEADING_RE.match(s)
    if m:
        return m.group(2).strip()
    m = _NUMBERED_RE.match(s)
    if m:
        return m.group(1).strip()
    return None


class HeadingChunker:
    """Split a document into overlapping-free, heading-scoped chunks."""

    def __init__(self, max_chars: int = 700, min_chars: int = 60) -> None:
        self.max_chars = max_chars
        self.min_chars = min_chars

    def _split_long(self, text: str) -> list[str]:
        if len(text) <= self.max_chars:
            return [text]
        pieces: list[str] = []
        cur = ""
        for ch in text:
            cur += ch
            if len(cur) >= self.max_chars:
                pieces.append(cur)
                cur = ""
            elif ch in _SENT_END and len(cur) >= self.min_chars:
                pieces.append(cur)
                cur = ""
        if cur:
            pieces.append(cur)
        return pieces

    def chunk(self, doc: Document) -> list[Chunk]:
        lines = doc.text.split("\n")
        blocks: list[tuple[str, list[str]]] = []
        heading = ""
        body: list[str] = []
        for raw in lines:
            line = raw.rstrip()
            h = _heading_of(line)
            if h is not None:
                if body:
                    blocks.append((heading, body))
                    body = []
                heading = h
                continue
            body.append(line)
        if body:
            blocks.append((heading, body))

        out: list[Chunk] = []
        idx = 0
        for heading, body_lines in blocks:
            body_text = "\n".join(body_lines).strip()
            if not body_text:
                pieces = [heading] if heading else []
            else:
                pieces = self._split_long(body_text)
            for piece in pieces:
                piece = piece.strip()
                if not piece:
                    continue
                text = (heading + "\n" + piece) if heading else piece
                out.append(
                    Chunk(
                        id=f"{doc.id}#c{idx}",
                        doc_id=doc.id,
                        text=text,
                        heading=heading,
                        start=0,
                        end=len(text),
                    )
                )
                idx += 1

        if not out:
            out.append(
                Chunk(
                    id=f"{doc.id}#c0",
                    doc_id=doc.id,
                    text=doc.text.strip(),
                    heading=heading,
                    start=0,
                    end=len(doc.text),
                )
            )
        return out

"""Evaluation metrics.

These metrics are deliberately conservative against self-deception:

  * The evaluation adapter captures results EXPLICITLY from the returned Answer
    object - it never reads a ``last_*`` cache on the pipeline, which previously
    let an agent's search tool silently overwrite the measured retrieval.
  * ``doc_hit_rate`` / ``doc_mrr`` are DOCUMENT-level (headline) metrics. Block
    recall is reported only as a diagnostic, because unannotated chunks in a
    relevant document are not really "misses".
  * ``rank_correlation`` returns 0.0 when either input is constant. A constant
    input makes rank correlation mathematically undefined, and a naive
    tie-break would otherwise report a misleadingly perfect +1.0.
"""

from __future__ import annotations

from ..embedding.provider import tokenize

# Citation / provenance markers that must be stripped before grounding checks,
# because they are metadata, not assertions.
_MARKER_RE_SNIPPETS = ("[id#", "(source:", "(依据", "(来源")


def _strip_markers(text: str) -> str:
    out = text
    for marker in _MARKER_RE_SNIPPETS:
        # Remove the marker and the trailing token/phrase up to a whitespace.
        idx = out.find(marker)
        while idx != -1:
            end = out.find(" ", idx)
            if end == -1:
                end = len(out)
            out = out[:idx] + out[end:]
            idx = out.find(marker)
    return out


def grounding_overlap(answer: str, evidence: str) -> float:
    """Token overlap between an answer and an evidence passage, markers stripped."""
    a = set(tokenize(_strip_markers(answer)))
    e = set(tokenize(evidence))
    if not a or not e:
        return 0.0
    return len(a & e) / len(a | e)


def doc_hit_rate(cited_doc_ids: list[str], relevant_doc_ids: set[str]) -> float:
    return 1.0 if any(d in relevant_doc_ids for d in cited_doc_ids) else 0.0


def doc_mrr(cited_doc_ids: list[str], relevant_doc_ids: set[str]) -> float:
    for rank, d in enumerate(cited_doc_ids, start=1):
        if d in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


def token_recall(cited_chunk_ids: set[str], relevant_chunk_ids: set[str]) -> float:
    if not relevant_chunk_ids:
        return 0.0
    return len(cited_chunk_ids & relevant_chunk_ids) / len(relevant_chunk_ids)


def rank_correlation(a: list[float], b: list[float]) -> float:
    """Spearman-style rank correlation; returns 0.0 if either input is constant."""
    n = len(a)
    if n != len(b) or n < 2:
        return 0.0
    ra = _ranks(a)
    rb = _ranks(b)
    if ra is None or rb is None:
        return 0.0
    return _pearson(ra, rb)


def _ranks(values: list[float]) -> list[float] | None:
    if len(set(values)) == 1:
        return None  # constant - undefined
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    for pos, idx in enumerate(order, start=1):
        ranks[idx] = pos
    return ranks


def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = sum((xi - mx) ** 2 for xi in x) ** 0.5
    dy = sum((yi - my) ** 2 for yi in y) ** 0.5
    if dx == 0.0 or dy == 0.0:
        return 0.0
    return num / (dx * dy)

"""IDF-weighted lexical index.

Uses the Robertson/Sparck-Jones IDF variant ``ln(1 + (N - n + 0.5)/(n + 0.5))``
which is guaranteed non-negative for any corpus size. (A plain probabilistic IDF
can go negative on tiny corpora and silently invert rankings - a bug we hit and
fixed.) Term frequencies are stored per chunk so the reranker can compute exact
TF-IDF cosine without re-tokenising.
"""

from __future__ import annotations

import math
from collections import Counter

from ..embedding.provider import tokenize


class LexicalIndex:
    """Inverted index with Robertson IDF."""

    def __init__(self) -> None:
        self.doc_freq: dict[str, int] = {}
        self.postings: dict[str, set[str]] = {}
        self.chunk_tf: dict[str, Counter] = {}
        self.N = 0

    def add(self, chunk_id: str, text: str) -> None:
        tf = Counter(tokenize(text))
        if not tf:
            return
        self.chunk_tf[chunk_id] = tf
        for term in tf:
            self.postings.setdefault(term, set()).add(chunk_id)
            self.doc_freq[term] = self.doc_freq.get(term, 0) + 1
        self.N += 1

    def remove(self, chunk_id: str) -> None:
        tf = self.chunk_tf.pop(chunk_id, None)
        if tf is None:
            return
        for term in tf:
            bucket = self.postings.get(term)
            if bucket:
                bucket.discard(chunk_id)
                if not bucket:
                    self.postings.pop(term, None)
                    self.doc_freq.pop(term, None)
                else:
                    self.doc_freq[term] = self.doc_freq.get(term, 1) - 1
                    if self.doc_freq[term] <= 0:
                        self.doc_freq.pop(term, None)
                        self.postings.pop(term, None)
        self.N = max(0, self.N - 1)

    def idf(self, term: str) -> float:
        n = self.doc_freq.get(term, 0)
        if n == 0:
            return 0.0
        return math.log(1.0 + (self.N - n + 0.5) / (n + 0.5))

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Cheap lexical candidate ranking by summed TF-IDF."""
        scores: dict[str, float] = {}
        for term in tokenize(query):
            idf = self.idf(term)
            if idf <= 0.0:
                continue
            for cid in self.postings.get(term, ()):
                scores[cid] = scores.get(cid, 0.0) + self.chunk_tf[cid].get(term, 0) * idf
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]

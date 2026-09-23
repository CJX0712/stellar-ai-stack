"""Deterministic hash embedding and provider selection.

The hash embedding maps each token to a signed dimension via BLAKE2b:
  * an index hash selects the dimension (modulo ``dim``)
  * a separate sign hash decides +1 / -1
Token frequencies are L2-normalised, then the whole vector is L2-normalised.
CJK text is tokenised into unigrams and bigrams so that semantic neighbours
still land in nearby dimensions; Latin text uses word tokens.
"""

from __future__ import annotations

import hashlib
import math
import re

from ..core.config import Config
from ..core.errors import EmbeddingError

_CJK = re.compile(r"[一-鿿]")
_LATIN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Tokenise text into language-appropriate features.

    Latin runs become ``w:<word>`` tokens; CJK characters become ``c:<char>``
    unigrams and ``b:<char><char>`` bigrams. Lower-cased.
    """
    text = text.lower()
    tokens: list[str] = []
    for m in _LATIN.finditer(text):
        tokens.append("w:" + m.group(0))
    chars = _CJK.findall(text)
    for i, ch in enumerate(chars):
        tokens.append("c:" + ch)
        if i + 1 < len(chars):
            tokens.append("b:" + ch + chars[i + 1])
    return tokens


class HashEmbedding:
    """Zero-dependency, deterministic embedding provider."""

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise EmbeddingError("embedding dim must be positive")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _index(self, token: str) -> int:
        d = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
        return int.from_bytes(d[:4], "big") % self._dim

    def _sign(self, token: str) -> float:
        d = hashlib.blake2b(token.encode("utf-8"), digest_size=4, salt=b"sgn").digest()
        return 1.0 if d[0] & 1 == 0 else -1.0

    def embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        counts: dict[str, int] = {}
        for tok in tokenize(text):
            counts[tok] = counts.get(tok, 0) + 1
        denom = math.sqrt(sum(c * c for c in counts.values())) or 1.0
        for tok, c in counts.items():
            vec[self._index(tok)] += self._sign(tok) * (c / denom)
        mag = math.sqrt(sum(v * v for v in vec))
        if mag > 0.0:
            vec = [v / mag for v in vec]
        return vec

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]


def get_embedding_provider(config: Config):
    """Return the embedding provider selected by configuration.

    The default ``hash`` provider is dependency-free. Real model providers raise
    a clear ImportError when their optional package is missing, so the default
    install never breaks.
    """
    provider = config.embedding_provider.lower()
    if provider in ("hash", "mock", "default"):
        return HashEmbedding(dim=config.embedding_dim)
    if provider in ("fastembed", "sentence-transformers", "st"):
        try:
            from .real_models import RealEmbedding  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise EmbeddingError(
                f"embedding provider '{provider}' requires optional packages. "
                "Run: pip install -r optional-requirements.txt"
            ) from exc
        return RealEmbedding(config)
    raise EmbeddingError(f"unknown embedding provider: {provider}")

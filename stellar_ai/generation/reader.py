"""Extractive reader.

Picks the sentence from a context that best matches a question using a
length-insensitive word-set cosine. This is a genuine (if simple) generator used
by the default mock LLM so that ``git clone && verify`` produces real answers
with zero dependencies and zero network.
"""

from __future__ import annotations

import math
import re

from ..embedding.provider import tokenize

_SPLIT_RE = re.compile(r"(?<=[。！？.!?])\s*")


def split_sentences(text: str) -> list[str]:
    parts = _SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _word_set(text: str) -> set[str]:
    return set(tokenize(text))


def _overlap(q_tokens: set[str], s_tokens: set[str]) -> float:
    if not q_tokens or not s_tokens:
        return 0.0
    inter = q_tokens & s_tokens
    return len(inter) / math.sqrt(len(q_tokens) * len(s_tokens))


def best_sentence(question: str, context: str) -> str:
    q = _word_set(question)
    sentences = split_sentences(context)
    if not sentences:
        return ""
    best, best_score = "", -1.0
    for s in sentences:
        score = _overlap(q, _word_set(s))
        if score > best_score:
            best_score, best = score, s
    return best


def extractive_answer(question: str, context: str) -> str:
    answer = best_sentence(question, context)
    if not answer:
        sentences = split_sentences(context)
        answer = sentences[0] if sentences else context.strip()[:200]
    return answer

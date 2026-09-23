"""Evaluation layer."""

from .harness import EvalHarness, Scorecard
from .metrics import (
    doc_hit_rate,
    doc_mrr,
    grounding_overlap,
    rank_correlation,
    token_recall,
)

__all__ = [
    "EvalHarness",
    "Scorecard",
    "doc_hit_rate",
    "doc_mrr",
    "token_recall",
    "grounding_overlap",
    "rank_correlation",
]

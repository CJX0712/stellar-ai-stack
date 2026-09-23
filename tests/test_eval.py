from stellar_ai.eval.harness import EvalHarness
from stellar_ai.eval.metrics import (
    doc_hit_rate,
    doc_mrr,
    grounding_overlap,
    rank_correlation,
)


def test_rank_correlation_constant_returns_zero():
    assert rank_correlation([1.0, 1.0, 1.0], [3.0, 1.0, 2.0]) == 0.0


def test_rank_correlation_normal():
    r = rank_correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
    assert -1.0 <= r <= 1.0


def test_doc_hit_and_mrr():
    assert doc_hit_rate(["d1", "d2"], {"d1"}) == 1.0
    assert doc_hit_rate(["d2"], {"d1"}) == 0.0
    assert doc_mrr(["x", "d1"], {"d1"}) == 0.5


def test_grounding_strips_markers():
    # A citation marker must not be counted as a shared token.
    ans = "[id#0] (source: 恒星) 氢和氦"
    ev = "恒星的主要成分是氢和氦"
    score = grounding_overlap(ans, ev)
    assert score > 0.0


def test_harness_on_corpus(rag, eval_cases):
    harness = EvalHarness(rag, top_n=5)
    sc = harness.run(eval_cases)
    assert sc.cases == len(eval_cases)
    # Retrieval consistently surfaces the relevant document.
    assert sc.doc_hit_rate >= 1.0
    assert sc.doc_mrr >= 1.0
    assert sc.mean_grounding > 0.0

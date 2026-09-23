import math

from stellar_ai.retrieval.lexical import LexicalIndex
from stellar_ai.retrieval.rerank import Reranker


def test_idf_non_negative_on_tiny_corpus():
    from stellar_ai.embedding.provider import tokenize

    idx = LexicalIndex()
    idx.add("c1", "恒星 hydrogen 聚变")
    idx.add("c2", "恒星 oxygen 寿命")
    # '恒星' (as the c:恒 / b:恒星 bigrams) appears in both docs.
    assert idx.idf("b:恒星") > 0.0
    # A term in only one doc still yields a positive idf.
    hydrogen_tok = next(t for t in tokenize("hydrogen") if t.startswith("w:"))
    assert idx.idf(hydrogen_tok) > 0.0
    # Robertson IDF is guaranteed non-negative even for a term in every doc.
    assert idx.idf("b:恒星") >= 0.0


def test_lexical_search_ranks_relevant_first():
    idx = LexicalIndex()
    idx.add("c1", "恒星 主要由 氢 和 氦 组成")
    idx.add("c2", "篮球 比赛 投篮")
    ranked = idx.search("氢 氦", k=2)
    assert ranked and ranked[0][0] == "c1"


def test_rerank_tfidf_cosine_ordering():
    idx = LexicalIndex()
    idx.add("c1", "深度学习 神经网络 大模型")
    idx.add("c2", "烹饪 美食 菜谱")
    reranker = Reranker(idx)
    ranked = reranker.rerank("深度学习 神经网络", ["c1", "c2"], top_n=2)
    assert ranked[0][0] == "c1"


def test_hybrid_retriever_relevant_doc_first(knowledge):
    results = knowledge.retriever.retrieve("恒星的主要成分", top_n=3)
    assert results
    assert results[0].chunk.doc_id == "d1"


def test_hybrid_retriever_second_query(knowledge):
    results = knowledge.retriever.retrieve("什么是机器学习", top_n=3)
    assert results
    assert results[0].chunk.doc_id == "d2"

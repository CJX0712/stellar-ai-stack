import math

from stellar_ai.embedding.provider import HashEmbedding, tokenize


def test_dim_and_determinism():
    e = HashEmbedding(dim=64)
    assert e.dim == 64
    v1 = e.embed_one("恒星的主要成分")
    v2 = e.embed_one("恒星的主要成分")
    assert v1 == v2  # deterministic


def test_normalized():
    e = HashEmbedding(dim=64)
    v = e.embed_one("人工智能 机器学习 深度学习")
    mag = math.sqrt(sum(x * x for x in v))
    assert abs(mag - 1.0) < 1e-9


def test_similar_closer_than_dissimilar():
    e = HashEmbedding(dim=128)
    base = e.embed_one("恒星 核聚变 氢 氦")
    sim = e.embed_one("恒星 主要由 氢 和 氦 组成")
    dis = e.embed_one("篮球 比赛 投篮 得分")
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    assert dot(base, sim) > dot(base, dis)


def test_tokenize_cjk_bigrams():
    toks = tokenize("恒星")
    assert "c:恒" in toks
    assert "b:恒星" in toks
    assert "w:ai" in tokenize("ai")

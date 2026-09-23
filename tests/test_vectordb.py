import math

from stellar_ai.core.errors import VectorStoreError
from stellar_ai.core.types import Chunk
from stellar_ai.vectordb.store import InMemoryVectorStore


def _chunk(cid, dim, value):
    vec = [0.0] * dim
    vec[0] = value
    mag = math.sqrt(sum(v * v for v in vec))
    norm = [v / mag for v in vec]
    return Chunk(id=cid, doc_id="d", text="x"), norm


def test_put_query_sorted():
    store = InMemoryVectorStore()
    dim = 8
    c1, v1 = _chunk("c1", dim, 1.0)
    c2, v2 = _chunk("c2", dim, 0.5)
    store.put_chunks([c1, c2], [v1, v2])
    q = [1.0] + [0.0] * (dim - 1)
    res = store.query(q, k=2)
    assert res[0][0] == "c1"
    assert res[0][1] >= res[1][1]
    assert store.count() == 2


def test_drop_document():
    store = InMemoryVectorStore()
    dim = 4
    c1, v1 = _chunk("c1", dim, 1.0)
    c2, v2 = _chunk("c2", dim, 0.5)
    c2.doc_id = "d2"
    store.put_chunks([c1, c2], [v1, v2])
    store.drop_document("d")
    assert store.count() == 1
    # c1 removed; query should no longer return it.
    assert all(cid != "c1" for cid, _ in store.query([1.0, 0, 0, 0], k=5))


def test_dimension_mismatch_raises():
    store = InMemoryVectorStore()
    c, v = _chunk("c1", 4, 1.0)
    store.put_chunks([c], [v])
    try:
        store.query([1.0, 0, 0], k=1)
        assert False, "expected VectorStoreError"
    except VectorStoreError:
        pass

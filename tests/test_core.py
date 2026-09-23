from stellar_ai.core.config import Config
from stellar_ai.core.errors import (
    ConfigError,
    EmbeddingError,
    StellarError,
    VectorStoreError,
)
from stellar_ai.core.types import Answer, Chunk, Citation, Document, RetrievedChunk


def test_config_defaults():
    c = Config.from_env({})
    assert c.embedding_provider == "hash"
    assert c.llm_provider == "mock"
    assert c.vectorstore_provider == "memory"
    assert c.rerank_top_n == 5


def test_config_env_override():
    env = {
        "STELLAR_LLM_PROVIDER": "openai",
        "STELLAR_RERANK_TOP_N": "8",
        "STELLAR_EMBEDDING_DIM": "64",
    }
    c = Config.from_env(env)
    assert c.llm_provider == "openai"
    assert c.rerank_top_n == 8
    assert c.embedding_dim == 64


def test_config_redacts_api_key():
    c = Config.from_env({"STELLAR_LLM_API_KEY": "secret"})
    assert c.as_dict()["llm_api_key"] == "***"


def test_error_hierarchy():
    assert issubclass(EmbeddingError, StellarError)
    assert issubclass(VectorStoreError, StellarError)
    assert issubclass(ConfigError, StellarError)


def test_types_roundtrip():
    doc = Document(id="d", title="t", text="x")
    chunk = Chunk(id="d#0", doc_id="d", text="x")
    rc = RetrievedChunk(chunk=chunk, score=0.5)
    cit = Citation(doc_id="d", chunk_id="d#0", snippet="x", score=0.5)
    ans = Answer(question="q", answer="a", citations=[cit], contexts=["x"])
    assert ans.question == "q"
    assert rc.chunk is chunk

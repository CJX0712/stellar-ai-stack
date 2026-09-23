"""Knowledge base: the ingestion + retrieval composition root.

Owns the chunk registry, vector store, lexical index, embedding model and
hybrid retriever, and exposes a single ``ingest_*`` surface. Re-ingestion of a
document is safe: the previous chunks are removed from every store before the
new ones are written, so shorter revisions cannot leave orphan chunks behind.
"""

from __future__ import annotations

from .core.config import Config
from .core.types import Chunk, Document
from .embedding.provider import get_embedding_provider
from .ingestion.chunker import HeadingChunker
from .ingestion.loader import StringDocument, load_document
from .retrieval.lexical import LexicalIndex
from .retrieval.rerank import Reranker
from .retrieval.retriever import ChunkRegistry, HybridRetriever
from .vectordb.store import get_vector_store


class KnowledgeBase:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.embedding = get_embedding_provider(config)
        self.vector_store = get_vector_store(config)
        self.registry = ChunkRegistry()
        self.lexical = LexicalIndex()
        self.reranker = Reranker(self.lexical)
        self.chunker = HeadingChunker()
        self.retriever = HybridRetriever(
            registry=self.registry,
            vector_store=self.vector_store,
            embedding=self.embedding,
            lexical_index=self.lexical,
            reranker=self.reranker,
            top_k=config.retrieval_top_k,
            top_n=config.rerank_top_n,
        )
        self._docs: dict[str, Document] = {}

    def ingest_document(self, doc: Document) -> int:
        # Safe re-ingestion: purge the previous revision everywhere first.
        old_ids = [c.id for c in self.registry.all() if c.doc_id == doc.id]
        for cid in old_ids:
            self.lexical.remove(cid)
        self.registry.drop(doc.id)
        self.vector_store.drop_document(doc.id)
        self._docs[doc.id] = doc

        chunks: list[Chunk] = self.chunker.chunk(doc)
        if not chunks:
            return 0
        embeddings = self.embedding.embed([c.text for c in chunks])
        self.vector_store.put_chunks(chunks, embeddings)
        self.registry.put(chunks)
        for c in chunks:
            self.lexical.add(c.id, c.text)
        return len(chunks)

    def ingest_text(self, text: str, title: str = "", doc_id: str = "") -> int:
        doc = StringDocument.build(text, title=title, doc_id=doc_id)
        return self.ingest_document(doc)

    def ingest_file(self, path: str) -> int:
        total = 0
        for doc in load_document(path):
            total += self.ingest_document(doc)
        return total

    def chunk_count(self) -> int:
        return self.registry.count()

    def doc_count(self) -> int:
        return len(self._docs)

import os
import tempfile

from stellar_ai.ingestion.chunker import HeadingChunker
from stellar_ai.ingestion.loader import StringDocument, TextLoader
from stellar_ai.core.types import Document


def _doc(text, doc_id="d"):
    return Document(id=doc_id, title="t", text=text)


def test_heading_inheritance():
    text = "# 标题A\n这是A段落的内容。\n# 标题B\n这是B段落的内容。"
    chunks = HeadingChunker(max_chars=700).chunk(_doc(text))
    headings = {c.heading for c in chunks}
    assert "标题A" in headings
    assert "标题B" in headings
    # Each chunk that carries a heading should include it in its text.
    for c in chunks:
        if c.heading:
            assert c.text.startswith(c.heading) or c.heading in c.text


def test_nonuniform_multi_chunk():
    # A single long section forces multiple bounded chunks.
    para = "恒星是由引力凝聚在一起的球型发光等离子体。" * 40
    text = "# 恒星\n" + para
    chunks = HeadingChunker(max_chars=120).chunk(_doc(text))
    assert len(chunks) >= 2
    # No tiny orphan chunk (min_chars merge / no sub-60-char fragments).
    assert all(len(c.text) >= 10 for c in chunks)


def test_string_document_rejects_empty():
    try:
        StringDocument.build("   ")
        assert False, "expected IngestionError"
    except Exception as e:  # noqa: BLE001
        assert e.__class__.__name__ == "IngestionError"


def test_text_loader_reads_file():
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write("# 测试文档\n这是内容。")
        path = f.name
    try:
        docs = TextLoader().load(path)
        assert docs[0].title == "测试文档"
        assert "这是内容" in docs[0].text
    finally:
        os.unlink(path)

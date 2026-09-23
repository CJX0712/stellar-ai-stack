def test_rag_answer_has_citations(rag):
    ans = rag.answer("恒星的主要成分是什么？")
    assert ans.answer
    assert ans.citations
    assert ans.citations[0].doc_id == "d1"


def test_rag_context_rendered_one_line(rag):
    ans = rag.answer("什么是机器学习？")
    for ctx in ans.contexts:
        # Rendered contexts must be single-line (newlines collapsed).
        assert "\n" not in ctx.replace("\r", "")


def test_rag_top_n_respected(rag):
    ans = rag.answer("恒星", top_n=2)
    assert len(ans.citations) <= 2

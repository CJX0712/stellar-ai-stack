from stellar_ai.llm.provider import MockLLM, OpenAIProvider


def test_mock_returns_final_answer_with_context():
    llm = MockLLM()
    prompt = (
        "User question: 恒星的主要成分是什么？\n"
        "<kb-context>\n"
        "[id#0] (source: 恒星) 恒星的主要成分是氢和氦，通过核聚变释放能量。\n"
        "</kb-context>\n"
        "Final Answer:"
    )
    out = llm.complete(prompt)
    assert out.startswith("Final Answer:")
    assert "氢" in out and "氦" in out


def test_mock_neutral_without_context():
    llm = MockLLM()
    out = llm.complete("User question: 任意问题\nFinal Answer:")
    assert out.startswith("Final Answer:")


def test_mock_agent_action_when_no_context():
    llm = MockLLM()
    prompt = (
        "可用工具：search、calculator、date。\n"
        "User question: 介绍一下恒星\nAction Input:"
    )
    out = llm.complete(prompt)
    assert "Action: search" in out


def test_openai_provider_requires_url_scheme():
    # Construction should not require network; just validate wiring.
    p = OpenAIProvider(model="x", base_url="http://localhost:11434/v1")
    assert p.base_url.endswith("/v1")

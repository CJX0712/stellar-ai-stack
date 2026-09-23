from stellar_ai.agent.agent import Agent, safe_eval


def test_safe_eval_basic():
    val, err = safe_eval("2+3*4")
    assert err is None and val == 14


def test_safe_eval_div_zero():
    _, err = safe_eval("1/0")
    assert err == "division by zero"


def test_safe_eval_rejects_code():
    _, err = safe_eval("import os")
    assert err is not None


def test_agent_calculator_preroute(agent):
    out = agent.run("计算 12 * 34")
    assert "408" in out


def test_agent_date_preroute(agent):
    out = agent.run("今天几号？")
    assert "今天" in out


def test_agent_search_and_synthesize(agent):
    out = agent.run("介绍一下机器学习")
    assert out and ("机器学习" in out or "人工智能" in out)


class ScriptedLLM:
    """Returns a scripted ReAct sequence to exercise the tool loop."""

    def __init__(self):
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        if "kb-context" in prompt:
            return "Final Answer: 机器学习是人工智能的一个分支。"
        return "Action: search\nAction Input: 机器学习"


def test_agent_react_loop_executes_search(system):
    llm = ScriptedLLM()
    agent = Agent(system.knowledge.retriever, llm, max_steps=4)
    out = agent.run("请介绍机器学习")
    assert "机器学习是人工智能的一个分支" in out
    assert llm.calls >= 2  # at least the Action step and the synthesis step

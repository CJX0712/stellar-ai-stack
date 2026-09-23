"""ReAct agent with deterministic tool routing.

Two classes of tasks are resolved WITHOUT calling the LLM at all (a hard rule
that keeps small models honest and makes the default offline run deterministic):

  * arithmetic  -> a safe expression evaluator (handles + - * / and unary minus,
                   normalises full-width punctuation, never executes arbitrary
                   code; ZeroDivisionError / OverflowError become ToolError)
  * date/time   -> the current date

Everything else enters a short ReAct loop. The agent asks the LLM for an Action;
``search`` triggers retrieval and is immediately synthesised into a final answer
(via the same extractive generator as RAG), while ``calculator`` / ``date``
return their result directly. The loop is bounded by ``max_steps``.
"""

from __future__ import annotations

import ast
import datetime
import math
import re

from ..core.errors import AgentError
from ..core.interfaces import LLMProvider, Retriever

_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.USub,
    ast.UAdd,
)

# Characters kept when normalising an arithmetic expression. Operators are
# intentionally included; stripping them would corrupt legitimate expressions.
_ALLOWED_CHARS = set("0123456789.+-*/() ")

_DATE_WORDS = ("今天", "日期", "现在", "时间", "几月", "星期", "today", "date", "now")


def _normalize_expr(text: str) -> str:
    # Strip full-width punctuation that would otherwise hang on the expression.
    text = text.replace("？", " ").replace("。", " ").replace("，", " ").replace("、", " ")
    return "".join(ch for ch in text if ch in _ALLOWED_CHARS).strip()


def safe_eval(expr: str):
    """Evaluate a benign arithmetic expression. Returns (value, error)."""
    norm = _normalize_expr(expr)
    if not norm or not re.search(r"[+\-*/]", norm) or not re.search(r"\d", norm):
        return None, "not an arithmetic expression"
    try:
        tree = ast.parse(norm, mode="eval")
    except SyntaxError:
        return None, "syntax error"
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return None, "disallowed expression"
    try:
        value = eval(compile(tree, "<expr>", "eval"))  # noqa: S307 - guarded AST
    except ZeroDivisionError:
        return None, "division by zero"
    except OverflowError:
        return None, "overflow"
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        return None, "overflow"
    return value, None


class Agent:
    def __init__(self, retriever: Retriever, llm: LLMProvider, max_steps: int = 6) -> None:
        self.retriever = retriever
        self.llm = llm
        self.max_steps = max_steps

    # ----- deterministic pre-routing -------------------------------------
    def _preroute(self, task: str):
        if any(w in task.lower() for w in _DATE_WORDS):
            now = datetime.datetime.now()
            return f"今天是 {now.year}年{now.month}月{now.day}日。"
        value, err = safe_eval(task)
        if err is None and value is not None:
            return f"{task.strip()} 的计算结果是 {value}。"
        return None

    # ----- tools ---------------------------------------------------------
    def _tool_search(self, query: str) -> str:
        results = self.retriever.retrieve(query, self.retriever.top_n)
        if not results:
            return "（无相关结果）"
        lines = []
        for i, rc in enumerate(results):
            lines.append(f"[{i}] {rc.chunk.text.replace(chr(10), ' ')}")
        return "\n".join(lines)

    def _tool_calculator(self, expr: str) -> str:
        value, err = safe_eval(expr)
        return str(value) if err is None else f"错误: {err}"

    def _tool_date(self, _: str) -> str:
        now = datetime.datetime.now()
        return f"今天是 {now.year}年{now.month}月{now.day}日。"

    # ----- parsing -------------------------------------------------------
    @staticmethod
    def _parse_final(raw: str):
        if "Final Answer:" in raw:
            return raw.split("Final Answer:", 1)[1].strip()
        return None

    @staticmethod
    def _parse_action(raw: str):
        m = re.search(r"Action:\s*([a-zA-Z_]+)", raw)
        if not m:
            return None, None
        action = m.group(1).strip().lower()
        im = re.search(r"Action\s*Input:\s*(.*?)(?:\n|$)", raw, re.I)
        arg = im.group(1).strip() if im else ""
        return action, arg

    # ----- synthesis -----------------------------------------------------
    def _synthesize(self, task: str, observation: str) -> str:
        if not observation:
            return "无法完成该任务：未检索到相关信息。"
        prompt = (
            f"User question: {task}\n"
            "<kb-context>\n"
            f"{observation}\n"
            "</kb-context>\n"
            "Final Answer:"
        )
        raw = self.llm.complete(prompt)
        return self._parse_final(raw) or raw.strip()

    # ----- main loop -----------------------------------------------------
    def run(self, task: str) -> str:
        pre = self._preroute(task)
        if pre is not None:
            return pre

        system = (
            "你是一个可以使用工具的智能体。可用工具：search（检索知识库）、"
            "calculator（计算算术表达式）、date（获取当前日期）。\n"
            "如需检索，请输出：\nAction: search\nAction Input: <查询>\n"
            f"User question: {task}\n"
        )
        last_obs = ""
        for _ in range(self.max_steps):
            raw = self.llm.complete(system)
            final = self._parse_final(raw)
            if final:
                return final
            action, arg = self._parse_action(raw)
            if action is None:
                return self._synthesize(task, last_obs)
            if action == "calculator":
                return self._tool_calculator(arg or task)
            if action == "date":
                return self._tool_date(arg)
            if action == "search":
                obs = self._tool_search(arg or task)
                return self._synthesize(task, obs)
            last_obs = f"未知工具: {action}"
            system += f"\nAction: {action}\nAction Input: {arg}\nObservation: {last_obs}\n"
        return self._synthesize(task, last_obs)

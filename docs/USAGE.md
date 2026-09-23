# 使用指南 · Stellar AI

## 场景一：Web 控制台（最直观）

```bash
stellar-ai serve --port 8000
```
浏览器打开 `http://127.0.0.1:8000`：
1. **摄入知识**：粘贴文档或填文件路径 → 点"摄入文本 / 按文件摄入"。
2. **问答**：输入问题 → "提问"，返回答案 + 引用来源。
3. **智能体**：输入任务（算术 / 日期 / 检索类），返回执行结果。

## 场景二：命令行

```bash
# 摄入一段文本
stellar-ai ingest --text "深度学习使用多层神经网络。" --title 深度学习

# 摄入整个文件（.txt / .md / .pdf 需装 pypdf）
stellar-ai ingest --file ./kb/intro.md

# 提问
stellar-ai query --question "什么是深度学习？" --top-n 3

# 智能体
stellar-ai agent --task "计算 128 / 4"
stellar-ai agent --task "今天几号？"
stellar-ai agent --task "介绍一下人工智能"
```

> CLI 每个子命令是独立进程，内存索引不跨进程保留；持续使用请配合 `serve`。

## 场景三：作为 Python 库

```python
from stellar_ai import build_system

sys = build_system()
kb = sys.knowledge

# 摄入
kb.ingest_text("太阳是离地球最近的恒星，提供光和热。", doc_id="sun")
kb.ingest_file("./kb/ai.md")                 # 也可批量摄入文件

# 检索增强问答
ans = sys.rag.answer("恒星的主要成分是什么？")
print(ans.answer)                            # 答案文本
for c in ans.citations:                      # 引用来源
    print(c.doc_id, c.score, c.snippet)

# 智能体（算术/日期走确定性路由，其余走检索+生成）
print(sys.agent.run("计算 99 * 99"))
print(sys.agent.run("介绍一下机器学习"))

# 评估
from stellar_ai.eval import EvalHarness
cases = [
    {"question": "恒星的主要成分是什么？", "relevant_doc_ids": ["sun"]},
]
sc = EvalHarness(sys.rag, top_n=5).run(cases)
print(sc.doc_hit_rate, sc.doc_mrr, sc.mean_grounding)
```

## 场景四：HTTP API

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{"text":"恒星的主要成分是氢和氦。","title":"恒星","doc_id":"d1"}'

curl -X POST http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"恒星的主要成分是什么？","top_n":3}'

curl -X POST http://127.0.0.1:8000/agent \
  -H 'Content-Type: application/json' \
  -d '{"task":"计算 12 * 34"}'
```

响应示例（/query）：
```json
{
  "question": "恒星的主要成分是什么？",
  "answer": "恒星的主要成分是氢和氦，通过核聚变释放能量。",
  "citations": [{"doc_id": "d1", "chunk_id": "d1#c0", "snippet": "...", "score": 0.43}],
  "contexts": ["恒星\n恒星的主要成分是氢和氦，通过核聚变释放能量。"]
}
```

## 提示工程说明（默认 mock LLM）

- 提示中用唯一分隔符 `<kb-context>` 包裹知识，`Final Answer:` 引导最终答案。
- 切换真实 LLM 后，同样的接口不变，只是 `complete(prompt)` 的实际行为由模型决定。
- 智能体使用 ReAct 格式：`Action:` / `Action Input:` / `Observation:` / `Final Answer:`。

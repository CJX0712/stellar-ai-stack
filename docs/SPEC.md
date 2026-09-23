# 接口规范与指标基线 · Stellar AI

本文档是契约的权威来源，与活代码（`stellar_ai/core/interfaces.py`、`stellar_ai/eval/metrics.py`）保持一致。

## 1. 公共数据类型

| 类型 | 字段 |
|---|---|
| `Document` | `id, title, text, source, metadata` |
| `Chunk` | `id, doc_id, text, heading, start, end, metadata` |
| `RetrievedChunk` | `chunk: Chunk, score: float` |
| `Citation` | `doc_id, chunk_id, snippet, score` |
| `Answer` | `question, answer, citations: list[Citation], contexts: list[str], metadata` |

## 2. 模块接口

### EmbeddingProvider
```python
@property
def dim() -> int: ...
def embed(texts: list[str]) -> list[list[float]]: ...
```
- 默认 `HashEmbedding`：BLAKE2b 有符号哈希，中文按字符 unigram + bigram，英文按词；输出 L2 归一化单位向量。
- 同文本 → 同向量（确定性）。

### VectorStore
```python
def put_chunks(chunks, embeddings) -> None
def drop_document(doc_id) -> None
def query(vector, k) -> list[tuple[chunk_id, cosine_score]]
def count() -> int
```
- 默认 `InMemoryVectorStore`：精确余弦（向量入库时归一化，点积即余弦）。
- `drop_document` 保证重摄入安全（移除旧分块）。

### Retriever
```python
def retrieve(query, top_n=None) -> list[RetrievedChunk]
```
- 默认 `HybridRetriever`：稠密（向量）分支 + 稀疏（IDF 词法）分支 → RRF 融合 → TF-IDF 余弦重排。

### LLMProvider
```python
def complete(prompt: str) -> str
```
- 默认 `MockLLM`：从 `<kb-context>` 中抽取最相关句子作为 `Final Answer:`；无上下文时返回中性答案；ReAct 提示下返回 `search` 动作。
- `OpenAIProvider`：OpenAI 兼容 / Ollama / vLLM（本地端点 `trust_env=False` 规避 SOCKS）。

### RAGPipeline
```python
def answer(question, top_n=None) -> Answer
```
- 组装提示：唯一分隔符 `<kb-context>`（指令中绝不出现该串），调用 LLM，解析 `Final Answer:`。
- `contexts` 已压成单行（换行→空格），供评估与展示。

### Agent
```python
def run(task) -> str
```
- 确定性前置路由：算术表达式（安全 `ast` 求值，支持 `+ - * /`、一元负号，归一化全角标点，ZeroDivisionError/OverflowError → ToolError）/ 日期 → 直接返回。
- 否则进入 ReAct 循环：LLM 给 `Action` → 工具（`search` / `calculator` / `date`）→ 观察 → 合成 `Final Answer`。步数受 `max_steps` 限制。

## 3. 评估指标

| 指标 | 定义 | 说明 |
|---|---|---|
| `doc_hit_rate` | 检索结果中至少包含一个相关文档的查询占比 | 文档级（头条），避免未标注块低估 |
| `doc_mrr` | 首个相关文档排名的倒数均值 | 文档级排序质量 |
| `token_recall` | 命中相关块 / 相关块总数 | 块级诊断项，仅诊断 |
| `grounding_overlap` | 答案与证据的去标记词集合交并比 | 支撑度；先剥离 `[id#n]`、`(source: …)` 等出处标记 |
| `rank_correlation` | Spearman 式秩相关；任一输入为常量返回 `0.0` | 防止常量输入被误判为完全可信 |

## 4. 基线门限（已写入 `verify.py`）

| 指标 | 实测基线 | 门限 |
|---|---|---|
| `doc_hit_rate` | 1.000 | >= 0.95 |
| `doc_mrr` | 1.000 | >= 0.95 |
| `mean_grounding` | 0.986 | >= 0.10 |

门限贴着基线设：真回归必挂，数值抖动不挂。

## 5. 配置（环境变量）

见 `README.md` 配置表，全部由 `Config.from_env` 读取，默认值即零依赖离线模式。

## 6. 评估适配器防自欺条款

- **显式捕获**：评估只读取每次 `answer()` 返回的 `Answer.citations`，绝不读管线上的 `last_*` 缓存。
- **检索聚合剔除**：算术/日期等"设计上不检索"的用例从 `doc_hit_rate` / `doc_mrr` 聚合中剔除，单独计数。
- **确定性断言**：对比 payload 不含墙钟延迟字段。

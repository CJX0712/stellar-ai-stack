# 系统架构 · Stellar AI

## 设计原则

1. **单一职责**：每个模块只做一件事。
2. **依赖倒置**：模块只依赖 `core/interfaces.py` 中的 `Protocol`，运行时注入实现 → 可独立验证、可用 fake 替换。
3. **零依赖默认实现**：默认路径是纯 Python 标准库（mock LLM + 哈希嵌入 + 内存余弦 + IDF 重排），无 GPU / 无 Key / 无网络全绿。
4. **生产可插拔**：真实后端经环境变量切换，不在默认锁版内。

## 分层与接口契约

```
Layer 0  core            配置 / 日志 / 类型 / 错误 / Protocol 接口
Layer 1  ingestion       DocumentLoader / Chunker
Layer 2  embedding       EmbeddingProvider.embed(texts) -> list[list[float]]
Layer 3  vectordb        VectorStore.put_chunks / drop_document / query / count
Layer 4  retrieval       Retriever.retrieve(query, top_n) -> list[RetrievedChunk]
Layer 5  generation      RAGPipeline.answer(question, top_n) -> Answer
Layer 6  agent           Agent.run(task) -> str
Layer 7  api / cli       create_app() / 命令行
Layer 8  eval            EvalHarness.run(cases) -> Scorecard
```

### 关键接口（`stellar_ai/core/interfaces.py`）

| Protocol | 方法 | 默认实现 |
|---|---|---|
| `EmbeddingProvider` | `embed(texts) -> list[list[float]]` | `HashEmbedding`（BLAKE2b 有符号哈希，中文 bigram） |
| `VectorStore` | `put_chunks / drop_document / query / count` | `InMemoryVectorStore`（精确余弦） |
| `Retriever` | `retrieve(query, top_n)` | `HybridRetriever`（稠密 + 稀疏，RRF 融合） |
| `LLMProvider` | `complete(prompt) -> str` | `MockLLM`（抽取式，离线） |
| `RAGPipeline` | `answer(question, top_n) -> Answer` | `RAGPipeline` |
| `Agent` | `run(task) -> str` | `Agent`（ReAct + 确定性前置路由） |

## 调用关系（端到端链路）

```
Ingest:
  load_document / StringDocument.build
      -> HeadingChunker.chunk            (标题继承 + 长段切分)
      -> EmbeddingProvider.embed         (哈希嵌入)
      -> VectorStore.put_chunks          (内存余弦)
      -> LexicalIndex.add               (IDF 词法索引)
      -> ChunkRegistry.put               (chunk_id -> Chunk)

Query:
  RAGPipeline.answer
      -> Retriever.retrieve              (HybridRetriever)
            = VectorStore.query (稠密) + LexicalIndex.search (稀疏)
            -> reciprocal_rank_fusion    (RRF)
            -> Reranker.rerank           (TF-IDF 余弦)
      -> 组装 <kb-context> 提示
      -> LLMProvider.complete
      -> Answer(citations, contexts)

Agent:
  Agent.run
      -> 确定性前置路由(算术/日期) ? 直接返回
      -> LLMProvider.complete(Action) -> 工具(search/calculator/date)
      -> 观察 -> 合成 Final Answer
```

## 模块依赖图

```
composition.build_system
   └─> KnowledgeBase ─┬─> embedding.get_embedding_provider
                      ├─> vectordb.get_vector_store
                      ├─> retrieval.LexicalIndex / Reranker / HybridRetriever
                      └─> ingestion.HeadingChunker / loader
   └─> llm.get_llm_provider
   └─> generation.RAGPipeline(knowledge.retriever, llm)
   └─> agent.Agent(knowledge.retriever, llm)
   └─> api.create_app(system)  // 工厂内部注册全部路由
```

所有模块只依赖 `core`（接口 + 类型）；`composition` 是唯一装配点。

## 失败模式表（症状 / 根因 / 修法 / 守护测试）

| 症状 | 根因 | 修法 | 守护 |
|---|---|---|---|
| 重复摄入较短文档后旧内容仍可被检索 | `put_chunks` 是 upsert 不是 replace，旧分块成孤儿 | 摄入前 `drop_document` + `LexicalIndex.remove` | `test_vectordb` / smoke 重摄入 |
| 检索指标莫名偏低 | 评估读到对象的 `last_*` 缓存，被 agent 的 search 覆盖 | 评估适配器显式捕获本次 `Answer` | `test_eval` |
| IDF 在 2 篇语料时得分全 0 / 排序反转 | 概率 IDF 在词恰好出现一半文档时 = ln(1)=0 | 改用 Robertson IDF `ln(1+(N-n+0.5)/(n+0.5))` 恒非负 | `test_retrieval` |
| 每个 chunk 只剩首行进入模型（上下文丢 90%） | 证据块按多行渲染、消费方逐行解析 | 渲染时把每块压成一行；长段每块都带标题前缀 | `test_rag` |
| 模型对 12*(3+4) 直接输出错误 Final Answer | 小模型路由/算术不可靠 | 确定性前置路由：正则检出算术表达式 → calculator 先算 | `test_agent` |
| 标题未继承到首个分块 | 首个 block 的 start 在标题之后 | 用 `last heading before position` 语义 | `test_ingestion` |
| FastAPI 路由收集报 `Invalid args for response field` | 路由函数注解了联合返回类型 | 装饰器不加 `response_model`，返回 dict | `api/app.py` |
| pip 报 `Missing dependencies for SOCKS support` | 机器有 SOCKS5 隧道，pip 缺 PySocks | 先装 PySocks；锁版含 `PySocks==1.7.1` | `requirements.lock.txt` |
| 含中文脚本经 GBK 落盘后乱码 | PowerShell 按 GBK 解码 Python 的 UTF-8 stdout | verify 用 Python 直接写 UTF-8 文件；中文脚本用 Write 工具写 | `scripts/verify.py` |
| 评估指标被"设计上不检索"的用例拉低 | 算术/日期路由期望零证据却被计入平均 | 此类用例从检索聚合剔除（`counts_as_retrieval`） | `harness.py` |
| 同源常量输入被算成完全可信（+1.0） | 秩相关对常量输入未定义，tie-break 造序 | 任一输入常量返回 0.0 | `test_eval` |

## 确定性保证

- 哈希嵌入：同一文本 → 同一单位向量（BLAKE2b，无随机数）。
- 向量索引：精确余弦，无近似。
- 检索 / 重排：纯函数，无随机。
- mock LLM：抽取式，无随机。
- 因此 `verify.py` 的"确定性"阶段：同一问题两次回答逐字节相同。

## 可扩展性

新增一个真实后端（如 Qdrant、MongoDB、远端嵌入）只需：
1. 在对应层实现一个满足 `Protocol` 的类；
2. 在 `get_*_provider(config)` 中加一个分支；
3. 不改任何调用方代码。

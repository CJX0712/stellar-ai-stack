# Stellar AI · 世界级模块化 AI 系统

<p align="center">
  <a href="https://github.com/CJX0712/stellar-ai-stack/actions/workflows/ci.yml"><img src="https://github.com/CJX0712/stellar-ai-stack/actions/workflows/ci.yml/badge.svg" alt="ci"></a>
  <a href="https://github.com/CJX0712/stellar-ai-stack/releases"><img src="https://img.shields.io/github/v/release/CJX0712/stellar-ai-stack?sort=semver" alt="release"></a>
  <a href="https://github.com/CJX0712/stellar-ai-stack/blob/master/LICENSE"><img src="https://img.shields.io/github/license/CJX0712/stellar-ai-stack" alt="license"></a>
  <img src="https://img.shields.io/badge/author-%E6%99%A8%E6%98%9F-1f6feb" alt="author">
</p>

> 端到端可实际运行的 **RAG（检索增强生成）+ ReAct 智能体** 平台。
> 复用业界领先的开源范式（FastAPI 服务化、Protocol 注入式可插拔架构、混合检索、ReAct 智能体），
> 不重复造轮子；核心价值在**架构、可验证性与可复现性**。

作者：**晨星** ｜ License：MIT ｜ Version：1.0.0

---

## 一句话定位

一套**默认零依赖、离线可跑、确定性可验证**的 AI 系统：内置 mock LLM + BLAKE2b 哈希嵌入 +
内存余弦索引 + IDF 词法重排，无需 GPU、无需 API Key、无需联网即可跑通完整链路；
真实大模型 / 向量库 / 嵌入通过环境变量一键切换（`optional-requirements.txt`）。

## 核心特性

- **单一职责 + 接口注入**：每个模块只做一件事，通过 `Protocol` 解耦，可独立单测、可用 fake 替换。
- **零依赖默认实现**：默认路径纯 Python 标准库，离线 `git clone && verify` 全绿。
- **生产可插拔**：sentence-transformers / faiss-cpu / llama-cpp-python / Ollama 经环境变量接入。
- **混合检索**：稠密（向量）+ 稀疏（IDF 词法）双路，RRF 融合 + TF-IDF 余弦重排。
- **确定性智能体**：算术 / 日期走前置路由（不经 LLM），其余走 ReAct 循环。
- **可验证**：`scripts/verify.py` 八阶段自检（P0 字符门禁 → import → pytest → 运行时不变量 → 确定性 → 评估门限 → API E2E）。
- **一键复现**：`requirements.lock.txt` 为已验证可安装的精确版本锁。
- **文档即契约**：OpenAPI 由活代码生成，架构 / 部署 / 使用指南齐全。

## 端到端链路

```
ingest(load -> chunk) -> embed -> vectordb.put -> retriever.retrieve(混合)
   -> rerank -> rag.augment -> llm.complete -> Answer
Agent: run() 循环调用 retriever / llm / tools(calc, date) 形成 ReAct 闭环
```

## 快速开始

```bash
# 1. 准备环境（Python >= 3.11）
python -m venv .venv
.venv/Scripts/pip install -r requirements.lock.txt   # Windows
# .venv/bin/pip install -r requirements.lock.txt      # Linux/macOS

# 2. 安装本项目（提供 stellar-ai 命令行）
pip install -e .

# 3. 自检（必须全绿）
python scripts/verify.py

# 3. 启动服务
stellar-ai serve --host 127.0.0.1 --port 8000
#   浏览器打开 http://127.0.0.1:8000 使用控制台

# 4. 命令行
stellar-ai ingest --text "恒星的主要成分是氢和氦。" --title 恒星
stellar-ai query  --question "恒星的主要成分是什么？"
stellar-ai agent  --task "计算 12 * 34"
```

## 仓库结构

```
stellar-ai/
├── stellar_ai/                 # 源码包（模块化）
│   ├── core/                   # 配置 / 日志 / 类型 / 错误 / 接口契约
│   ├── ingestion/              # 文档加载 + 标题感知分块
│   ├── embedding/              # 哈希嵌入（默认）+ 真实模型接口
│   ├── vectordb/               # 内存余弦索引（默认）+ FAISS 接口
│   ├── retrieval/              # 混合检索 + IDF 重排
│   ├── llm/                    # mock LLM（默认）+ OpenAI 兼容
│   ├── generation/             # RAG 管线 + 抽取式阅读器
│   ├── agent/                  # ReAct 智能体（确定性路由）
│   ├── api/                    # FastAPI 工厂 + 单文件 HTML 控制台
│   ├── cli/                    # 命令行入口
│   ├── eval/                   # 评估适配器 + 文档级指标
│   ├── knowledge.py            # 摄入 + 检索组合根
│   └── composition.py          # build_system 装配根
├── tests/                      # 逐模块单测 + 进程内 HTTP E2E
├── scripts/                    # verify.py / scan_emoji.py / gen_openapi.py
├── docs/                       # SPEC / openapi / ADR / 部署 / 使用
├── requirements.lock.txt       # 精确版本锁（已验证可安装）
├── optional-requirements.txt   # 生产级模型后端（可选）
├── pyproject.toml
├── README.md / ARCHITECTURE.md
└── LICENSE
```

## 验证状态（基线）

| 指标 | 基值 | 门限 |
|---|---|---|
| 文档级命中率 doc_hit_rate | 1.000 | >= 0.95 |
| 文档级 MRR doc_mrr | 1.000 | >= 0.95 |
| 平均支撑度 grounding | 0.986 | >= 0.10 |
| 单元测试 + API E2E | 44 passed | 0 failed |

详见 `docs/SPEC.md` 与 `verify_report.json`。

## 配置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `STELLAR_EMBEDDING_PROVIDER` | `hash` | `hash` / `fastembed` / `sentence-transformers` |
| `STELLAR_VECTORSTORE_PROVIDER` | `memory` | `memory` / `faiss` |
| `STELLAR_LLM_PROVIDER` | `mock` | `mock` / `openai` / `ollama` / `llmcpp` |
| `STELLAR_LLM_MODEL` | `mock-1` | 模型名 |
| `STELLAR_LLM_BASE_URL` | `""` | OpenAI 兼容 / Ollama 端点 |
| `STELLAR_LLM_API_KEY` | `""` | API Key（不落日志） |
| `STELLAR_EMBEDDING_DIM` | `256` | 哈希嵌入维度 |
| `STELLAR_RETRIEVAL_TOP_K` | `20` | 每个分支候选数 |
| `STELLAR_RERANK_TOP_N` | `5` | 最终返回块数 |
| `STELLAR_AGENT_MAX_STEPS` | `6` | ReAct 最大步数 |
| `STELLAR_LOG_LEVEL` | `INFO` | 日志级别 |

## 文档导航

- 架构：`ARCHITECTURE.md`
- 接口契约 / 指标 / 配置：`docs/SPEC.md`
- API 契约：`docs/openapi.yaml`（由 `scripts/gen_openapi.py` 从活代码生成）
- 部署：`docs/DEPLOYMENT.md`
- 使用：`docs/USAGE.md`
- 设计决策：`docs/decisions/ADR-*.md`

---
© 2026 晨星 · MIT License

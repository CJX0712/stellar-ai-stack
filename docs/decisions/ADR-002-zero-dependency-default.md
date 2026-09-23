# ADR-002 · 零依赖默认实现

- 状态：已采纳（2026-09-24）
- 作者：晨星

## 背景

系统需在"干净环境一键复现"且"无 GPU / 无 API Key / 无网络"下可跑通，否则无法证明它能实际运行。
重型 AI 依赖（torch / transformers / faiss 二进制）在受限环境常编译失败或下载超时。

## 决策

默认实现全部用 Python 标准库，零第三方依赖：
- 嵌入：BLAKE2b 有符号哈希（中文 bigram），确定性、维度稳定。
- 向量库：内存精确余弦（`InMemoryVectorStore`）。
- 检索：IDF 词法（Robertson IDF）+ RRF 融合 + TF-IDF 余弦重排。
- LLM：抽取式 mock（从上下文挑最相关句子）。
- 仅 API 层引入 fastapi / uvicorn / pydantic；测试引入 pytest / httpx。

真实后端（sentence-transformers / faiss-cpu / llama-cpp-python）放入 `optional-requirements.txt`，
经环境变量切换，不在默认锁版内。

## 后果

- `git clone && pip install -r requirements.lock.txt && python scripts/verify.py` 在裸 CPython 上全绿。
- 默认配置就是被持续验证的配置，可信度高。
- 代价：默认检索/生成质量弱于真实大模型；但这是"可运行基线"，生产可一键升级。

## 备选

- 默认就接 torch + transformers：功能强但不可复现、不可离线 → 否决。
- 纯 stub mock：能通过测试但"跑起来"是假象 → 否决，mock 必须是真实现。

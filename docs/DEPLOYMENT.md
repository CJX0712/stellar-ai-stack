# 部署指南 · Stellar AI

## 一、环境要求

- Python >= 3.11（已在 3.13 验证）
- 无 GPU 要求；默认零依赖可离线运行
- 可选生产后端：`optional-requirements.txt`（sentence-transformers / faiss-cpu / llama-cpp-python / pypdf）

## 二、干净环境一键复现

```bash
# 克隆（发布后）
git clone https://github.com/CJX0712/stellar-ai-stack.git
cd stellar-ai-stack

# 创建洁净虚拟环境并安装锁版依赖
python -m venv .venv
.venv/Scripts/pip install -r requirements.lock.txt     # Windows
# .venv/bin/pip  install -r requirements.lock.txt        # Linux/macOS

# 安装本项目（提供 stellar-ai 命令行入口）
pip install -e .

# 验证（必须全绿：P0 -> import -> pytest -> 不变量 -> 确定性 -> 评估 -> E2E）
python scripts/verify.py
```

> 若处于 SOCKS 代理环境，锁版已含 `PySocks==1.7.1`，pip 可正常使用隧道；
> 普通网络下它无害且纯 Python，可正常安装。

## 三、本地运行

### 3.1 命令行
```bash
stellar-ai ingest --file ./docs/intro.md
stellar-ai query  --question "..." --top-n 3
stellar-ai agent  --task "计算 12 * 34"
```

### 3.2 HTTP 服务
```bash
stellar-ai serve --host 0.0.0.0 --port 8000
# 健康检查： GET /health
# 摄入：     POST /ingest   {text, title, doc_id} | {file_path}
# 问答：     POST /query    {question, top_n}
# 智能体：   POST /agent    {task}
# 控制台：   GET /          (单文件 HTML)
```

### 3.3 作为库
```python
from stellar_ai import build_system
sys = build_system()                 # 默认 mock / hash / memory
sys.knowledge.ingest_text("恒星的主要成分是氢和氦。", doc_id="d1")
ans = sys.rag.answer("恒星的主要成分是什么？")
print(ans.answer, [c.doc_id for c in ans.citations])
```

## 四、接入生产级模型

无需改代码，仅改环境变量（详见 `README.md` 配置表）：

```bash
# 真实嵌入 + 向量库 + 大模型（Ollama 本地）
export STELLAR_EMBEDDING_PROVIDER=fastembed
export STELLAR_VECTORSTORE_PROVIDER=faiss
export STELLAR_LLM_PROVIDER=ollama
export STELLAR_LLM_BASE_URL=http://localhost:11434/v1
export STELLAR_LLM_MODEL=qwen2.5:7b
pip install -r optional-requirements.txt
```

## 五、容器化（可选）

```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.lock.txt .
RUN pip install --no-cache-dir -r requirements.lock.txt
COPY . .
EXPOSE 8000
CMD ["python", "-m", "stellar_ai.cli.main", "serve", "--host", "0.0.0.0", "--port", "8000"]
```

注意：锁版依赖在 Windows 生成时会含 `PySocks`/`colorama` 等，均跨平台可装；
若需纯 Linux 镜像，可在镜像内重新 `pip install` 顶层包后 `pip freeze` 生成 Linux 锁。

## 六、CI 建议

在 CI 中执行：
```bash
python -m pip install -r requirements.lock.txt
python scripts/verify.py        # 产出 verify_report.json 作为产物
```
`verify.py` 任一阶段失败即非零退出，可作为合并门禁。

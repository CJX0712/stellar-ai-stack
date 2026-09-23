"""FastAPI application factory.

``create_app`` builds the system from configuration and registers every route
internally, returning an app that is ready to mount or test. No global side
effects: routes live on the returned instance, which is what makes the TestClient
E2E deterministic and lets multiple apps coexist in one process.
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ..composition import System, build_system
from ..core.config import Config
from ..core.logging import get_logger

logger = get_logger("api.app")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
INDEX_HTML = os.path.join(STATIC_DIR, "index.html")


class IngestRequest(BaseModel):
    text: str | None = None
    title: str = ""
    doc_id: str = ""
    file_path: str | None = None


class QueryRequest(BaseModel):
    question: str
    top_n: int | None = None


class AgentRequest(BaseModel):
    task: str


def create_app(config: Config | None = None, system: System | None = None) -> FastAPI:
    app = FastAPI(
        title="Stellar AI",
        version="1.0.0",
        description="World-class modular RAG + ReAct agent system (zero-dependency default).",
    )
    if system is None:
        system = build_system(config)

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "version": "1.0.0",
            "chunks": system.knowledge.chunk_count(),
            "docs": system.knowledge.doc_count(),
            "config": system.config.as_dict(),
        }

    @app.post("/ingest")
    def ingest(req: IngestRequest):
        if req.file_path:
            n = system.knowledge.ingest_file(req.file_path)
            return {"ok": True, "source": req.file_path, "chunks": n}
        if not req.text:
            return JSONResponse(
                status_code=400, content={"error": "provide 'text' or 'file_path'"}
            )
        n = system.knowledge.ingest_text(req.text, title=req.title, doc_id=req.doc_id)
        return {"ok": True, "chunks": n}

    @app.post("/query")
    def query(req: QueryRequest):
        if not req.question:
            return JSONResponse(status_code=400, content={"error": "question required"})
        answer = system.rag.answer(req.question, top_n=req.top_n)
        return {
            "question": answer.question,
            "answer": answer.answer,
            "citations": [c.__dict__ for c in answer.citations],
            "contexts": answer.contexts,
        }

    @app.post("/agent")
    def agent(req: AgentRequest):
        if not req.task:
            return JSONResponse(status_code=400, content={"error": "task required"})
        result = system.agent.run(req.task)
        return {"task": req.task, "result": result}

    @app.get("/")
    def index():
        if os.path.exists(INDEX_HTML):
            return FileResponse(INDEX_HTML, media_type="text/html")
        return JSONResponse(status_code=404, content={"error": "index.html missing"})

    return app

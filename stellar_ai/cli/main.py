"""Command-line interface for Stellar AI.

Subcommands:
  ingest   load text or a file into the knowledge base
  query    ask a RAG question (optionally ingest a file first)
  agent    run a task through the ReAct agent
  serve    start the HTTP API with uvicorn

The CLI builds an in-process System from environment configuration. Pair it with
``serve`` (which keeps the store alive) for persistent use, or pass ``--file`` to
``query``/``agent`` for a one-shot demo.
"""

from __future__ import annotations

import argparse
import sys

from ..composition import build_system
from ..core.logging import get_logger

logger = get_logger("cli.main")


def _build() -> "object":
    return build_system()


def cmd_ingest(args: argparse.Namespace) -> int:
    sys = _build()
    if args.file:
        n = sys.knowledge.ingest_file(args.file)
        print(f"ingested file {args.file}: {n} chunks")
    elif args.text:
        n = sys.knowledge.ingest_text(args.text, title=args.title)
        print(f"ingested text: {n} chunks")
    else:
        print("error: provide --text or --file")
        return 2
    print(f"total chunks={sys.knowledge.chunk_count()} docs={sys.knowledge.doc_count()}")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    sys = _build()
    if args.file:
        sys.knowledge.ingest_file(args.file)
    answer = sys.rag.answer(args.question, top_n=args.top_n)
    print(f"Q: {answer.question}")
    print(f"A: {answer.answer}")
    for c in answer.citations:
        print(f"  [cite] {c.doc_id} score={c.score} :: {c.snippet[:80]}")
    return 0


def cmd_agent(args: argparse.Namespace) -> int:
    sys = _build()
    if args.file:
        sys.knowledge.ingest_file(args.file)
    result = sys.agent.run(args.task)
    print(result)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from ..api.app import create_app

    app = create_app()
    logger.info("starting Stellar AI on %s:%s", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stellar-ai", description="Stellar AI CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="ingest text or a file")
    p_ing.add_argument("--text", default=None)
    p_ing.add_argument("--title", default="")
    p_ing.add_argument("--file", default=None)
    p_ing.set_defaults(func=cmd_ingest)

    p_q = sub.add_parser("query", help="ask a RAG question")
    p_q.add_argument("--question", required=True)
    p_q.add_argument("--file", default=None)
    p_q.add_argument("--top-n", type=int, default=None, dest="top_n")
    p_q.set_defaults(func=cmd_query)

    p_a = sub.add_parser("agent", help="run a ReAct agent task")
    p_a.add_argument("--task", required=True)
    p_a.add_argument("--file", default=None)
    p_a.set_defaults(func=cmd_agent)

    p_s = sub.add_parser("serve", help="start the HTTP API")
    p_s.add_argument("--host", default="127.0.0.1")
    p_s.add_argument("--port", type=int, default=8000)
    p_s.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

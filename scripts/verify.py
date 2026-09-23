"""Stellar AI end-to-end verification (eight phases).

Run with the project virtualenv active:
    python scripts/verify.py

Phases (any failure short-circuits, report is written to verify_report.json):
  1. P0 character gate        - no emoji literals in sources
  2. Module imports           - every module imports cleanly
  3. Unit + API tests         - pytest (incl. in-process HTTP E2E)
  4. Runtime invariants       - embedding normalised, vector/rerank ordering
  5. Determinism              - identical query twice yields identical answer
  6. Scorecard gate           - evaluation metrics above documented thresholds
  7. API E2E (explicit)       - ingest -> query -> agent via TestClient
  8. Report                   - JSON summary written for CI / humans

The thresholds are pinned to the measured baseline so that a real regression
fails the build while numerical jitter does not.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_PATH = os.path.join(ROOT, "verify_report.json")

# Documented baseline thresholds (see docs/SPEC.md).
GATE = {"doc_hit_rate": 0.95, "doc_mrr": 0.95, "mean_grounding": 0.10}


def _phase(name: str, ok: bool, detail: str) -> dict:
    return {"phase": name, "ok": bool(ok), "detail": detail}


def run() -> int:
    results: list[dict] = []
    failed = False

    # ---- Phase 1: P0 character gate -----------------------------------
    sys.path.insert(0, ROOT)
    from scripts.scan_emoji import scan_project

    findings = scan_project(ROOT)
    ok = len(findings) == 0
    results.append(_phase("P0 character gate", ok, f"{len(findings)} emoji literals"))
    if not ok:
        return _finish(results)

    # ---- Phase 2: module imports --------------------------------------
    import importlib

    mods = [
        "stellar_ai",
        "stellar_ai.core",
        "stellar_ai.ingestion",
        "stellar_ai.embedding",
        "stellar_ai.vectordb",
        "stellar_ai.retrieval",
        "stellar_ai.llm",
        "stellar_ai.generation",
        "stellar_ai.agent",
        "stellar_ai.eval",
        "stellar_ai.knowledge",
        "stellar_ai.composition",
        "stellar_ai.api.app",
        "stellar_ai.cli.main",
    ]
    import_errors = []
    for m in mods:
        try:
            importlib.import_module(m)
        except Exception as exc:  # noqa: BLE001
            import_errors.append(f"{m}: {exc}")
    ok = not import_errors
    results.append(_phase("module imports", ok, "; ".join(import_errors) or "all imported"))
    if not ok:
        return _finish(results)

    # ---- Phase 3: pytest (unit + API E2E) -----------------------------
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    # Pull the summary line (contains 'passed'/'failed').
    summary = ""
    for line in reversed(proc.stdout.splitlines() + proc.stderr.splitlines()):
        if "passed" in line or "failed" in line or "error" in line:
            summary = line.strip()
            break
    ok = proc.returncode == 0
    results.append(_phase("pytest", ok, summary or f"rc={proc.returncode}"))
    if not ok:
        return _finish(results)

    # ---- Phase 4: runtime invariants ----------------------------------
    from stellar_ai.composition import build_system

    sys_mod = build_system()
    e = sys_mod.knowledge.embedding
    v = e.embed_one("恒星 氢 氦")
    mag = math.sqrt(sum(x * x for x in v))
    inv1 = abs(mag - 1.0) < 1e-9

    store = sys_mod.knowledge.vector_store
    q = store.query(v, k=5)
    inv2 = all(q[i][1] >= q[i + 1][1] for i in range(len(q) - 1)) if len(q) > 1 else True
    inv = inv1 and inv2
    results.append(
        _phase("runtime invariants", inv, f"embed_norm={mag:.6f} query_sorted={inv2}")
    )
    if not inv:
        return _finish(results)

    # ---- Phase 5: determinism -----------------------------------------
    sys_det = build_system()
    sys_det.knowledge.ingest_text(
        "# 恒星\n恒星的主要成分是氢和氦，通过核聚变释放能量。", doc_id="d1"
    )
    a1 = sys_det.rag.answer("恒星的主要成分是什么？")
    a2 = sys_det.rag.answer("恒星的主要成分是什么？")
    det = a1.answer == a2.answer and [c.chunk_id for c in a1.citations] == [
        c.chunk_id for c in a2.citations
    ]
    results.append(
        _phase("determinism", det, f"answer_match={a1.answer == a2.answer}")
    )
    if not det:
        return _finish(results)

    # ---- Phase 6: scorecard gate --------------------------------------
    from stellar_ai.eval.harness import EvalHarness

    sys_eval = build_system()
    sys_eval.knowledge.ingest_text(
        "# 恒星\n恒星的主要成分是氢和氦，通过核聚变释放能量。", doc_id="d1"
    )
    sys_eval.knowledge.ingest_text(
        "# 人工智能\n机器学习是人工智能的一个分支，依赖数据驱动的方法。", doc_id="d2"
    )
    cases = [
        {"question": "恒星的主要成分是什么？", "relevant_doc_ids": ["d1"]},
        {"question": "什么是机器学习？", "relevant_doc_ids": ["d2"]},
        {"question": "深度学习使用了什么技术？", "relevant_doc_ids": ["d2"]},
    ]
    sc = EvalHarness(sys_eval.rag, top_n=5).run(cases)
    gate_ok = (
        sc.doc_hit_rate >= GATE["doc_hit_rate"]
        and sc.doc_mrr >= GATE["doc_mrr"]
        and sc.mean_grounding >= GATE["mean_grounding"]
    )
    results.append(
        _phase(
            "scorecard gate",
            gate_ok,
            f"doc_hit={sc.doc_hit_rate} doc_mrr={sc.doc_mrr} grounding={sc.mean_grounding}",
        )
    )
    if not gate_ok:
        return _finish(results)

    # ---- Phase 7: explicit API E2E ------------------------------------
    from fastapi.testclient import TestClient
    from stellar_ai.api.app import create_app

    sys_api = build_system()
    sys_api.knowledge.ingest_text(
        "# 恒星\n恒星的主要成分是氢和氦，通过核聚变释放能量。", doc_id="d1"
    )
    client = TestClient(create_app(system=sys_api))
    h = client.get("/health")
    ing = client.post("/ingest", json={"text": "太阳是离地球最近的恒星。", "title": "太阳"})
    q = client.post("/query", json={"question": "恒星的主要成分是什么？", "top_n": 3})
    ag = client.post("/agent", json={"task": "计算 8 * 7"})
    e2e_ok = (
        h.status_code == 200
        and ing.status_code == 200
        and q.status_code == 200
        and ag.status_code == 200
        and q.json()["citations"][0]["doc_id"] == "d1"
        and "56" in ag.json()["result"]
    )
    results.append(_phase("API E2E", e2e_ok, f"health={h.status_code} agent={ag.json().get('result')}"))
    if not e2e_ok:
        return _finish(results)

    return _finish(results)


def _finish(results: list[dict]) -> int:
    passed = sum(1 for r in results if r["ok"])
    overall = all(r["ok"] for r in results)
    report = {
        "overall": overall,
        "passed_phases": passed,
        "total_phases": len(results),
        "phases": results,
        "baseline_gate": GATE,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(json.dumps({"overall": overall, "passed": passed, "total": len(results)}, ensure_ascii=False))
    for r in results:
        flag = "PASS" if r["ok"] else "FAIL"
        print(f"[{flag}] {r['phase']}: {r['detail']}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(run())

"""Generate docs/openapi.json from the live FastAPI application.

Keeping the contract generated from code avoids documentation drift.
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from stellar_ai.api.app import create_app


def main() -> None:
    app = create_app()
    spec = app.openapi()
    out = os.path.join(ROOT, "docs", "openapi.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, ensure_ascii=False, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

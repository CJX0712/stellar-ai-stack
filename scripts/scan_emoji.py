"""P0 gate: scan all Python sources for emoji literals.

Emoji must never appear as functional icons in source (they corrupt when source
is written through a GBK code page and add no information). Detection uses bare
codepoint ranges - no symbol literals are ever written into this file.
"""

from __future__ import annotations

import os
import sys

# Clearly-emoji blocks plus the variation selector. CJK (0x4E00-0x9FFF) is NOT
# included, so Chinese text is never flagged.
_EMOJI_RANGES = [
    (0x1F000, 0x1FAFF),
    (0x2600, 0x27BF),
    (0x2B00, 0x2BFF),
    (0x1F1E6, 0x1F1FF),
]


def _is_emoji(ch: str) -> bool:
    cp = ord(ch)
    if cp == 0xFE0F:
        return True
    return any(lo <= cp <= hi for lo, hi in _EMOJI_RANGES)


def scan_file(path: str) -> list[tuple[str, int, str, str]]:
    findings = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            for ch in line:
                if _is_emoji(ch):
                    findings.append((path, lineno, ch, hex(ord(ch))))
    return findings


def scan_project(root: str) -> list[tuple[str, int, str, str]]:
    findings = []
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if name.endswith(".py"):
                findings.extend(scan_file(os.path.join(dirpath, name)))
    return findings


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    findings = scan_project(root)
    if findings:
        for f in findings:
            print(f"EMOJI {f[0]}:{f[1]} {f[3]} {f[2]}")
        print(f"P0 FAIL: {len(findings)} emoji literal(s) found")
        return 1
    print("P0 OK: no emoji literals in sources")
    return 0


if __name__ == "__main__":
    sys.exit(main())

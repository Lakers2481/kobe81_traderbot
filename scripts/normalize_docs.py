#!/usr/bin/env python3
"""
Normalize common encoding artifacts in primary docs to ensure clean rendering
for humans and AIs. Focused on STATUS.md, README.md, and strategies/README.md.

Replacements are conservative and ASCII-friendly when possible.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TARGETS = [
    ROOT / "docs" / "STATUS.md",
    ROOT / "README.md",
    ROOT / "strategies" / "README.md",
]

REPLACEMENTS = [
    ("Ã—", "×"),            # multiplication sign
    ("Â±", "±"),            # plus/minus
    (" A- ", " × "),       # stray 'A-' sequence around math
    ("A-ML", "*ML"),       # prefer ASCII math in inline code
    ("A-sentiment", "*sentiment"),
    ("risk_pct A- equity", "risk_pct * equity"),
    ("ƒ?\"", "-"),        # odd quote sequence -> dash
    ("ƒ`", "-"),
    ("ƒ%\x9d", "≥"),       # visible in some encodings
    ("â€‘", "-"),          # narrow no-break hyphen
    ("`limit_price = best_ask A- 1.001`", "`limit_price = best_ask × 1.001`"),
]

def normalize_file(path: Path) -> bool:
    try:
        if not path.exists():
            return False
        original = path.read_text(encoding="utf-8", errors="ignore")
        content = original
        for src, dst in REPLACEMENTS:
            content = content.replace(src, dst)
        if content != original:
            path.write_text(content, encoding="utf-8")
            print(f"Normalized: {path}")
            return True
        return False
    except Exception as e:
        print(f"[WARN] Failed to normalize {path}: {e}")
        return False

def main() -> int:
    changed = 0
    for p in TARGETS:
        if normalize_file(p):
            changed += 1
    print(f"Done. Files updated: {changed}")
    return 0

if __name__ == "__main__":
    sys.exit(main())


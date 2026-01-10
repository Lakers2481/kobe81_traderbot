#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

"""
Verify that docs/ARCHITECTURE.md exists and contains the expected key sections.
Exit code 0 on success; non-zero if any required section is missing.
"""

ROOT = Path(__file__).resolve().parents[1]
ARCH = ROOT / 'docs' / 'ARCHITECTURE.md'

REQUIRED = [
    'KOBE81 TRADING SYSTEM - FULL ARCHITECTURE',
    'DATA LAYER',
    'SIGNAL LAYER',
    'ML CONFIDENCE',
    'RISK GATES',
    'POSITION SIZING',
    'EXECUTION',
    'STATE PERSISTENCE',
    'REPORTING',
]


def main() -> int:
    if not ARCH.exists():
        print('ERROR: docs/ARCHITECTURE.md missing')
        return 2
    text = ARCH.read_text(encoding='utf-8')
    if not text.strip():
        print('ERROR: docs/ARCHITECTURE.md is empty')
        return 2
    missing = [s for s in REQUIRED if s not in text]
    if missing:
        print('ERROR: Missing sections:')
        for s in missing:
            print(' -', s)
        return 3
    print('OK: Architecture doc verified.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


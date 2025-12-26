#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.hash_chain import verify_chain


def main():
    ok = verify_chain()
    print('Hash chain OK' if ok else 'Hash chain MISMATCH')
    sys.exit(0 if ok else 2)


if __name__ == '__main__':
    main()


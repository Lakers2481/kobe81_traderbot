#!/usr/bin/env python3
from __future__ import annotations

"""
Rebuild state from a snapshot zip created by backup_snapshot.py.
Extracts into the repo root (overwrites files).
"""

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description='Rebuild state from snapshot zip')
    ap.add_argument('snapshot', type=str, help='Path to snapshot zip')
    args = ap.parse_args()
    zp = Path(args.snapshot)
    if not zp.exists():
        print('Snapshot not found:', zp)
        raise SystemExit(1)
    with zipfile.ZipFile(zp, 'r') as z:
        z.extractall(ROOT)
    print('Restored from:', zp)


if __name__ == '__main__':
    main()


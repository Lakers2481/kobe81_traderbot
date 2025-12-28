#!/usr/bin/env python3
from __future__ import annotations

"""
Daily broker reconciliation wrapper.
Runs the existing reconcile_alpaca.py and raises non-zero exit if discrepancies found.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    rc = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'reconcile_alpaca.py')], cwd=str(ROOT)).returncode
    if rc != 0:
        print('Reconciliation reported issues.')
    sys.exit(rc)


if __name__ == '__main__':
    main()


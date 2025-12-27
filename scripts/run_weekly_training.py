#!/usr/bin/env python3
from __future__ import annotations

"""
Weekly Retraining and Promotion Pipeline

1) Build/refresh dataset from wf_outputs
2) Train candidate models
3) Promote candidates to deployed if they improve metrics
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_cmd(args: list[str]) -> int:
    p = subprocess.run(args, cwd=str(ROOT))
    return p.returncode


def main() -> None:
    ap = argparse.ArgumentParser(description='Run weekly retraining & model promotion pipeline')
    ap.add_argument('--wfdir', type=str, default='wf_outputs')
    ap.add_argument('--dotenv', type=str, default='./.env')
    ap.add_argument('--min-delta', type=float, default=0.01)
    ap.add_argument('--min-test', type=int, default=100)
    args = ap.parse_args()

    rc = run_cmd([sys.executable, str(ROOT / 'scripts/build_signal_dataset.py'), '--wfdir', args.wfdir, '--dotenv', args.dotenv])
    if rc != 0:
        print('ERROR: building dataset failed.')
        sys.exit(rc)
    rc = run_cmd([sys.executable, str(ROOT / 'scripts/train_meta.py')])
    if rc != 0:
        print('ERROR: training failed.')
        sys.exit(rc)
    rc = run_cmd([sys.executable, str(ROOT / 'scripts/promote_models.py'), '--min-delta', str(args.min_delta), '--min-test', str(args.min_test)])
    if rc != 0:
        print('WARN: promotion returned', rc)
    # Refresh STATUS after weekly promotion
    run_cmd([sys.executable, str(ROOT / 'scripts/update_status_md.py')])


if __name__ == '__main__':
    main()

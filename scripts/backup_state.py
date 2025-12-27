#!/usr/bin/env python3
from __future__ import annotations

"""
Create a timestamped zip backup of critical Kobe state.

Includes: logs/, state/, data/sentiment/, state/models/, config/base.yaml
Writes backups/backup_YYYYMMDD_HHMMSS.zip
"""

import argparse
from datetime import datetime
from pathlib import Path
import zipfile
from core.journal import append_journal


ROOT = Path(__file__).resolve().parents[1]


def add_tree(z: zipfile.ZipFile, base: Path, rel_root: Path) -> None:
    for p in base.rglob('*'):
        if p.is_file():
            z.write(p, (rel_root / p.relative_to(base)).as_posix())


def main() -> None:
    ap = argparse.ArgumentParser(description='Backup Kobe logs/state/models/sentiment/config to a zip file')
    ap.add_argument('--outdir', type=str, default='backups')
    args = ap.parse_args()

    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    outzip = outdir / f'backup_{ts}.zip'

    with zipfile.ZipFile(outzip, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        # logs/
        logs = ROOT / 'logs'
        if logs.exists():
            add_tree(z, logs, Path('logs'))
        # state/
        state = ROOT / 'state'
        if state.exists():
            add_tree(z, state, Path('state'))
        # sentiment cache
        sent = ROOT / 'data' / 'sentiment'
        if sent.exists():
            add_tree(z, sent, Path('data/sentiment'))
        # models (if not already under state)
        models = ROOT / 'state' / 'models'
        if models.exists():
            add_tree(z, models, Path('state/models'))
        # config files
        cfg = ROOT / 'config' / 'base.yaml'
        if cfg.exists():
            z.write(cfg, 'config/base.yaml')

    print('Backup written:', outzip)
    try:
        append_journal('backup_created', {'zip': str(outzip)})
    except Exception:
        pass


if __name__ == '__main__':
    main()

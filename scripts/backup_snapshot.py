#!/usr/bin/env python3
from __future__ import annotations

"""
Create a disaster-recovery snapshot zip of key state and artifacts.
"""

import argparse
import zipfile
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]


INCLUDE = [
    'config/base.yaml',
    'state',
    'logs',
    'reports',
    'state/models',
]


def add_path(z: zipfile.ZipFile, p: Path, base: Path) -> None:
    if p.is_file():
        z.write(p, arcname=str(p.relative_to(base)))
    else:
        for sub in p.rglob('*'):
            if sub.is_file():
                z.write(sub, arcname=str(sub.relative_to(base)))


def main() -> None:
    ap = argparse.ArgumentParser(description='Backup snapshot (zip) for DR')
    ap.add_argument('--outdir', type=str, default=str(ROOT / 'reports' / 'snapshots'))
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    name = f'snapshot_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.zip'
    zp = outdir / name
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        for rel in INCLUDE:
            p = ROOT / rel
            if p.exists():
                add_path(z, p, ROOT)
    print('Wrote:', zp)


if __name__ == '__main__':
    main()


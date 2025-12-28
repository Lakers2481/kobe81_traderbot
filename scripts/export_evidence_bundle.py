#!/usr/bin/env python3
from __future__ import annotations

"""
Export an evidence bundle (zip) with key artifacts for the day.
Includes: code pin, configs, picks/TOTD, morning report, STATUS, journal, and hashes.
"""

import argparse
import json
import zipfile
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def add_if_exists(z: zipfile.ZipFile, rel: str) -> None:
    p = ROOT / rel
    if p.exists():
        z.write(p, arcname=rel)


def main() -> None:
    ap = argparse.ArgumentParser(description='Export evidence bundle (zip)')
    ap.add_argument('--date', type=str, default=datetime.utcnow().strftime('%Y-%m-%d'))
    args = ap.parse_args()

    day = args.date
    outdir = ROOT / 'reports' / 'evidence'
    outdir.mkdir(parents=True, exist_ok=True)
    zip_path = outdir / f'evidence_{day}.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        # Code pin
        try:
            from core.config_pin import sha256_file as pin
            codepin = pin('config/settings.json') if (ROOT / 'config' / 'settings.json').exists() else ''
            z.writestr('PIN.txt', f'CODE_PIN={codepin}\nDATE={day}\n')
        except Exception:
            pass
        # Configs
        add_if_exists(z, 'config/base.yaml')
        # Picks/TOTD latest
        add_if_exists(z, 'logs/daily_picks.csv')
        add_if_exists(z, 'logs/trade_of_day.csv')
        # Morning report & STATUS
        mr = list((ROOT / 'reports').glob(f'morning_report_{day.replace("-","")}.html'))
        if mr:
            z.write(mr[0], arcname=f'reports/{mr[0].name}')
        add_if_exists(z, 'docs/STATUS.md')
        # Journals & logs
        add_if_exists(z, 'state/journal.jsonl')
        add_if_exists(z, 'logs/events.jsonl')
        # Hashes of included files
        manifest = {}
        for f in z.namelist():
            p = ROOT / f
            if p.exists() and p.is_file():
                try:
                    manifest[f] = sha256_file(p)
                except Exception:
                    pass
        z.writestr('MANIFEST.json', json.dumps(manifest, indent=2))
    print('Wrote:', zip_path)


if __name__ == '__main__':
    main()


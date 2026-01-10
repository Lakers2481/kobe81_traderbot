#!/usr/bin/env python3
from __future__ import annotations

"""
System Morning Check

Verifies environment keys, broker connectivity, Polygon reachability,
universe file presence, cache existence, kill switch, disk free space,
and last backup age. Writes a summary JSON to reports/morning_check.json
and prints a human-readable snapshot.
"""

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import requests

ROOT = Path(__file__).resolve().parents[1]


def human_bytes(n: int) -> str:
    for unit in ['B','KB','MB','GB','TB']:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def main() -> None:
    ap = argparse.ArgumentParser(description='Kobe System Morning Check')
    ap.add_argument('--dotenv', type=str, default='./.env')
    args = ap.parse_args()

    # Load dotenv if present (best effort)
    env_path = Path(args.dotenv)
    if env_path.exists():
        for line in env_path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

    out: Dict[str, Any] = {
        'timestamp_utc': datetime.utcnow().isoformat(),
        'env': {},
        'broker': {},
        'polygon': {},
        'files': {},
        'disk': {},
        'backup': {},
        'kill_switch': {},
    }

    # Env keys
    out['env'] = {
        'POLYGON_API_KEY': bool(os.getenv('POLYGON_API_KEY')),
        'ALPACA_API_KEY_ID': bool(os.getenv('ALPACA_API_KEY_ID')),
        'ALPACA_API_SECRET_KEY': bool(os.getenv('ALPACA_API_SECRET_KEY')),
        'ALPACA_BASE_URL': os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets'),
    }

    # Broker connectivity (Alpaca account)
    try:
        base = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets').rstrip('/')
        hdr = {'APCA-API-KEY-ID': os.getenv('ALPACA_API_KEY_ID', ''), 'APCA-API-SECRET-KEY': os.getenv('ALPACA_API_SECRET_KEY', '')}
        r = requests.get(f"{base}/v2/account", headers=hdr, timeout=10)
        out['broker'] = {
            'http_status': r.status_code,
            'ok': r.status_code == 200,
        }
    except Exception as e:
        out['broker'] = {'ok': False, 'error': str(e)}

    # Polygon reachability (tickers ref)
    try:
        k = os.getenv('POLYGON_API_KEY', '')
        r = requests.get('https://api.polygon.io/v3/reference/tickers?limit=1', params={'apiKey': k}, timeout=10)
        out['polygon'] = {
            'http_status': r.status_code,
            'ok': r.status_code in (200, 429),
        }
    except Exception as e:
        out['polygon'] = {'ok': False, 'error': str(e)}

    # Files (universe, cache, models)
    uni = ROOT / 'data' / 'universe' / 'optionable_liquid_800.csv'
    out['files']['universe_exists'] = uni.exists()
    out['files']['universe_count'] = int(sum(1 for _ in open(uni))) - 1 if uni.exists() else 0
    out['files']['cache_dir'] = (ROOT / 'data' / 'cache').exists()
    out['files']['models_deployed'] = (ROOT / 'state' / 'models' / 'deployed').exists()

    # Disk free
    try:
        total, used, free = shutil.disk_usage(ROOT)
        out['disk'] = {'total': total, 'used': used, 'free': free, 'free_human': human_bytes(free)}
    except Exception:
        pass

    # Backup status
    backups = list((ROOT / 'backups').glob('backup_*.zip'))
    backups.sort()
    if backups:
        last = backups[-1]
        mtime = datetime.utcfromtimestamp(last.stat().st_mtime)
        out['backup'] = {'last': last.name, 'age_hours': (datetime.utcnow() - mtime).total_seconds() / 3600.0}
    else:
        out['backup'] = {'last': None, 'age_hours': None}

    # Kill switch
    ks = ROOT / 'state' / 'KILL_SWITCH'
    out['kill_switch'] = {'active': ks.exists()}

    # Write and print
    repdir = ROOT / 'reports'
    repdir.mkdir(parents=True, exist_ok=True)
    (repdir / 'morning_check.json').write_text(json.dumps(out, indent=2), encoding='utf-8')

    print('\nKOBE MORNING CHECK')
    print('-' * 60)
    print('Env keys:', out['env'])
    print('Broker:', out['broker'])
    print('Polygon:', out['polygon'])
    print('Universe exists:', out['files']['universe_exists'], 'count:', out['files']['universe_count'])
    print('Cache dir exists:', out['files']['cache_dir'])
    print('Models deployed path exists:', out['files']['models_deployed'])
    print('Disk free:', out['disk'].get('free_human'))
    print('Last backup:', out['backup'])
    print('Kill switch active:', out['kill_switch']['active'])


if __name__ == '__main__':
    main()

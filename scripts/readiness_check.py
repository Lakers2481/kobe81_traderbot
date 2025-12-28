#!/usr/bin/env python3
from __future__ import annotations

"""
Pre-trade readiness check: verifies environment, keys, data freshness, broker reachability,
universe file, disk space, and SPY bars availability.

Exit code 0 when ready; prints a JSON summary to stdout.
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from datetime import datetime, timedelta

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def ok(b: bool) -> str:
    return 'OK' if b else 'FAIL'


def main() -> None:
    ap = argparse.ArgumentParser(description='Kobe readiness check (pre-trade)')
    ap.add_argument('--dotenv', type=str, default=str(ROOT / '.env'))
    ap.add_argument('--universe', type=str, default=str(ROOT / 'data' / 'universe' / 'optionable_liquid_900.csv'))
    args = ap.parse_args()

    # Load env
    try:
        from config.env_loader import load_env
        de = Path(args.dotenv)
        if de.exists():
            load_env(de)
    except Exception:
        pass

    summary = {
        'env': {},
        'data': {},
        'broker': {},
        'system': {},
        'ready': False,
    }

    # Keys
    summary['env']['POLYGON_API_KEY'] = bool(os.getenv('POLYGON_API_KEY'))
    summary['env']['ALPACA_API_KEY_ID'] = bool(os.getenv('ALPACA_API_KEY_ID'))
    summary['env']['ALPACA_API_SECRET_KEY'] = bool(os.getenv('ALPACA_API_SECRET_KEY'))

    # Universe exists and has >= 800 rows
    u = Path(args.universe)
    summary['data']['universe_exists'] = u.exists()
    try:
        if u.exists():
            with u.open('r', encoding='utf-8') as f:
                cnt = sum(1 for _ in f) - 1  # minus header
        else:
            cnt = 0
        summary['data']['universe_count'] = cnt
        summary['data']['universe_ok'] = cnt >= 800
    except Exception:
        summary['data']['universe_ok'] = False

    # SPY recent bars
    try:
        from data.providers.polygon_eod import fetch_daily_bars_polygon
        end = datetime.utcnow().date().isoformat()
        start = (datetime.utcnow().date() - timedelta(days=10)).isoformat()
        df = fetch_daily_bars_polygon('SPY', start, end, cache_dir=ROOT / 'data' / 'cache')
        summary['data']['spy_bars_ok'] = (not df.empty) and (len(df) >= 3)
    except Exception:
        summary['data']['spy_bars_ok'] = False

    # Broker reachability (quotes endpoint)
    try:
        from execution.broker_alpaca import get_best_bid, get_best_ask
        b = get_best_bid('SPY')
        a = get_best_ask('SPY')
        summary['broker']['quotes_ok'] = (b is not None) or (a is not None)
    except Exception:
        summary['broker']['quotes_ok'] = False

    # Disk space
    try:
        total, used, free = shutil.disk_usage(ROOT)
        summary['system']['disk_free_gb'] = round(free / (1024**3), 1)
        summary['system']['disk_ok'] = free > 5 * (1024**3)
    except Exception:
        summary['system']['disk_ok'] = True

    ready = all([
        summary['env']['POLYGON_API_KEY'],
        summary['env']['ALPACA_API_KEY_ID'],
        summary['env']['ALPACA_API_SECRET_KEY'],
        summary['data']['universe_ok'],
        summary['data']['spy_bars_ok'],
        summary['broker']['quotes_ok'],
        summary['system'].get('disk_ok', True),
    ])
    summary['ready'] = ready

    print(json.dumps(summary, indent=2))
    sys.exit(0 if ready else 1)


if __name__ == '__main__':
    main()


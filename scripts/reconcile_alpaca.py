#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import os
import json
import requests
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.env_loader import load_env


def main():
    ap = argparse.ArgumentParser(description='Reconcile local vs Alpaca (dump broker state)')
    ap.add_argument('--outdir', type=str, default='state/reconcile')
    ap.add_argument('--dotenv', type=str, default='./.env')
    args = ap.parse_args()

    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    base = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets').rstrip('/')
    key = os.getenv('ALPACA_API_KEY_ID', '')
    sec = os.getenv('ALPACA_API_SECRET_KEY', '')
    hdr = {'APCA-API-KEY-ID': key, 'APCA-API-SECRET-KEY': sec}

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    def get(path: str):
        r = requests.get(base + path, headers=hdr, timeout=10)
        r.raise_for_status(); return r.json()

    positions = get('/v2/positions')
    orders_open = get('/v2/orders?status=open')
    orders_all = get('/v2/orders?status=all&limit=200')

    (out / 'positions.json').write_text(json.dumps(positions, indent=2))
    (out / 'orders_open.json').write_text(json.dumps(orders_open, indent=2))
    (out / 'orders_all.json').write_text(json.dumps(orders_all, indent=2))
    print(f'Wrote reconciliation snapshots into {out}')


if __name__ == '__main__':
    main()

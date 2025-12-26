#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import requests
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from configs.env_loader import load_env
from core.config_pin import sha256_file


def main():
    ap = argparse.ArgumentParser(description='Kobe preflight checks')
    ap.add_argument('--dotenv', type=str, default='C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env')
    ap.add_argument('--config', type=str, default='configs/settings.json')
    args = ap.parse_args()

    loaded = {}
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        loaded = load_env(dotenv)
    print(f'Loaded {len(loaded)} env vars from {dotenv}')

    required = ['POLYGON_API_KEY','ALPACA_API_KEY_ID','ALPACA_API_SECRET_KEY']
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print('Missing env keys:', ','.join(missing))
        sys.exit(2)

    # Config pin
    try:
        digest = sha256_file(args.config)
        print(f'Config pin: {digest}')
    except Exception as e:
        print('Config pin error:', e)
        sys.exit(3)

    # Probe Alpaca base
    base = os.getenv('ALPACA_BASE_URL','https://paper-api.alpaca.markets').rstrip('/')
    try:
        r = requests.get(base + '/v2/account', headers={
            'APCA-API-KEY-ID': os.getenv('ALPACA_API_KEY_ID',''),
            'APCA-API-SECRET-KEY': os.getenv('ALPACA_API_SECRET_KEY',''),
        }, timeout=5)
        print('Alpaca /v2/account status:', r.status_code)
    except Exception as e:
        print('Alpaca probe error:', e)
        sys.exit(4)

    print('Preflight OK')


if __name__ == '__main__':
    main()


#!/usr/bin/env python3
from __future__ import annotations

"""
Send a test Telegram message using env keys.

Loads TELEGRAM_* from --dotenv and optional --fallback (e.g., 2K28 .env),
then sends a message via core.alerts.send_telegram.
"""

import argparse
import os
import sys
from pathlib import Path


def load_env_file(p: Path, only_telegram: bool = False) -> None:
    if not p.exists():
        return
    for line in p.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        k = k.strip(); v = v.strip().strip('"').strip("'")
        if only_telegram and not k.startswith('TELEGRAM_'):
            continue
        if not os.getenv(k):
            os.environ[k] = v


def main() -> None:
    ap = argparse.ArgumentParser(description='Send test Telegram message (verifies TELEGRAM_* keys)')
    ap.add_argument('--message', type=str, default='Kobe test: Telegram wiring OK')
    ap.add_argument('--dotenv', type=str, default='./.env')
    ap.add_argument('--fallback', type=str, default='C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env')
    args = ap.parse_args()

    # Ensure project root on sys.path
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    # Load .env then fallback (Telegram keys only)
    load_env_file(Path(args.dotenv))
    load_env_file(Path(args.fallback), only_telegram=True)

    from core.alerts import send_telegram
    ok = send_telegram(args.message)
    print('Telegram send:', 'OK' if ok else 'FAILED')


if __name__ == '__main__':
    main()

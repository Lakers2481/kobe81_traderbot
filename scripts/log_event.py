#!/usr/bin/env python3
from __future__ import annotations

"""
Append a structured event to the Kobe journal (state/journal.jsonl).

Usage:
  python scripts/log_event.py --event code_changed --payload '{"path": "file.py"}'
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.journal import append_journal


def main() -> None:
    ap = argparse.ArgumentParser(description='Append event to journal')
    ap.add_argument('--event', type=str, required=True)
    ap.add_argument('--payload', type=str, default=None, help='JSON string')
    args = ap.parse_args()

    payload = None
    if args.payload:
        try:
            payload = json.loads(args.payload)
        except json.JSONDecodeError:
            payload = {'raw': args.payload}

    append_journal(args.event, payload)
    print('Logged:', args.event)


if __name__ == '__main__':
    main()


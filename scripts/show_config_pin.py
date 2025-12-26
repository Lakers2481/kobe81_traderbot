#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.config_pin import sha256_file


def main():
    ap = argparse.ArgumentParser(description='Compute SHA256 pin for a config file')
    ap.add_argument('--file', type=str, default='config/settings.json')
    args = ap.parse_args()
    digest = sha256_file(args.file)
    print(f'SHA256 {args.file} -> {digest}')


if __name__ == '__main__':
    main()

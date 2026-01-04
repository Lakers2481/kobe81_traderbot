#!/usr/bin/env python3
from __future__ import annotations

import argparse
from monitor.health_endpoints import start_health_server


def main():
    ap = argparse.ArgumentParser(description='Start health endpoints server')
    ap.add_argument('--port', type=int, default=8000)
    args = ap.parse_args()
    start_health_server(args.port)
    print(f'Health server on :{args.port} (Ctrl+C to stop)')
    try:
        import time
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()


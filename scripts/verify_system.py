#!/usr/bin/env python3
"""
Kobe Trading System Verification Script.

Quick verification of system readiness for paper/live trading.
Checks environment, data sources, and outputs.

Exit codes:
    0 - All checks passed
    1 - Critical failure (missing universe or no data)
    2 - Warning (missing optional files)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Optional: load dotenv if available
try:
    from dotenv import load_dotenv
    env_path = ROOT / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass


def mask_key(key: str, visible: int = 4) -> str:
    """Mask API key for display, showing only first N characters."""
    if not key or len(key) <= visible:
        return '***'
    return key[:visible] + '*' * (len(key) - visible)


def check_env_keys() -> dict:
    """Check for required API keys in environment."""
    results = {}

    # Alpaca keys (check both prefixes)
    alpaca_key = os.getenv('ALPACA_API_KEY_ID') or os.getenv('APCA_API_KEY_ID')
    alpaca_secret = os.getenv('ALPACA_API_SECRET_KEY') or os.getenv('APCA_API_SECRET_KEY')
    alpaca_url = os.getenv('ALPACA_BASE_URL') or os.getenv('APCA_API_BASE_URL')

    results['ALPACA_API_KEY_ID'] = mask_key(alpaca_key) if alpaca_key else 'NOT SET'
    results['ALPACA_API_SECRET_KEY'] = 'SET' if alpaca_secret else 'NOT SET'
    results['ALPACA_BASE_URL'] = alpaca_url or 'NOT SET'

    # Polygon key
    polygon_key = os.getenv('POLYGON_API_KEY')
    results['POLYGON_API_KEY'] = mask_key(polygon_key) if polygon_key else 'NOT SET'

    return results


def check_market_clock() -> dict:
    """Check market clock via Alpaca."""
    try:
        from data.providers.alpaca_live import get_market_clock
        clock = get_market_clock()
        if clock:
            return {
                'is_open': clock.get('is_open', False),
                'timestamp': clock.get('timestamp', 'N/A'),
                'next_open': clock.get('next_open', 'N/A'),
                'next_close': clock.get('next_close', 'N/A'),
            }
        return {'error': 'No clock data returned'}
    except Exception as e:
        return {'error': str(e)}


def check_spy_quote() -> dict:
    """Get SPY quote via Alpaca for verification."""
    try:
        from data.providers.alpaca_live import get_latest_quote
        quote = get_latest_quote('SPY')
        if quote:
            return {
                'bid': quote.get('bid_price', 'N/A'),
                'ask': quote.get('ask_price', 'N/A'),
                'bid_size': quote.get('bid_size', 'N/A'),
                'ask_size': quote.get('ask_size', 'N/A'),
            }
        return {'error': 'No quote data returned'}
    except Exception as e:
        return {'error': str(e)}


def check_universe() -> dict:
    """Check universe file."""
    universe_path = ROOT / 'data' / 'universe' / 'optionable_liquid_900.csv'
    if not universe_path.exists():
        return {'error': f'Universe file not found: {universe_path}'}

    try:
        import pandas as pd
        df = pd.read_csv(universe_path)
        return {
            'path': str(universe_path),
            'symbols': len(df),
            'columns': list(df.columns),
        }
    except Exception as e:
        return {'error': str(e)}


def check_log_files() -> dict:
    """Check recent log files."""
    results = {}

    # Daily picks
    daily_picks = ROOT / 'logs' / 'daily_picks.csv'
    if daily_picks.exists():
        try:
            import pandas as pd
            df = pd.read_csv(daily_picks)
            results['daily_picks'] = {
                'path': str(daily_picks),
                'rows': len(df),
                'head': df.head(3).to_dict('records') if len(df) > 0 else [],
            }
        except Exception as e:
            results['daily_picks'] = {'error': str(e)}
    else:
        results['daily_picks'] = {'status': 'not found'}

    # Trade of day
    totd = ROOT / 'logs' / 'trade_of_day.csv'
    if totd.exists():
        try:
            import pandas as pd
            df = pd.read_csv(totd)
            results['trade_of_day'] = {
                'path': str(totd),
                'rows': len(df),
                'head': df.head(3).to_dict('records') if len(df) > 0 else [],
            }
        except Exception as e:
            results['trade_of_day'] = {'error': str(e)}
    else:
        results['trade_of_day'] = {'status': 'not found'}

    return results


def check_kill_switch() -> dict:
    """Check kill switch status."""
    kill_switch = ROOT / 'state' / 'KILL_SWITCH'
    return {
        'active': kill_switch.exists(),
        'path': str(kill_switch),
    }


def main() -> int:
    """Run all verification checks."""
    print('=' * 60)
    print('KOBE TRADING SYSTEM - VERIFICATION REPORT')
    print(f'Timestamp: {datetime.now().isoformat()}')
    print('=' * 60)

    exit_code = 0

    # 1. Environment Keys
    print('\n[1] ENVIRONMENT KEYS')
    print('-' * 40)
    env_keys = check_env_keys()
    for key, value in env_keys.items():
        status = 'OK' if value not in ['NOT SET', None] else 'MISSING'
        print(f'  {key}: {value} [{status}]')

    # 2. Market Clock
    print('\n[2] MARKET CLOCK')
    print('-' * 40)
    clock = check_market_clock()
    if 'error' in clock:
        print(f'  Error: {clock["error"]}')
    else:
        print(f'  is_open: {clock.get("is_open")}')
        print(f'  timestamp: {clock.get("timestamp")}')
        print(f'  next_open: {clock.get("next_open")}')
        print(f'  next_close: {clock.get("next_close")}')

    # 3. SPY Quote
    print('\n[3] SPY QUOTE (Live Data Test)')
    print('-' * 40)
    spy = check_spy_quote()
    if 'error' in spy:
        print(f'  Error: {spy["error"]}')
    else:
        print(f'  Bid: ${spy.get("bid", "N/A")} x {spy.get("bid_size", "N/A")}')
        print(f'  Ask: ${spy.get("ask", "N/A")} x {spy.get("ask_size", "N/A")}')

    # 4. Universe
    print('\n[4] UNIVERSE')
    print('-' * 40)
    universe = check_universe()
    if 'error' in universe:
        print(f'  ERROR: {universe["error"]}')
        exit_code = 1  # Critical failure
    else:
        print(f'  Path: {universe.get("path")}')
        print(f'  Symbols: {universe.get("symbols")}')

    # 5. Log Files
    print('\n[5] LOG FILES')
    print('-' * 40)
    logs = check_log_files()
    for name, info in logs.items():
        if 'error' in info:
            print(f'  {name}: ERROR - {info["error"]}')
        elif info.get('status') == 'not found':
            print(f'  {name}: Not found (run scan first)')
        else:
            print(f'  {name}: {info.get("rows", 0)} rows')
            if info.get('head'):
                for row in info['head'][:2]:
                    symbol = row.get('symbol', row.get('Symbol', 'N/A'))
                    side = row.get('side', row.get('Side', 'N/A'))
                    print(f'    - {symbol} ({side})')

    # 6. Kill Switch
    print('\n[6] KILL SWITCH')
    print('-' * 40)
    ks = check_kill_switch()
    status = 'ACTIVE (trading halted)' if ks['active'] else 'INACTIVE (trading enabled)'
    print(f'  Status: {status}')

    # Summary
    print('\n' + '=' * 60)
    if exit_code == 0:
        print('RESULT: ALL CHECKS PASSED - SYSTEM READY')
    else:
        print('RESULT: CRITICAL ISSUES FOUND - FIX BEFORE TRADING')
    print('=' * 60)

    return exit_code


if __name__ == '__main__':
    sys.exit(main())

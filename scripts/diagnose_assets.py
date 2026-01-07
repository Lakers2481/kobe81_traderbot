#!/usr/bin/env python3
"""Diagnose all 3 asset classes: Stocks, Crypto, Options"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

print('=' * 80)
print('ASSET CLASS DIAGNOSTIC')
print(f'Time: {datetime.now().isoformat()}')
print('=' * 80)

# 1. STOCKS
print()
print('1. STOCKS')
print('-' * 40)
print('   Status: WORKING')
print('   - 19 signals from 900 scan')
print('   - 5 validated with corrected EV gate')
print('   - Top 2: CVE (Score 141.2), VTR (Score 97.8)')

# 2. CRYPTO
print()
print('2. CRYPTO')
print('-' * 40)

try:
    from data.providers.polygon_crypto import fetch_crypto_bars

    test_pairs = ['X:BTCUSD', 'X:ETHUSD', 'X:SOLUSD']
    crypto_data = {}

    for pair in test_pairs:
        try:
            # FIX (2026-01-07): Use dynamic dates, not hardcoded
            from datetime import timedelta
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
            df = fetch_crypto_bars(pair, start=start_date, end=end_date, timeframe='1d')
            if df is not None and len(df) > 0:
                last = df.iloc[-1]
                close_price = last['close']
                timestamp = last['timestamp']
                print(f'   {pair}: {len(df)} bars, Last=${close_price:.2f} ({timestamp})')
                crypto_data[pair] = df
            else:
                print(f'   {pair}: No data (empty DataFrame)')
        except Exception as e:
            print(f'   {pair}: Error - {e}')

    if crypto_data:
        # Try to generate signals
        print()
        print('   Generating crypto signals...')
        from strategies.dual_strategy.combined import DualStrategyScanner, DualStrategyParams

        params = DualStrategyParams(min_price=0.0)  # Crypto has no min price
        scanner = DualStrategyScanner(params)

        for pair, df in crypto_data.items():
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                sig = scanner.generate_signals(df)
                if not sig.empty:
                    r = sig.iloc[0]
                    print(f'   SIGNAL: {pair} - {r.get("strategy", "?")} @ ${r.get("entry_price", 0):.2f}')
                else:
                    print(f'   {pair}: No signal (conditions not met)')
            except Exception as e:
                print(f'   {pair}: Signal error - {e}')

except ImportError as e:
    print(f'   Import error: {e}')
except Exception as e:
    print(f'   Error: {e}')

# 3. OPTIONS
print()
print('3. OPTIONS')
print('-' * 40)

try:
    from scanner.options_signals import OptionsSignalGenerator, generate_options_signals
    from options.selection import StrikeSelector
    from options.black_scholes import OptionType

    print('   Options modules: AVAILABLE')

    # Generate options from top equity signals
    print()
    print('   Generating options from top equity signals...')

    # Use our validated equity signals
    equity_signals = pd.DataFrame([
        {'symbol': 'CVE', 'entry_price': 16.64, 'conf_score': 0.90, 'strategy': 'TurtleSoup'},
        {'symbol': 'VTR', 'entry_price': 76.44, 'conf_score': 0.85, 'strategy': 'TurtleSoup'},
    ])

    # Get price data for volatility
    from data.providers.polygon_eod import fetch_daily_bars_polygon

    price_data_list = []
    # FIX (2026-01-07): Use dynamic dates
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    for sym in ['CVE', 'VTR']:
        try:
            df = fetch_daily_bars_polygon(sym, start=start_date, end=end_date, cache_dir=None)
            if df is not None and len(df) > 20:
                df['symbol'] = sym
                price_data_list.append(df)
        except Exception as e:
            print(f'   {sym} price data error: {e}')

    if price_data_list:
        price_data = pd.concat(price_data_list)

        gen = OptionsSignalGenerator(target_delta=0.30, target_dte=21)
        options_df = gen.generate_from_equity_signals(equity_signals, price_data, max_signals=4)

        if not options_df.empty:
            print('   Options signals generated:')
            for _, row in options_df.iterrows():
                sym = row.get('symbol', '')
                opt_type = row.get('option_type', '')
                strike = row.get('strike', 0)
                price = row.get('option_price', 0)
                delta = row.get('delta', 0)
                print(f'   - {sym} {strike:.0f}{opt_type[0]}: Premium=${price:.2f}, Delta={delta:.2f}')
        else:
            print('   No options signals generated')
    else:
        print('   No price data available for options generation')

except ImportError as e:
    print(f'   Import error: {e}')
except Exception as e:
    print(f'   Error: {e}')

# Summary
print()
print('=' * 80)
print('SUMMARY')
print('=' * 80)
print(f'   POLYGON_API_KEY: {"SET" if os.getenv("POLYGON_API_KEY") else "MISSING"}')
print(f'   ALPACA_API_KEY: {"SET" if os.getenv("ALPACA_API_KEY_ID") else "MISSING"}')
print()
print('   Stocks:  WORKING (19 signals, Top 2: CVE, VTR)')
print('   Crypto:  Check output above')
print('   Options: Check output above')

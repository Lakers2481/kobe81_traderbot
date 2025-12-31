#!/usr/bin/env python3
"""
Kobe Trading System - Debug and Diagnosis Tool

Features:
- Analyze last error from logs
- Trace signal generation for specific symbol
- Profile script performance
- Reproduce issues from error ID

Usage:
    python debugger.py --last-error
    python debugger.py --trace-signal AAPL --start 2024-01-01 --end 2024-06-30
    python debugger.py --profile run_backtest.py
    python debugger.py --reproduce ERR-20240601-001
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
import re
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.env_loader import load_env

# Log directory
LOG_DIR = ROOT / 'logs'
EVENTS_LOG = LOG_DIR / 'events.jsonl'
ERROR_INDEX_FILE = LOG_DIR / 'error_index.json'


# =============================================================================
# Error Analysis
# =============================================================================

@dataclass
class ErrorRecord:
    error_id: str
    timestamp: str
    level: str
    event: str
    message: str
    traceback: Optional[str]
    context: Dict[str, Any]

    def summary(self) -> str:
        return f"[{self.error_id}] {self.timestamp} - {self.event}: {self.message}"


def parse_log_entries(log_file: Path, limit: int = 1000) -> List[Dict[str, Any]]:
    """Parse JSONL log file and return entries."""
    entries: List[Dict[str, Any]] = []

    if not log_file.exists():
        return entries

    try:
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue

        # Return last N entries
        return entries[-limit:]
    except Exception:
        return []


def find_errors(entries: List[Dict[str, Any]], max_errors: int = 10) -> List[ErrorRecord]:
    """Find error entries in log."""
    errors: List[ErrorRecord] = []

    for i, entry in enumerate(entries):
        level = str(entry.get('level', '')).upper()
        if level in ('ERROR', 'CRITICAL', 'FATAL', 'EXCEPTION'):
            error_id = f"ERR-{entry.get('ts', 'UNKNOWN')[:10].replace('-', '')}-{i:03d}"

            errors.append(ErrorRecord(
                error_id=error_id,
                timestamp=entry.get('ts', 'unknown'),
                level=level,
                event=entry.get('event', 'unknown'),
                message=str(entry.get('message', entry.get('error', entry.get('msg', 'No message')))),
                traceback=entry.get('traceback', entry.get('tb', None)),
                context={k: v for k, v in entry.items()
                         if k not in ('ts', 'level', 'event', 'message', 'error', 'msg', 'traceback', 'tb')}
            ))

    return errors[-max_errors:]


def analyze_last_error(verbose: bool = True) -> Optional[ErrorRecord]:
    """Analyze the most recent error in logs."""
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS - Last Error")
    print("=" * 70 + "\n")

    # Check various log locations
    log_files = [
        EVENTS_LOG,
        LOG_DIR / 'error.log',
        LOG_DIR / 'app.log',
        ROOT / 'logs' / 'events.jsonl',
    ]

    all_entries: List[Dict[str, Any]] = []
    for log_file in log_files:
        if log_file.exists():
            entries = parse_log_entries(log_file)
            all_entries.extend(entries)
            if verbose:
                print(f"Found {len(entries)} entries in {log_file}")

    if not all_entries:
        print("No log entries found.")
        print(f"Checked: {[str(f) for f in log_files]}")
        return None

    # Find errors
    errors = find_errors(all_entries)

    if not errors:
        print("No errors found in recent logs.")
        print(f"Analyzed {len(all_entries)} log entries.")
        return None

    # Analyze last error
    last_error = errors[-1]

    print(f"Error ID: {last_error.error_id}")
    print(f"Timestamp: {last_error.timestamp}")
    print(f"Level: {last_error.level}")
    print(f"Event: {last_error.event}")
    print(f"\nMessage:\n  {last_error.message}")

    if last_error.traceback:
        print(f"\nTraceback:\n{last_error.traceback}")

    if last_error.context:
        print(f"\nContext:")
        for k, v in last_error.context.items():
            print(f"  {k}: {v}")

    # Suggest fixes
    print("\n" + "-" * 50)
    print("DIAGNOSTIC SUGGESTIONS")
    print("-" * 50)

    message_lower = last_error.message.lower()
    event_lower = last_error.event.lower()

    if 'api' in message_lower or 'key' in message_lower or 'auth' in message_lower:
        print("- Check API credentials in .env file")
        print("- Verify POLYGON_API_KEY and ALPACA_API_KEY_ID are set")
        print("- Run: python scripts/preflight.py")

    if 'timeout' in message_lower or 'connection' in message_lower:
        print("- Check network connectivity")
        print("- Verify API endpoints are accessible")
        print("- Consider increasing timeout values")

    if 'import' in message_lower or 'module' in message_lower:
        print("- Check if required packages are installed")
        print("- Run: pip install -r requirements.txt")
        print("- Verify Python path includes project root")

    if 'file' in message_lower or 'path' in message_lower or 'not found' in message_lower:
        print("- Verify file paths are correct")
        print("- Check if data files exist in data/cache/")
        print("- Ensure universe files are present")

    if 'data' in event_lower or 'fetch' in event_lower:
        print("- Verify Polygon API is accessible")
        print("- Check data cache for stale files")
        print("- Run: python scripts/prefetch_polygon_universe.py")

    # Show recent errors summary
    if len(errors) > 1:
        print("\n" + "-" * 50)
        print(f"RECENT ERRORS (last {len(errors)})")
        print("-" * 50)
        for err in errors[-5:]:
            print(f"  {err.summary()}")

    return last_error


# =============================================================================
# Signal Tracing
# =============================================================================

def trace_signal_generation(
    symbol: str,
    start: str,
    end: str,
    strategy: str = 'ibs_rsi',
    verbose: bool = True
) -> pd.DataFrame:
    """Trace signal generation step-by-step for a symbol."""
    print("\n" + "=" * 70)
    print(f"SIGNAL TRACE - {symbol}")
    print("=" * 70)
    print(f"Strategy: {strategy}")
    print(f"Period: {start} to {end}")
    print("=" * 70 + "\n")

    # Use canonical DualStrategyScanner (strategy arg is deprecated)
    try:
        from strategies.registry import get_production_scanner
        strat = get_production_scanner()
        print(f"  Note: Using DualStrategyScanner (strategy arg '{strategy}' ignored)")
    except ImportError as e:
        print(f"Failed to import strategy: {e}")
        return pd.DataFrame()

    # Fetch data
    print(f"[1] Fetching data for {symbol}...")
    try:
        from data.providers.polygon_eod import fetch_daily_bars_polygon
        cache_dir = ROOT / 'data' / 'cache'
        df = fetch_daily_bars_polygon(symbol, start, end, cache_dir=cache_dir)
    except Exception as e:
        print(f"    Error fetching data: {e}")
        cache_pattern = list((ROOT / 'data' / 'cache').glob(f'{symbol}_*.csv'))
        if cache_pattern:
            df = pd.read_csv(cache_pattern[0], parse_dates=['timestamp'])
            df = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
        else:
            print(f"    No cached data found for {symbol}")
            return pd.DataFrame()
    if df.empty:
        print(f"    No data available for {symbol}")
        return pd.DataFrame()
    print(f"    Loaded {len(df)} bars")
    print(f"    Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    print(f"\n[2] Generating signals (delegated to strategy)...")
    df = df.sort_values("timestamp").copy()
    sigs = strat.scan_signals_over_time(df)
    if sigs.empty:
        print("    No signals generated.")
        return sigs
    # Show last few
    print("\n[3] Last 5 signals:")
    print(sigs.tail(5).to_string(index=False))
    return sigs


# =============================================================================
# Performance Profiling
# =============================================================================

def profile_script(script_path: str, args: List[str] = None) -> None:
    """Profile a Python script's performance."""
    print("\n" + "=" * 70)
    print(f"PERFORMANCE PROFILE - {script_path}")
    print("=" * 70 + "\n")

    script = Path(script_path)
    if not script.is_absolute():
        script = ROOT / 'scripts' / script_path

    if not script.exists():
        print(f"Script not found: {script}")
        return

    args = args or []

    # Use cProfile
    print(f"Profiling: {script}")
    print(f"Arguments: {args}")
    print("-" * 50 + "\n")

    try:
        # Read and compile script
        with open(script, 'r', encoding='utf-8') as f:
            code = f.read()

        # Set up profiler
        profiler = cProfile.Profile()

        # Run with profiler
        globals_dict = {
            '__name__': '__main__',
            '__file__': str(script),
        }
        sys.argv = [str(script)] + args

        old_cwd = os.getcwd()
        os.chdir(str(script.parent))

        try:
            profiler.enable()
            exec(compile(code, str(script), 'exec'), globals_dict)
            profiler.disable()
        finally:
            os.chdir(old_cwd)

        # Generate stats
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats('cumulative')
        stats.print_stats(30)

        print("\n" + "=" * 50)
        print("TOP 30 FUNCTIONS BY CUMULATIVE TIME")
        print("=" * 50)
        print(stream.getvalue())

        # Additional analysis
        stream2 = io.StringIO()
        stats2 = pstats.Stats(profiler, stream=stream2)
        stats2.sort_stats('tottime')
        stats2.print_stats(15)

        print("\n" + "=" * 50)
        print("TOP 15 FUNCTIONS BY SELF TIME")
        print("=" * 50)
        print(stream2.getvalue())

        # Summary
        print("\n" + "=" * 50)
        print("PROFILING RECOMMENDATIONS")
        print("=" * 50)
        print("- Focus on functions with highest cumulative time")
        print("- Look for functions called many times (check ncalls)")
        print("- Consider caching for expensive repeated operations")
        print("- Use vectorized operations instead of loops for data processing")

    except Exception as e:
        print(f"Profiling error: {e}")
        traceback.print_exc()


def quick_profile_function(func_name: str, module_path: str, *args, **kwargs) -> None:
    """Quick profile a specific function."""
    print(f"\nProfiling function: {func_name} from {module_path}")

    try:
        import importlib
        module = importlib.import_module(module_path)
        func = getattr(module, func_name)

        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()

        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats('cumulative')
        stats.print_stats(20)

        print(stream.getvalue())
        return result

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()


# =============================================================================
# Issue Reproduction
# =============================================================================

def reproduce_error(error_id: str, verbose: bool = True) -> None:
    """Attempt to reproduce an error from its ID."""
    print("\n" + "=" * 70)
    print(f"REPRODUCE ERROR - {error_id}")
    print("=" * 70 + "\n")

    # Search for error in logs
    all_entries: List[Dict[str, Any]] = []
    for log_file in [EVENTS_LOG, LOG_DIR / 'error.log']:
        if log_file.exists():
            all_entries.extend(parse_log_entries(log_file, limit=5000))

    errors = find_errors(all_entries, max_errors=1000)
    target_error: Optional[ErrorRecord] = None

    for err in errors:
        if err.error_id == error_id or error_id in err.error_id:
            target_error = err
            break

    if not target_error:
        print(f"Error ID not found: {error_id}")
        print("\nAvailable error IDs:")
        for err in errors[-10:]:
            print(f"  {err.error_id}")
        return

    print(f"Found error: {target_error.summary()}")
    print(f"\nEvent: {target_error.event}")
    print(f"Message: {target_error.message}")

    if target_error.context:
        print(f"\nContext data:")
        for k, v in target_error.context.items():
            print(f"  {k}: {v}")

    # Try to extract reproduction info
    print("\n" + "-" * 50)
    print("REPRODUCTION ATTEMPT")
    print("-" * 50)

    context = target_error.context
    event = target_error.event.lower()

    # Check if we can reproduce based on context
    if 'symbol' in context:
        symbol = context['symbol']
        print(f"\nSymbol involved: {symbol}")

        # Try signal trace
        start = context.get('start', (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
        end = context.get('end', datetime.now().strftime('%Y-%m-%d'))
        strategy = context.get('strategy', 'rsi2')

        print(f"Attempting signal trace for {symbol}...")
        try:
            trace_signal_generation(symbol, start, end, strategy, verbose=verbose)
        except Exception as e:
            print(f"Signal trace failed: {e}")

    if 'script' in context or 'file' in context:
        script = context.get('script', context.get('file', ''))
        print(f"\nScript involved: {script}")
        print("To reproduce, run:")
        print(f"  python {script}")

    if target_error.traceback:
        print("\nOriginal traceback:")
        print(target_error.traceback)

    print("\n" + "-" * 50)
    print("SUGGESTED ACTIONS")
    print("-" * 50)
    print("1. Review the error context above")
    print("2. Check if input data is valid")
    print("3. Verify API connectivity (python scripts/preflight.py)")
    print("4. Run with verbose logging to capture more details")
    print("5. Add try/except blocks around suspected code")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Kobe Trading System - Debug and Diagnosis Tool'
    )
    parser.add_argument(
        '--dotenv', type=str,
        default='./.env',
        help='Path to .env file'
    )
    parser.add_argument(
        '--last-error', action='store_true',
        help='Analyze the last error from logs'
    )
    parser.add_argument(
        '--trace-signal', type=str, metavar='SYMBOL',
        help='Trace signal generation for a symbol'
    )
    parser.add_argument(
        '--start', type=str, default=None,
        help='Start date for signal trace (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end', type=str, default=None,
        help='End date for signal trace (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--strategy', type=str, choices=['ibs_rsi', 'turtle_soup', 'ict'], default='ibs_rsi',
        help='Strategy for signal trace'
    )
    parser.add_argument(
        '--profile', type=str, metavar='SCRIPT',
        help='Profile a script (e.g., run_backtest.py)'
    )
    parser.add_argument(
        '--profile-args', type=str, default='',
        help='Arguments for profiled script'
    )
    parser.add_argument(
        '--reproduce', type=str, metavar='ERROR_ID',
        help='Reproduce an issue from error ID'
    )
    parser.add_argument(
        '--list-errors', action='store_true',
        help='List recent errors from logs'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Load environment
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        loaded = load_env(dotenv)
        if args.verbose:
            print(f"Loaded {len(loaded)} environment variables from {dotenv}")

    # Execute requested action
    if args.last_error:
        analyze_last_error(verbose=args.verbose)

    elif args.trace_signal:
        start = args.start or (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        end = args.end or datetime.now().strftime('%Y-%m-%d')
        trace_signal_generation(
            args.trace_signal,
            start,
            end,
            strategy=args.strategy,
            verbose=args.verbose
        )

    elif args.profile:
        profile_args = args.profile_args.split() if args.profile_args else []
        profile_script(args.profile, profile_args)

    elif args.reproduce:
        reproduce_error(args.reproduce, verbose=args.verbose)

    elif args.list_errors:
        print("\n" + "=" * 70)
        print("RECENT ERRORS")
        print("=" * 70 + "\n")

        all_entries: List[Dict[str, Any]] = []
        for log_file in [EVENTS_LOG, LOG_DIR / 'error.log']:
            if log_file.exists():
                all_entries.extend(parse_log_entries(log_file))

        errors = find_errors(all_entries, max_errors=20)
        if errors:
            for err in errors:
                print(err.summary())
        else:
            print("No errors found in recent logs.")

    else:
        parser.print_help()
        print("\nExamples:")
        print("  python debugger.py --last-error")
        print("  python debugger.py --trace-signal AAPL --start 2024-01-01 --end 2024-06-30")
        print("  python debugger.py --profile run_backtest.py")
        print("  python debugger.py --reproduce ERR-20240601-001")
        print("  python debugger.py --list-errors")


if __name__ == '__main__':
    main()

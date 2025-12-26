#!/usr/bin/env python3
"""
Kobe Daily Stock Scanner

Scans the universe for trading signals using RSI-2, IBS, CRSI (mean-reversion),
and Donchian breakout (trend) strategies.

Features:
- Loads universe from data/universe/optionable_liquid_final.csv
- Fetches latest EOD data via Polygon
- Runs both RSI-2 and IBS strategies (or filter by --strategy)
- Outputs signals to stdout and logs/signals.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.env_loader import load_env
from data.providers.multi_source import fetch_daily_bars_multi
from data.universe.loader import load_universe
from strategies.connors_rsi2.strategy import ConnorsRSI2Strategy
from strategies.ibs.strategy import IBSStrategy
from strategies.connors_crsi.strategy import ConnorsCRSIStrategy
from strategies.donchian.strategy import DonchianBreakoutStrategy, DonchianParams
from config.settings_loader import get_selection_config
from core.regime_filter import get_regime_filter_config, filter_signals_by_regime, fetch_spy_bars
from core.earnings_filter import filter_signals_by_earnings


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
DEFAULT_UNIVERSE = ROOT / "data" / "universe" / "optionable_liquid_final.csv"
SIGNALS_LOG = ROOT / "logs" / "signals.jsonl"
CACHE_DIR = ROOT / "data" / "cache"
LOOKBACK_DAYS = 300  # Need 200+ days for SMA(200) + buffer


# -----------------------------------------------------------------------------
# Scanner functions
# -----------------------------------------------------------------------------
def fetch_symbol_data(
    symbol: str,
    start_date: str,
    end_date: str,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Fetch EOD data for a single symbol."""
    try:
        df = fetch_daily_bars_multi(
            symbol=symbol,
            start=start_date,
            end=end_date,
            cache_dir=cache_dir,
        )
        return df
    except Exception as e:
        print(f"  [WARN] Failed to fetch {symbol}: {e}", file=sys.stderr)
        return pd.DataFrame()


def run_strategies(
    data: pd.DataFrame,
    strategies: List[str],
    apply_filters: bool,
    spy_bars: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Run specified strategies and return combined signals."""
    all_signals: List[pd.DataFrame] = []

    if "rsi2" in strategies or "all" in strategies:
        try:
            rsi2_strat = ConnorsRSI2Strategy()
            rsi2_signals = rsi2_strat.generate_signals(data)
            if not rsi2_signals.empty:
                rsi2_signals["strategy"] = "rsi2"
                all_signals.append(rsi2_signals)
        except Exception as e:
            print(f"  [WARN] RSI-2 strategy error: {e}", file=sys.stderr)

    if "ibs" in strategies or "all" in strategies:
        try:
            ibs_strat = IBSStrategy()
            ibs_signals = ibs_strat.generate_signals(data)
            if not ibs_signals.empty:
                ibs_signals["strategy"] = "ibs"
                all_signals.append(ibs_signals)
        except Exception as e:
            print(f"  [WARN] IBS strategy error: {e}", file=sys.stderr)

    if "crsi" in strategies or "all" in strategies:
        try:
            crsi_strat = ConnorsCRSIStrategy()
            crsi_signals = crsi_strat.generate_signals(data)
            if not crsi_signals.empty:
                crsi_signals["strategy"] = "crsi"
                all_signals.append(crsi_signals)
        except Exception as e:
            print(f"  [WARN] CRSI strategy error: {e}", file=sys.stderr)

    if "donchian" in strategies or "all" in strategies:
        try:
            sel_cfg = get_selection_config()
            don_params = DonchianParams(min_price=float(sel_cfg.get('min_price', 5.0)))
            don_strat = DonchianBreakoutStrategy(don_params)
            don_signals = don_strat.generate_signals(data)
            if not don_signals.empty:
                # breakout strength for ranking: (close - donchian_hi) / ATR approx using stop distance
                if 'entry_price' in don_signals.columns and 'stop_loss' in don_signals.columns:
                    try:
                        don_signals['r'] = (don_signals['entry_price'] - don_signals['stop_loss']).abs()
                        don_signals['breakout_strength'] = (don_signals['entry_price'] - don_signals['donchian_hi']) / don_signals['r'].replace(0, pd.NA)
                    except Exception:
                        pass
                don_signals["strategy"] = "donchian"
                all_signals.append(don_signals)
        except Exception as e:
            print(f"  [WARN] Donchian strategy error: {e}", file=sys.stderr)

    if all_signals:
        sigs = pd.concat(all_signals, ignore_index=True)
        # Apply regime/earnings filters if requested
        if apply_filters and spy_bars is not None and not spy_bars.empty:
            try:
                sigs = filter_signals_by_regime(sigs, spy_bars, get_regime_filter_config())
            except Exception:
                pass
        if apply_filters and not sigs.empty:
            try:
                recs = sigs.to_dict('records')
                sigs = pd.DataFrame(filter_signals_by_earnings(recs))
            except Exception:
                pass
        return sigs
    return pd.DataFrame()


def log_signals(signals: pd.DataFrame, scan_id: str) -> None:
    """Append signals to JSONL log file."""
    SIGNALS_LOG.parent.mkdir(parents=True, exist_ok=True)

    with SIGNALS_LOG.open("a", encoding="utf-8") as f:
        for _, row in signals.iterrows():
            record = {
                "ts": datetime.utcnow().isoformat(),
                "scan_id": scan_id,
                "event": "signal",
                **{k: v for k, v in row.items() if pd.notna(v)},
            }
            # Convert Timestamp to string
            for k, v in record.items():
                if isinstance(v, pd.Timestamp):
                    record[k] = v.isoformat()
            f.write(json.dumps(record, default=str) + "\n")


def format_signal_row(row: pd.Series) -> str:
    """Format a single signal for display."""
    parts = [
        f"{row.get('strategy', '?'):>5}",
        f"{row.get('symbol', '?'):<6}",
        f"{row.get('side', '?'):<6}",
        f"@ ${row.get('entry_price', 0):>8.2f}",
        f"stop ${row.get('stop_loss', 0):>8.2f}",
    ]
    reason = row.get("reason", "")
    if reason:
        parts.append(f"| {reason}")
    return " ".join(parts)


def print_signals_table(signals: pd.DataFrame) -> None:
    """Print signals in a formatted table."""
    if signals.empty:
        print("\n  No signals generated.")
        return

    print("\n  SIGNALS")
    print("  " + "-" * 76)
    print(f"  {'STRAT':>5} {'SYMBOL':<6} {'SIDE':<6} {'ENTRY':>12} {'STOP':>12} | REASON")
    print("  " + "-" * 76)

    for _, row in signals.iterrows():
        print("  " + format_signal_row(row))

    print("  " + "-" * 76)
    print(f"  Total: {len(signals)} signal(s)")

    # Summary by strategy
    if "strategy" in signals.columns:
        by_strat = signals.groupby("strategy").size()
        print(f"  By strategy: {dict(by_strat)}")

    # Summary by side
    if "side" in signals.columns:
        by_side = signals.groupby("side").size()
        print(f"  By side: {dict(by_side)}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Kobe Daily Stock Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/scan.py                        # Scan all strategies
  python scripts/scan.py --strategy rsi2        # Only RSI-2 signals
  python scripts/scan.py --strategy ibs         # Only IBS signals
  python scripts/scan.py --cap 50               # Scan first 50 symbols
  python scripts/scan.py --json                 # Output as JSON
        """,
    )
    ap.add_argument(
        "--dotenv",
        type=str,
        default="C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env",
        help="Path to .env file",
    )
    ap.add_argument(
        "--universe",
        type=str,
        default=str(DEFAULT_UNIVERSE),
        help="Path to universe CSV file",
    )
    ap.add_argument(
        "--strategy",
        type=str,
        choices=["rsi2", "ibs", "crsi", "donchian", "all"],
        default="all",
        help="Strategy to run (default: all)",
    )
    ap.add_argument(
        "--cap",
        type=int,
        default=None,
        help="Limit number of symbols to scan",
    )
    ap.add_argument("--top3", action="store_true", help="Select top 3 picks: 2 MR (CRSI/RSI2/IBS) + 1 Donchian")
    ap.add_argument("--min-price", type=float, default=None, help="Override min price for selection")
    ap.add_argument("--no-filters", action="store_true", help="Disable regime/earnings filters")
    ap.add_argument("--date", type=str, default=None, help="Use YYYY-MM-DD as end date (default: last business day)")
    ap.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD), default: lookback from today",
    )
    ap.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD), default: today",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Output signals as JSON",
    )
    ap.add_argument(
        "--no-log",
        action="store_true",
        help="Skip writing to signals.jsonl",
    )
    ap.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    args = ap.parse_args()

    # Load environment
    dotenv_path = Path(args.dotenv)
    if dotenv_path.exists():
        loaded = load_env(dotenv_path)
        if args.verbose:
            print(f"Loaded {len(loaded)} env vars from {dotenv_path}")
    else:
        if args.verbose:
            print(f"Warning: dotenv file not found: {dotenv_path}", file=sys.stderr)

    # Check Polygon API key
    if not os.getenv("POLYGON_API_KEY"):
        print("Error: POLYGON_API_KEY not set. Please provide via --dotenv.", file=sys.stderr)
        return 1

    # Load universe
    universe_path = Path(args.universe)
    if not universe_path.exists():
        print(f"Error: Universe file not found: {universe_path}", file=sys.stderr)
        return 1

    symbols = load_universe(universe_path, cap=args.cap)
    if not symbols:
        print(f"Error: No symbols loaded from {universe_path}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Loaded {len(symbols)} symbols from {universe_path}")

    # Determine date range
    # Use last business day by default to avoid partial bars
    end_date = args.date or args.end or (datetime.utcnow().date().isoformat())
    if args.start:
        start_date = args.start
    else:
        end_dt = datetime.fromisoformat(end_date)
        start_dt = end_dt - timedelta(days=LOOKBACK_DAYS)
        start_date = start_dt.date().isoformat()

    if args.verbose:
        print(f"Date range: {start_date} to {end_date}")

    # Determine strategies to run
    strategies = [args.strategy] if args.strategy != "all" else ["rsi2", "ibs", "crsi", "donchian"]
    if args.verbose:
        print(f"Strategies: {strategies}")

    # Scan ID for logging
    scan_id = f"SCAN_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    # Fetch data and run strategies
    print(f"\nKobe Scanner - {scan_id}")
    print(f"Scanning {len(symbols)} symbols for {', '.join(strategies)} signals...")
    print("-" * 60)

    all_data: List[pd.DataFrame] = []
    success_count = 0
    fail_count = 0

    # Ensure cache directory exists
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for i, symbol in enumerate(symbols, 1):
        if args.verbose:
            print(f"  [{i}/{len(symbols)}] Fetching {symbol}...", end=" ")

        df = fetch_symbol_data(symbol, start_date, end_date, CACHE_DIR)
        if not df.empty and len(df) > 0:
            all_data.append(df)
            success_count += 1
            if args.verbose:
                print(f"{len(df)} bars")
        else:
            fail_count += 1
            if args.verbose:
                print("SKIP (no data)")

    print(f"\nFetched: {success_count} symbols, skipped: {fail_count}")

    if not all_data:
        print("Error: No data fetched for any symbols.", file=sys.stderr)
        return 1

    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)
    print(f"Total bars: {len(combined):,}")

    # Run strategies
    print("\nRunning strategies...")
    # Load SPY for regime filter if enabled
    spy_bars = None
    apply_filters = not args.no_filters
    if apply_filters:
        try:
            spy_bars = fetch_spy_bars(start_date, end_date, cache_dir=CACHE_DIR)
        except Exception:
            spy_bars = None

    # Selection config overrides
    sel_cfg = get_selection_config()
    if args.min_price is not None and args.min_price > 0:
        sel_cfg['min_price'] = float(args.min_price)

    signals = run_strategies(combined, strategies, apply_filters=apply_filters, spy_bars=spy_bars)

    # Output results
    if args.json:
        if not signals.empty:
            # Convert to JSON-serializable format
            output = []
            for _, row in signals.iterrows():
                rec = {k: v for k, v in row.items() if pd.notna(v)}
                for k, v in rec.items():
                    if isinstance(v, pd.Timestamp):
                        rec[k] = v.isoformat()
                output.append(rec)
            print(json.dumps(output, indent=2, default=str))
        else:
            print("[]")
    else:
        if not args.top3:
            print_signals_table(signals)
        else:
            # Rank MR and Donchian and pick 2 + 1
            picks = []
            if not signals.empty:
                df = signals.copy()
                # Enforce min_price
                if 'entry_price' in df.columns:
                    df = df[df['entry_price'] >= float(sel_cfg.get('min_price', 5.0))]
                # Split MR vs Donchian
                mr = df[df['strategy'].isin(['crsi','rsi2','ibs'])].copy()
                dn = df[df['strategy'] == 'donchian'].copy()
                # MR ranking: lower CRSI/RSI2/IBS better
                if not mr.empty:
                    mr['crsi'] = mr.get('crsi', pd.Series([pd.NA]*len(mr)))
                    mr['rsi2'] = mr.get('rsi2', pd.Series([pd.NA]*len(mr)))
                    mr['ibs'] = mr.get('ibs', pd.Series([pd.NA]*len(mr)))
                    w = sel_cfg.get('score_weights', {'crsi':0.5,'rsi2':0.3,'ibs':0.2})
                    def score_row(row):
                        vals = []
                        s = 0.0
                        total = 0.0
                        if pd.notna(row.get('crsi')):
                            s += float(w.get('crsi',0.0)) * (100.0 - float(row['crsi']))/100.0
                            total += float(w.get('crsi',0.0))
                        if pd.notna(row.get('rsi2')):
                            s += float(w.get('rsi2',0.0)) * (100.0 - float(row['rsi2']))/100.0
                            total += float(w.get('rsi2',0.0))
                        if pd.notna(row.get('ibs')):
                            # IBS in [0,1], lower is better
                            s += float(w.get('ibs',0.0)) * (1.0 - float(row['ibs']))
                            total += float(w.get('ibs',0.0))
                        return s/total if total>0 else 0.0
                    mr['score'] = mr.apply(score_row, axis=1)
                    mr = mr.sort_values('score', ascending=False)
                    picks.append(mr.head(2))
                # Donchian: rank by breakout_strength (higher better)
                if not dn.empty:
                    if 'breakout_strength' not in dn.columns:
                        try:
                            dn['r'] = (dn['entry_price'] - dn['stop_loss']).abs()
                            dn['breakout_strength'] = (dn['entry_price'] - dn['donchian_hi']) / dn['r'].replace(0, pd.NA)
                        except Exception:
                            dn['breakout_strength'] = 0.0
                    dn = dn.sort_values('breakout_strength', ascending=False)
                    picks.append(dn.head(1))
            out = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame()
            if out.empty:
                print_signals_table(signals)
            else:
                # Compute confidence score and write picks + trade of the day
                out = out.copy()
                # MR confidence uses composite 'score' if present; fallback to inverted indicators
                if 'score' not in out.columns:
                    out['score'] = 0.0
                # Donchian confidence from breakout_strength normalized
                if 'breakout_strength' not in out.columns:
                    out['breakout_strength'] = 0.0
                def conf(row):
                    strat = str(row.get('strategy','')).lower()
                    if strat in ('crsi','rsi2','ibs'):
                        try:
                            return float(row.get('score', 0.0))
                        except Exception:
                            return 0.0
                    if strat == 'donchian':
                        try:
                            bs = float(row.get('breakout_strength', 0.0))
                            # Normalize: cap at 3.0
                            return max(0.0, min(bs, 3.0)) / 3.0
                        except Exception:
                            return 0.0
                    return 0.0
                out['conf_score'] = out.apply(conf, axis=1)

                # Write Top 3 picks
                picks_path = ROOT / 'logs' / 'daily_picks.csv'
                picks_path.parent.mkdir(parents=True, exist_ok=True)
                out.to_csv(picks_path, index=False)

                # Choose Trade of the Day (highest confidence)
                totd = out.sort_values('conf_score', ascending=False).head(1)
                totd_path = ROOT / 'logs' / 'trade_of_day.csv'
                totd.to_csv(totd_path, index=False)

                print("\nTOP 3 PICKS")
                print("-" * 60)
                print(out[['strategy','symbol','side','entry_price','stop_loss','take_profit','conf_score']].to_string(index=False))
                print(f"\nWrote: {picks_path}")
                print("\nTRADE OF THE DAY")
                print("-" * 60)
                print(totd[['strategy','symbol','side','entry_price','stop_loss','take_profit','conf_score']].to_string(index=False))
                print(f"\nWrote: {totd_path}")

    # Log signals
    if not args.no_log and not signals.empty:
        log_signals(signals, scan_id)
        print(f"\nSignals logged to: {SIGNALS_LOG}")

    # Summary
    print("\n" + "=" * 60)
    print(f"Scan complete: {len(signals)} signal(s) generated")

    return 0


if __name__ == "__main__":
    sys.exit(main())

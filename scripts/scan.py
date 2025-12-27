#!/usr/bin/env python3
"""
Kobe Daily Stock Scanner

Scans the universe for trading signals using ICT Turtle Soup (failed breakout mean-reversion)
and Donchian breakout (trend-following).

Features:
- Loads universe from data/universe/optionable_liquid_900.csv (900 symbols)
- Fetches latest EOD data via Polygon
- Runs Donchian and ICT strategies (or filter by --strategy)
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
from strategies.donchian.strategy import DonchianBreakoutStrategy, DonchianParams
from strategies.ict.turtle_soup import TurtleSoupStrategy
from config.settings_loader import get_selection_config
from core.regime_filter import get_regime_filter_config, filter_signals_by_regime, fetch_spy_bars
from core.earnings_filter import filter_signals_by_earnings
from ml_meta.features import compute_features_frame
from ml_meta.model import load_model, predict_proba, FEATURE_COLS
from altdata.sentiment import load_daily_cache, normalize_sentiment_to_conf

# Cognitive system (optional)
try:
    from cognitive.signal_processor import get_signal_processor
    COGNITIVE_AVAILABLE = True
except ImportError:
    COGNITIVE_AVAILABLE = False


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
DEFAULT_UNIVERSE = ROOT / "data" / "universe" / "optionable_liquid_900.csv"
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

    if "turtle_soup" in strategies or "ict" in strategies or "all" in strategies:
        try:
            ict_strat = TurtleSoupStrategy()
            ict_signals = ict_strat.generate_signals(data)
            if not ict_signals.empty:
                ict_signals["strategy"] = "turtle_soup"
                all_signals.append(ict_signals)
        except Exception as e:
            print(f"  [WARN] ICT Turtle Soup strategy error: {e}", file=sys.stderr)

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
  python scripts/scan.py --strategy donchian    # Only Donchian signals
  python scripts/scan.py --strategy turtle_soup # Only ICT signals
  python scripts/scan.py --cap 50               # Scan first 50 symbols
  python scripts/scan.py --json                 # Output as JSON
        """,
    )
    ap.add_argument(
        "--dotenv",
        type=str,
        default="./.env",
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
        choices=["donchian", "turtle_soup", "ict", "all"],
        default="all",
        help="Strategy to run (default: all)",
    )
    ap.add_argument(
        "--cap",
        type=int,
        default=None,
        help="Limit number of symbols to scan",
    )
    ap.add_argument("--top3", action="store_true", help="Select top 3 picks: 2 ICT (MR) + 1 Donchian")
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
    ap.add_argument("--ml", action="store_true", help="Score signals with ML meta-models if available")
    ap.add_argument("--min-conf", type=float, default=0.55, help="Min confidence [0-1] to approve TOTD when --ml is on")
    ap.add_argument("--min-adv-usd", type=float, default=5000000.0, help="Minimum 60-day ADV in USD to consider for Top-3/TOTD")
    ap.add_argument("--ensure-top3", action="store_true", help="Guarantee 3 picks; fill from highest-confidence leftovers")
    ap.add_argument("--out-picks", type=str, default=str(ROOT / 'logs' / 'daily_picks.csv'), help="Output CSV for Top-3 picks")
    ap.add_argument("--out-totd", type=str, default=str(ROOT / 'logs' / 'trade_of_day.csv'), help="Output CSV for Trade of the Day")
    # (ml arg already defined above)
    ap.add_argument(
        "--no-log",
        action="store_true",
        help="Skip writing to signals.jsonl",
    )
    ap.add_argument(
        "--cognitive",
        action="store_true",
        help="Enable cognitive brain evaluation for smarter signal filtering",
    )
    ap.add_argument(
        "--cognitive-min-conf",
        type=float,
        default=0.5,
        help="Minimum cognitive confidence to approve signal (default: 0.5)",
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
    strategies = [args.strategy] if args.strategy != "all" else ["donchian", "turtle_soup"]
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

    # Optional ML scoring
    if args.ml and not signals.empty:
        try:
            feats = compute_features_frame(combined)
            feats['timestamp'] = pd.to_datetime(feats['timestamp']).dt.normalize()
            sigs = signals.copy()
            sigs['timestamp'] = pd.to_datetime(sigs['timestamp']).dt.normalize()
            sigs = pd.merge(sigs, feats, on=['symbol','timestamp'], how='left')
            for col in FEATURE_COLS:
                if col not in sigs.columns:
                    sigs[col] = 0.0
            m_don = load_model('donchian')
            m_ict = load_model('turtle_soup')
            conf_vals = []
            for _, r in sigs.iterrows():
                strat = str(r.get('strategy','')).lower()
                row = r.reindex(FEATURE_COLS).astype(float).to_frame().T
                if strat == 'donchian' and m_don is not None:
                    conf_vals.append(float(predict_proba(m_don, row)[0]))
                elif strat in ('turtle_soup','ict') and m_ict is not None:
                    conf_vals.append(float(predict_proba(m_ict, row)[0]))
                else:
                    conf_vals.append(float(r.get('conf_score', 0.0)) if 'conf_score' in r else 0.0)
            sigs['conf_score'] = conf_vals

            # Blend sentiment if available for end_date
            try:
                end_day = pd.to_datetime(end_date).date().isoformat()
                sent = load_daily_cache(end_day)
                if not sent.empty and 'date' in sent.columns:
                    sent['date'] = pd.to_datetime(sent['date']).dt.normalize()
                    sigs = pd.merge(
                        sigs,
                        sent.rename(columns={'date': 'timestamp'})[['timestamp','symbol','sent_mean']],
                        on=['timestamp','symbol'], how='left'
                    )
                    sigs['sent_mean'] = sigs['sent_mean'].astype(float).fillna(0.0)
                    sent_conf = sigs['sent_mean'].apply(normalize_sentiment_to_conf)
                    # Blend: 0.8 ML probability + 0.2 sentiment
                    sigs['conf_score'] = 0.8 * sigs['conf_score'].astype(float) + 0.2 * sent_conf.astype(float)
            except Exception:
                pass

            signals = sigs
        except Exception as e:
            if args.verbose:
                print(f"  [WARN] ML scoring failed: {e}")

    # Optional cognitive brain evaluation
    cognitive_evaluated = []
    if args.cognitive and COGNITIVE_AVAILABLE and not signals.empty:
        print("\nRunning cognitive brain evaluation...")
        try:
            processor = get_signal_processor()
            processor.min_confidence = args.cognitive_min_conf

            # Build fast confidences from ML scores
            fast_confs = {}
            if 'conf_score' in signals.columns:
                for _, row in signals.iterrows():
                    sym = row.get('symbol', '')
                    if sym and pd.notna(row.get('conf_score')):
                        fast_confs[sym] = float(row['conf_score'])

            # Evaluate through cognitive system
            approved_df, cognitive_evaluated = processor.evaluate_signals(
                signals=signals,
                market_data=combined,
                spy_data=spy_bars,
                fast_confidences=fast_confs,
            )

            if not approved_df.empty:
                # Merge cognitive results back
                signals = approved_df
                print(f"  Cognitive: {len(cognitive_evaluated)} evaluated -> {len(approved_df)} approved")

                # Show cognitive reasoning for each approved signal
                if args.verbose:
                    for ev in cognitive_evaluated:
                        if ev.approved:
                            sym = ev.original_signal.get('symbol', '?')
                            print(f"    {sym}: conf={ev.cognitive_confidence:.2f}, mode={ev.decision_mode}")
                            if ev.concerns:
                                print(f"      Concerns: {ev.concerns[:2]}")
            else:
                print("  Cognitive: All signals rejected (below confidence threshold)")
                signals = pd.DataFrame()

        except Exception as e:
            print(f"  [WARN] Cognitive evaluation failed: {e}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
    elif args.cognitive and not COGNITIVE_AVAILABLE:
        print("  [WARN] Cognitive system not available (import failed)")

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
            # Rank ICT (MR) and Donchian and pick 2 + 1
            picks = []
            if not signals.empty:
                df = signals.copy()
                # Enforce min_price
                if 'entry_price' in df.columns:
                    df = df[df['entry_price'] >= float(sel_cfg.get('min_price', 5.0))]
                # ADV $ filter using combined data last 60 days
                try:
                    bars = combined.copy()
                    bars['usd_vol'] = (bars['close'] * bars['volume']).astype(float)
                    adv = bars.groupby('symbol')['usd_vol'].rolling(60, min_periods=10).mean().reset_index(level=0, drop=True)
                    bars['adv_usd60'] = adv
                    bars_last = bars.sort_values('timestamp').groupby('symbol').tail(1)[['symbol','adv_usd60']]
                    df = pd.merge(df, bars_last, on='symbol', how='left')
                    df = df[df['adv_usd60'] >= float(args.min_adv_usd)]
                except Exception:
                    pass
                # Split ICT MR vs Donchian
                mr = df[df['strategy'].isin(['turtle_soup'])].copy()
                dn = df[df['strategy'] == 'donchian'].copy()
                # Simple MR ranking: prefer tighter stops (higher r_multiple potential); fallback to most recent
                if not mr.empty:
                    if 'time_stop_bars' in mr.columns:
                        mr = mr.sort_values(['time_stop_bars'], ascending=True)
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
                    # Prefer ML conf_score if present
                    if 'conf_score' in row and pd.notna(row['conf_score']):
                        try:
                            return float(row['conf_score'])
                        except Exception:
                            pass
                    strat = str(row.get('strategy','')).lower()
                    if strat == 'donchian':
                        try:
                            bs = float(row.get('breakout_strength', 0.0))
                            # Normalize: cap at 3.0
                            return max(0.0, min(bs, 3.0)) / 3.0
                        except Exception:
                            return 0.0
                    return float(row.get('score', 0.0)) if strat in ('turtle_soup','ict') else 0.0
                out['conf_score'] = out.apply(conf, axis=1)

                # Ensure Top-3 by filling from highest-confidence leftovers if requested
                if args.ensure_top3 and len(out) < 3:
                    left = df.copy()
                    # Remove already picked rows
                    if not out.empty:
                        keycols = ['timestamp','symbol','side']
                        left = left.merge(out[keycols], on=keycols, how='left', indicator=True)
                        left = left[left['_merge'] == 'left_only'].drop(columns=['_merge'])
                    # Attach confidence to leftovers (prefer ML, else heuristic)
                    if 'conf_score' not in left.columns:
                        def base_conf(r):
                            if str(r.get('strategy','')).lower() == 'donchian':
                                try:
                                    bs = float(r.get('breakout_strength', 0.0))
                                    return max(0.0, min(bs, 3.0)) / 3.0
                                except Exception:
                                    return 0.0
                            return float(r.get('score', 0.0)) if str(r.get('strategy','')).lower() in ('turtle_soup','ict') else 0.0
                        left['conf_score'] = left.apply(base_conf, axis=1)
                    left = left.sort_values('conf_score', ascending=False)
                    need = 3 - len(out)
                    if need > 0 and not left.empty:
                        out = pd.concat([out, left.head(need)], ignore_index=True)

                # Write Top 3 picks
                picks_path = Path(args.out_picks)
                picks_path.parent.mkdir(parents=True, exist_ok=True)
                out.to_csv(picks_path, index=False)

                # Choose Trade of the Day (highest confidence)
                totd = out.sort_values('conf_score', ascending=False).head(1)
                approve_totd = True
                if args.ml and not totd.empty:
                    try:
                        approve_totd = float(totd.iloc[0]['conf_score']) >= float(args.min_conf)
                    except Exception:
                        approve_totd = True
                totd_path = Path(args.out_totd)
                if approve_totd and not totd.empty:
                    totd.to_csv(totd_path, index=False)
                else:
                    # Write empty placeholder to indicate skip
                    pd.DataFrame(columns=out.columns).to_csv(totd_path, index=False)

                print("\nTOP 3 PICKS")
                print("-" * 60)
                print(out[['strategy','symbol','side','entry_price','stop_loss','take_profit','conf_score']].to_string(index=False))
                print(f"\nWrote: {picks_path}")
                print("\nTRADE OF THE DAY")
                print("-" * 60)
                if approve_totd and not totd.empty:
                    print(totd[['strategy','symbol','side','entry_price','stop_loss','take_profit','conf_score']].to_string(index=False))
                    print(f"\nWrote: {totd_path}")
                else:
                    print("No TOTD due to low confidence or no picks.")

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

#!/usr/bin/env python3
"""
TOTD Backtest - Single Trade-of-the-Day simulation for 2025.

Simulates daily TOTD selection: ICT Turtle Soup vs IBS_RSI strategies.
Uses R-based exits: 1R stop, 2R target, 5-bar time stop.

Usage:
  python scripts/backtest_totd.py \
    --universe data/universe/optionable_liquid_900.csv \
    --start 2025-01-01 --end 2025-12-31 \
    --cap 900 --min-conf 0.60 --min-adv-usd 5000000

Output: reports/totd_YYYY/{trades.csv, summary.json, monthly.csv, report.html}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.env_loader import load_env
from data.providers.multi_source import fetch_daily_bars_multi
from data.universe.loader import load_universe
from strategies.registry import get_production_scanner
from core.regime_filter import get_regime_filter_config, filter_signals_by_regime, fetch_spy_bars
from core.earnings_filter import filter_signals_by_earnings

# Optional ML + sentiment
try:
    from ml_meta.features import compute_features_frame
    from ml_meta.model import load_model, predict_proba, FEATURE_COLS
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    from altdata.sentiment import load_daily_cache, normalize_sentiment_to_conf
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
DEFAULT_UNIVERSE = ROOT / "data" / "universe" / "optionable_liquid_900.csv"
CACHE_DIR = ROOT / "data" / "cache"
LOOKBACK_DAYS = 300  # Need 200+ for SMA(200)


# -----------------------------------------------------------------------------
# Trading Calendar
# -----------------------------------------------------------------------------
def get_trading_days(start: str, end: str) -> List[str]:
    """
    Get NYSE trading days between start and end dates.
    Uses pandas_market_calendars for accurate holiday handling.
    """
    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar('NYSE')
        schedule = nyse.schedule(start_date=start, end_date=end)
        return schedule.index.strftime('%Y-%m-%d').tolist()
    except ImportError:
        # Fallback: use business days (less accurate)
        dates = pd.bdate_range(start=start, end=end)
        return dates.strftime('%Y-%m-%d').tolist()


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
def fetch_bars_for_date(
    symbols: List[str],
    end_date: str,
    lookback_days: int = LOOKBACK_DAYS,
    cache_dir: Optional[Path] = None,
    verbose: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch EOD bars for all symbols ending at end_date with lookback.
    Returns dict of symbol -> DataFrame.
    """
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    start_dt = end_dt - timedelta(days=lookback_days + 50)  # Extra buffer for holidays
    start_date = start_dt.strftime('%Y-%m-%d')

    bars_dict = {}
    for i, symbol in enumerate(symbols):
        try:
            df = fetch_daily_bars_multi(
                symbol=symbol,
                start=start_date,
                end=end_date,
                cache_dir=cache_dir,
            )
            if not df.empty and len(df) >= 60:  # Need minimum bars
                bars_dict[symbol] = df
            if verbose and (i + 1) % 100 == 0:
                print(f"  Fetched {i + 1}/{len(symbols)} symbols...")
        except Exception as e:
            if verbose:
                print(f"  [WARN] {symbol}: {e}", file=sys.stderr)

    return bars_dict


def fetch_spy_data(end_date: str, cache_dir: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """Fetch SPY data for regime filter."""
    try:
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        start_dt = end_dt - timedelta(days=LOOKBACK_DAYS + 50)
        return fetch_spy_bars(start_dt.strftime('%Y-%m-%d'), end_date, cache_dir=cache_dir)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Signal Generation
# -----------------------------------------------------------------------------
def run_daily_scan(
    bars_dict: Dict[str, pd.DataFrame],
    scan_date: str,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run both strategies (ICT + IBS_RSI) on all symbols for a specific date.
    Returns DataFrame of signals with columns:
      symbol, strategy, side, entry_price, stop_loss, take_profit, reason, ...

    Uses scan_signals_over_time() to find ALL historical signals, then filters
    to only those on scan_date (avoids missing signals that generate_signals()
    would miss by only looking at the last bar).
    """
    all_signals: List[pd.DataFrame] = []

    # Initialize production scanner (CORRECT WAY)
    scanner = get_production_scanner()

    for symbol, df in bars_dict.items():
        # Filter bars up to scan_date (no lookahead)
        df_filtered = df.copy()
        df_filtered['_date'] = pd.to_datetime(df_filtered['timestamp']).dt.strftime('%Y-%m-%d')
        df_filtered = df_filtered[df_filtered['_date'] <= scan_date].drop(columns=['_date'])
        
        if df_filtered.empty or len(df_filtered) < 220:
            continue

        try:
            # The scanner will run both strategies
            sigs = scanner.scan_signals_over_time(df_filtered)
            if not sigs.empty:
                all_signals.append(sigs)
        except Exception as e:
            if verbose:
                print(f"  [WARN] Scan failed for {symbol}: {e}", file=sys.stderr)

    if all_signals:
        combined = pd.concat(all_signals, ignore_index=True)
        # Filter to only signals on scan_date
        combined['ts_date'] = pd.to_datetime(combined['timestamp']).dt.strftime('%Y-%m-%d')
        combined = combined[combined['ts_date'] == scan_date].drop(columns=['ts_date'])
        return combined

    return pd.DataFrame()


# -----------------------------------------------------------------------------
# Filters
# -----------------------------------------------------------------------------
def apply_filters(
    signals: pd.DataFrame,
    bars_dict: Dict[str, pd.DataFrame],
    spy_bars: Optional[pd.DataFrame],
    scan_date: str,
    min_adv_usd: float = 5_000_000,
    use_filters: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Apply filters:
    - Regime filter (SPY trend)
    - Earnings filter (Â±2 days blackout)
    - Liquidity filter (ADV USD >= threshold)
    """
    if signals.empty:
        return signals

    df = signals.copy()

    # Regime filter
    if use_filters and spy_bars is not None and not spy_bars.empty:
        try:
            cfg = get_regime_filter_config()
            df = filter_signals_by_regime(df, spy_bars, cfg)
        except Exception as e:
            if verbose:
                print(f"  [WARN] Regime filter failed: {e}")

    # Earnings filter
    if use_filters and not df.empty:
        try:
            recs = df.to_dict('records')
            df = pd.DataFrame(filter_signals_by_earnings(recs))
        except Exception as e:
            if verbose:
                print(f"  [WARN] Earnings filter failed: {e}")

    # Liquidity filter (ADV USD over last 60 days)
    if not df.empty and min_adv_usd > 0:
        liquid_symbols = []
        for symbol in df['symbol'].unique():
            if symbol in bars_dict:
                sym_bars = bars_dict[symbol]
                sym_bars = sym_bars[sym_bars['timestamp'] <= scan_date].tail(60)
                if len(sym_bars) >= 20:
                    adv_usd = (sym_bars['close'] * sym_bars['volume']).mean()
                    if adv_usd >= min_adv_usd:
                        liquid_symbols.append(symbol)
        df = df[df['symbol'].isin(liquid_symbols)]

    return df


# -----------------------------------------------------------------------------
# Confidence Scoring
# -----------------------------------------------------------------------------
def compute_confidence(
    signals: pd.DataFrame,
    bars_dict: Dict[str, pd.DataFrame],
    scan_date: str,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Compute confidence score: 0.8 * ML + 0.2 * sentiment.
    Falls back to heuristic (sweep_strength/breakout_strength) if ML unavailable.
    """
    if signals.empty:
        return signals

    df = signals.copy()

    # Initialize confidence with heuristic
    def heuristic_conf(row):
        strat = str(row.get('strategy', '')).lower()
        if strat == 'turtle_soup':
            # sweep_strength typically 0-3 range, normalize
            ss = float(row.get('sweep_strength', 0.5))
            return min(max(ss / 2.0, 0.1), 1.0)  # Normalize to 0.1-1.0
        elif strat == 'ibs_rsi':
            # breakout_strength if available
            if 'breakout_strength' in row:
                bs = float(row.get('breakout_strength', 0.5))
                return min(max(bs, 0.1), 1.0)
            # Fallback: use R ratio
            entry = float(row.get('entry_price', 0))
            stop = float(row.get('stop_loss', 0))
            if entry > 0 and stop > 0:
                r = abs(entry - stop) / entry
                return min(max(0.5 + r * 10, 0.1), 1.0)
        return 0.5

    df['conf_score'] = df.apply(heuristic_conf, axis=1)

    # Try ML scoring
    if ML_AVAILABLE:
        try:
            # Combine bars for feature computation
            all_bars = pd.concat(list(bars_dict.values()), ignore_index=True)
            all_bars = all_bars[all_bars['timestamp'] <= scan_date]

            feats = compute_features_frame(all_bars)
            feats['timestamp'] = pd.to_datetime(feats['timestamp']).dt.normalize()

            df_ml = df.copy()
            df_ml['timestamp'] = pd.to_datetime(df_ml['timestamp']).dt.normalize()
            df_ml = pd.merge(df_ml, feats, on=['symbol', 'timestamp'], how='left', suffixes=('', '_feat'))

            for col in FEATURE_COLS:
                if col not in df_ml.columns:
                    df_ml[col] = 0.0

            m_ibs = load_model('ibs_rsi')
            m_ict = load_model('turtle_soup')

            ml_scores = []
            for _, r in df_ml.iterrows():
                strat = str(r.get('strategy', '')).lower()
                feat_row = r.reindex(FEATURE_COLS).astype(float).to_frame().T

                if strat == 'ibs_rsi' and m_ibs is not None:
                    ml_scores.append(float(predict_proba(m_ibs, feat_row)[0]))
                elif strat == 'turtle_soup' and m_ict is not None:
                    ml_scores.append(float(predict_proba(m_ict, feat_row)[0]))
                else:
                    ml_scores.append(float(r.get('conf_score', 0.5)))

            df['ml_score'] = ml_scores
        except Exception as e:
            if verbose:
                print(f"  [WARN] ML scoring failed: {e}")
            df['ml_score'] = df['conf_score']
    else:
        df['ml_score'] = df['conf_score']

    # Try sentiment scoring
    sent_scores = [0.5] * len(df)
    if SENTIMENT_AVAILABLE:
        try:
            sent = load_daily_cache(scan_date)
            if not sent.empty and 'symbol' in sent.columns and 'sent_mean' in sent.columns:
                sent_map = dict(zip(sent['symbol'], sent['sent_mean']))
                # FIX: Use median of available sentiment for missing (not 0.0)
                fill_value = float(pd.Series(list(sent_map.values())).median()) if sent_map else 0.5
                sent_scores = [
                    normalize_sentiment_to_conf(sent_map.get(row['symbol'], fill_value))
                    for _, row in df.iterrows()
                ]
        except Exception:
            pass

    df['sent_score'] = sent_scores

    # Final confidence: 0.8 * ML + 0.2 * sentiment
    df['conf_score'] = 0.8 * df['ml_score'].astype(float) + 0.2 * df['sent_score'].astype(float)

    return df


# -----------------------------------------------------------------------------
# TOTD Selection
# -----------------------------------------------------------------------------
def select_top3(signals: pd.DataFrame, ensure_top3: bool = True) -> pd.DataFrame:
    """
    Select Top-3 picks: 2 ICT (mean-reversion) + 1 IBS_RSI (trend-following).
    If ensure_top3=True, fill from highest conf remaining if needed.
    """
    if signals.empty:
        return pd.DataFrame()

    df = signals.copy()
    picks = []

    # Split by strategy
    ict = df[df['strategy'] == 'turtle_soup'].sort_values('conf_score', ascending=False)
    don = df[df['strategy'] == 'ibs_rsi'].sort_values('conf_score', ascending=False)

    # Take top 2 ICT
    if len(ict) >= 2:
        picks.append(ict.head(2))
    elif len(ict) == 1:
        picks.append(ict.head(1))

    # Take top 1 IBS_RSI
    if len(don) >= 1:
        picks.append(don.head(1))

    if not picks:
        return pd.DataFrame()

    out = pd.concat(picks, ignore_index=True)

    # Ensure Top-3 if needed
    if ensure_top3 and len(out) < 3:
        used_keys = set(out['symbol'].tolist())
        remaining = df[~df['symbol'].isin(used_keys)].sort_values('conf_score', ascending=False)
        need = 3 - len(out)
        if len(remaining) >= need:
            out = pd.concat([out, remaining.head(need)], ignore_index=True)

    return out


def select_totd(top3: pd.DataFrame, min_conf: float = 0.60) -> Optional[pd.Series]:
    """
    Select Trade of the Day: highest confidence from Top-3 if >= min_conf.
    Returns None if no qualifying pick.
    """
    if top3.empty:
        return None

    best = top3.sort_values('conf_score', ascending=False).iloc[0]

    if float(best['conf_score']) >= min_conf:
        return best

    return None


# -----------------------------------------------------------------------------
# Trade Simulation
# -----------------------------------------------------------------------------
def simulate_trade(
    signal: pd.Series,
    bars_dict: Dict[str, pd.DataFrame],
    entry_date: str,
    time_stop_bars: int = 5,
) -> Optional[Dict[str, Any]]:
    """
    Simulate R-based trade:
    - Entry: next bar open after entry_date
    - 1R = abs(entry_price - stop_loss) from signal
    - Stop = 1R below entry (longs)
    - Target = +2R above entry
    - Time exit = time_stop_bars bars if neither hit

    Returns trade result dict or None if simulation failed.
    """
    symbol = signal['symbol']
    strategy = signal['strategy']
    side = signal.get('side', 'long')
    signal_entry = float(signal['entry_price'])
    signal_stop = float(signal['stop_loss'])

    if symbol not in bars_dict:
        return None

    df = bars_dict[symbol].copy()
    df = df.sort_values('timestamp')
    df['date'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d')

    # Find entry bar (next bar after signal date)
    signal_idx = df[df['date'] == entry_date].index
    if len(signal_idx) == 0:
        return None

    signal_iloc = df.index.get_loc(signal_idx[-1])
    if signal_iloc + 1 >= len(df):
        return None  # No next bar for entry

    entry_bar = df.iloc[signal_iloc + 1]
    actual_entry = float(entry_bar['open'])
    entry_bar['date']

    # Calculate 1R from signal
    signal_r = abs(signal_entry - signal_stop)
    if signal_r <= 0:
        return None

    # Set stop and target based on 1R from actual entry
    if side == 'long':
        stop_price = actual_entry - signal_r
        target_price = actual_entry + 2.0 * signal_r
    else:
        stop_price = actual_entry + signal_r
        target_price = actual_entry - 2.0 * signal_r

    # Simulate exit over next bars
    exit_price = None
    exit_reason = None
    bars_held = 0

    for i in range(1, time_stop_bars + 1):
        bar_iloc = signal_iloc + 1 + i
        if bar_iloc >= len(df):
            break

        bar = df.iloc[bar_iloc]
        bars_held += 1

        if side == 'long':
            # Check stop hit (low <= stop)
            if float(bar['low']) <= stop_price:
                exit_price = stop_price
                bar['date']
                exit_reason = 'stop'
                break
            # Check target hit (high >= target)
            if float(bar['high']) >= target_price:
                exit_price = target_price
                bar['date']
                exit_reason = 'target'
                break
        else:
            # Short: stop hit if high >= stop
            if float(bar['high']) >= stop_price:
                exit_price = stop_price
                bar['date']
                exit_reason = 'stop'
                break
            # Target hit if low <= target
            if float(bar['low']) <= target_price:
                exit_price = target_price
                bar['date']
                exit_reason = 'target'
                break

    # Time stop if neither hit
    if exit_price is None:
        time_exit_iloc = signal_iloc + 1 + time_stop_bars
        if time_exit_iloc < len(df):
            exit_bar = df.iloc[time_exit_iloc]
            exit_price = float(exit_bar['close'])
            exit_bar['date']
            exit_reason = 'time'
            bars_held = time_stop_bars
        else:
            # Not enough bars, use last available
            exit_bar = df.iloc[-1]
            exit_price = float(exit_bar['close'])
            exit_bar['date']
            exit_reason = 'time_early'
            bars_held = len(df) - (signal_iloc + 2)

    if exit_price is None:
        return None

    # Calculate P&L and R multiple
    if side == 'long':
        pnl_dollar = exit_price - actual_entry
    else:
        pnl_dollar = actual_entry - exit_price

    r_multiple = pnl_dollar / signal_r if signal_r > 0 else 0
    pnl_pct = (pnl_dollar / actual_entry) * 100 if actual_entry > 0 else 0

    return {
        'date': entry_date,
        'symbol': symbol,
        'strategy': strategy,
        'side': side,
        'entry_price': round(actual_entry, 2),
        'stop_price': round(stop_price, 2),
        'target_price': round(target_price, 2),
        'exit_price': round(exit_price, 2),
        'exit_reason': exit_reason,
        'bars_held': bars_held,
        'pnl_dollar': round(pnl_dollar, 2),
        'pnl_pct': round(pnl_pct, 2),
        'r_multiple': round(r_multiple, 2),
        'conf_score': round(float(signal.get('conf_score', 0)), 3),
        '1R': round(signal_r, 2),
    }


# -----------------------------------------------------------------------------
# Backtest Runner
# -----------------------------------------------------------------------------
def run_backtest(
    symbols: List[str],
    start: str,
    end: str,
    min_conf: float = 0.60,
    min_adv_usd: float = 5_000_000,
    use_filters: bool = True,
    ensure_top3: bool = True,
    verbose: bool = False,
) -> Tuple[List[Dict], Dict[str, pd.DataFrame]]:
    """
    Run TOTD backtest over date range.
    Returns (trades list, bars_dict for reference).
    """
    print(f"\nTOTD Backtest: {start} to {end}")
    print(f"Universe: {len(symbols)} symbols")
    print(f"Min confidence: {min_conf}")
    print(f"Min ADV USD: ${min_adv_usd:,.0f}")
    print(f"Filters enabled: {use_filters}")
    print("-" * 60)

    # Get trading days
    trading_days = get_trading_days(start, end)
    print(f"Trading days: {len(trading_days)}")
    if not trading_days:
        return [], {}

    # --- Step 1: Fetch all data for the entire period ---
    print("\nFetching data (this may take a while)...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    all_bars_list = []
    for i, symbol in enumerate(symbols):
        try:
            df = fetch_daily_bars_multi(symbol, start, end, cache_dir=CACHE_DIR)
            if not df.empty and len(df) >= 220:
                df['symbol'] = symbol
                all_bars_list.append(df)
            if verbose and (i + 1) % 100 == 0:
                print(f"  Fetched {i + 1}/{len(symbols)} symbols...")
        except Exception as e:
            if verbose: print(f"  [WARN] {symbol}: {e}", file=sys.stderr)
    
    if not all_bars_list:
        print("Error: No data loaded for any symbols.")
        return [], {}
        
    combined_bars = pd.concat(all_bars_list, ignore_index=True)
    bars_dict = {sym: df for sym, df in combined_bars.groupby('symbol')}
    print(f"Loaded data for {len(bars_dict)} symbols, {len(combined_bars)} total bars.")

    # --- Step 2: Generate ALL signals for the entire period ---
    print("\nGenerating all signals for the period...")
    scanner = get_production_scanner()
    all_signals = scanner.scan_signals_over_time(combined_bars.copy())
    all_signals['date'] = pd.to_datetime(all_signals['timestamp']).dt.strftime('%Y-%m-%d')
    print(f"Generated {len(all_signals)} total signals.")

    # --- Step 3: Loop day-by-day to select TOTD and simulate ---
    trades: List[Dict] = []
    totd_count = 0
    skip_count = 0
    
    spy_bars = fetch_spy_data(end, cache_dir=CACHE_DIR)
    if spy_bars is not None:
        print(f"Loaded SPY data for filters: {len(spy_bars)} bars")

    print("\nRunning daily simulation...")
    for i, day in enumerate(trading_days):
        if verbose or (i + 1) % 20 == 0:
            print(f"  Day {i + 1}/{len(trading_days)}: {day}")

        # Get signals for the current day
        signals_today = all_signals[all_signals['date'] == day].copy()
        if signals_today.empty:
            skip_count += 1
            continue

        # Apply filters
        filtered_signals = apply_filters(
            signals_today, bars_dict, spy_bars, day,
            min_adv_usd=min_adv_usd, use_filters=use_filters, verbose=verbose
        )
        if filtered_signals.empty:
            skip_count += 1
            continue
            
        # Compute confidence
        confident_signals = compute_confidence(filtered_signals, bars_dict, day, verbose=verbose)
        
        # Select Top-3 and TOTD
        top3 = select_top3(confident_signals, ensure_top3=ensure_top3)
        if top3.empty:
            skip_count += 1
            continue
        
        totd = select_totd(top3, min_conf=min_conf)
        if totd is None:
            skip_count += 1
            continue
        
        # Simulate trade
        trade = simulate_trade(totd, bars_dict, day, time_stop_bars=5)
        if trade is not None:
            trades.append(trade)
            totd_count += 1
        else:
            skip_count += 1

    print("\nSimulation complete:")
    print(f"  TOTD trades: {totd_count}")
    print(f"  Skipped days: {skip_count}")

    return trades, bars_dict


# -----------------------------------------------------------------------------
# Metrics Computation
# -----------------------------------------------------------------------------
def compute_metrics(trades: List[Dict]) -> Dict[str, Any]:
    """
    Compute summary metrics from trades.
    """
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_r': 0.0,
            'avg_r': 0.0,
            'best_trade_r': 0.0,
            'worst_trade_r': 0.0,
        }

    df = pd.DataFrame(trades)

    wins = df[df['r_multiple'] > 0]
    losses = df[df['r_multiple'] <= 0]

    total_wins = wins['r_multiple'].sum() if len(wins) > 0 else 0
    total_losses = abs(losses['r_multiple'].sum()) if len(losses) > 0 else 0

    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0

    return {
        'total_trades': len(df),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': round(len(wins) / len(df) * 100, 1) if len(df) > 0 else 0,
        'profit_factor': round(profit_factor, 2),
        'total_r': round(df['r_multiple'].sum(), 2),
        'avg_r': round(df['r_multiple'].mean(), 3) if len(df) > 0 else 0,
        'best_trade_r': round(df['r_multiple'].max(), 2) if len(df) > 0 else 0,
        'worst_trade_r': round(df['r_multiple'].min(), 2) if len(df) > 0 else 0,
        'total_pnl_pct': round(df['pnl_pct'].sum(), 2) if len(df) > 0 else 0,
        'avg_bars_held': round(df['bars_held'].mean(), 1) if len(df) > 0 else 0,
    }


def compute_metrics_by_strategy(trades: List[Dict]) -> Dict[str, Dict[str, Any]]:
    """Compute metrics grouped by strategy."""
    if not trades:
        return {}

    df = pd.DataFrame(trades)
    result = {}

    for strat in df['strategy'].unique():
        strat_trades = df[df['strategy'] == strat].to_dict('records')
        result[strat] = compute_metrics(strat_trades)

    return result


def compute_monthly(trades: List[Dict]) -> pd.DataFrame:
    """Compute monthly breakdown."""
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)
    df['month'] = pd.to_datetime(df['date']).dt.to_period('M').astype(str)

    monthly = df.groupby('month').agg({
        'r_multiple': ['count', 'sum', 'mean'],
        'pnl_pct': 'sum',
    }).round(2)

    monthly.columns = ['trades', 'total_r', 'avg_r', 'total_pnl_pct']

    # Add win rate
    def monthly_win_rate(group):
        return round(len(group[group['r_multiple'] > 0]) / len(group) * 100, 1) if len(group) > 0 else 0

    monthly['win_rate'] = df.groupby('month').apply(monthly_win_rate)

    return monthly.reset_index()


# -----------------------------------------------------------------------------
# Report Writing
# -----------------------------------------------------------------------------
def write_reports(
    trades: List[Dict],
    output_dir: Path,
    start: str,
    end: str,
) -> None:
    """Write CSV, JSON, and HTML reports."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(trades) if trades else pd.DataFrame()

    # Split by strategy
    if not df.empty:
        ict_trades = df[df['strategy'] == 'turtle_soup'].to_dict('records')
        don_trades = df[df['strategy'] == 'ibs_rsi'].to_dict('records')
    else:
        ict_trades = []
        don_trades = []

    # Write per-strategy CSVs
    if ict_trades:
        pd.DataFrame(ict_trades).to_csv(output_dir / 'turtle_soup_trades.csv', index=False)
    if don_trades:
        pd.DataFrame(don_trades).to_csv(output_dir / 'ibs_rsi_trades.csv', index=False)

    # Write all trades
    if not df.empty:
        df.to_csv(output_dir / 'all_trades.csv', index=False)

    # Compute metrics
    ict_metrics = compute_metrics(ict_trades)
    don_metrics = compute_metrics(don_trades)
    all_metrics = compute_metrics(trades)

    # Write summary JSONs
    with open(output_dir / 'turtle_soup_summary.json', 'w') as f:
        json.dump(ict_metrics, f, indent=2)
    with open(output_dir / 'ibs_rsi_summary.json', 'w') as f:
        json.dump(don_metrics, f, indent=2)
    with open(output_dir / 'combined_summary.json', 'w') as f:
        json.dump({'all': all_metrics, 'turtle_soup': ict_metrics, 'ibs_rsi': don_metrics}, f, indent=2)

    # Write monthly CSVs
    if ict_trades:
        ict_monthly = compute_monthly(ict_trades)
        ict_monthly.to_csv(output_dir / 'turtle_soup_monthly.csv', index=False)
    if don_trades:
        don_monthly = compute_monthly(don_trades)
        don_monthly.to_csv(output_dir / 'ibs_rsi_monthly.csv', index=False)

    all_monthly = compute_monthly(trades)
    if not all_monthly.empty:
        all_monthly.to_csv(output_dir / 'combined_monthly.csv', index=False)

    # Write combined summary CSV
    summary_rows = []
    summary_rows.append({'strategy': 'turtle_soup', **ict_metrics})
    summary_rows.append({'strategy': 'ibs_rsi', **don_metrics})
    summary_rows.append({'strategy': 'combined', **all_metrics})
    pd.DataFrame(summary_rows).to_csv(output_dir / 'summary_comparison.csv', index=False)

    # Write HTML report
    year = start[:4]
    html = generate_html_report(trades, ict_metrics, don_metrics, all_metrics, start, end)
    with open(output_dir / f'totd_report_{year}.html', 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\nReports written to: {output_dir}")
    print("  - all_trades.csv")
    print("  - turtle_soup_trades.csv, ibs_rsi_trades.csv")
    print("  - *_summary.json")
    print("  - *_monthly.csv")
    print(f"  - totd_report_{year}.html")


def generate_html_report(
    trades: List[Dict],
    ict_metrics: Dict,
    don_metrics: Dict,
    all_metrics: Dict,
    start: str,
    end: str,
) -> str:
    """Generate formatted HTML report."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>TOTD Backtest Report: {start} to {end}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #1a1a1a; color: #e0e0e0; }}
        h1, h2, h3 {{ color: #00ff00; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #444; padding: 8px 12px; text-align: left; }}
        th {{ background: #2a2a2a; color: #00ff00; }}
        tr:nth-child(even) {{ background: #222; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; padding: 10px 15px; background: #2a2a2a; border-radius: 5px; }}
        .metric-label {{ color: #888; font-size: 12px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #00ff00; }}
        .win {{ color: #00ff00; }}
        .loss {{ color: #ff4444; }}
        .section {{ margin: 30px 0; padding: 20px; background: #222; border-radius: 8px; }}
    </style>
</head>
<body>
    <h1>TOTD Backtest Report</h1>
    <p>Period: {start} to {end}</p>

    <div class="section">
        <h2>Combined Summary</h2>
        <div class="metric">
            <div class="metric-label">Total Trades</div>
            <div class="metric-value">{all_metrics.get('total_trades', 0)}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Win Rate</div>
            <div class="metric-value">{all_metrics.get('win_rate', 0)}%</div>
        </div>
        <div class="metric">
            <div class="metric-label">Total R</div>
            <div class="metric-value {'win' if all_metrics.get('total_r', 0) > 0 else 'loss'}">{all_metrics.get('total_r', 0):+.2f}R</div>
        </div>
        <div class="metric">
            <div class="metric-label">Profit Factor</div>
            <div class="metric-value">{all_metrics.get('profit_factor', 0):.2f}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Avg R/Trade</div>
            <div class="metric-value">{all_metrics.get('avg_r', 0):+.3f}</div>
        </div>
    </div>

    <div class="section">
        <h2>Strategy Comparison</h2>
        <table>
            <tr>
                <th>Strategy</th>
                <th>Trades</th>
                <th>Win Rate</th>
                <th>Total R</th>
                <th>Avg R</th>
                <th>PF</th>
                <th>Best/Worst</th>
            </tr>
            <tr>
                <td>ICT Turtle Soup</td>
                <td>{ict_metrics.get('total_trades', 0)}</td>
                <td>{ict_metrics.get('win_rate', 0)}%</td>
                <td class="{'win' if ict_metrics.get('total_r', 0) > 0 else 'loss'}">{ict_metrics.get('total_r', 0):+.2f}R</td>
                <td>{ict_metrics.get('avg_r', 0):+.3f}</td>
                <td>{ict_metrics.get('profit_factor', 0):.2f}</td>
                <td>{ict_metrics.get('best_trade_r', 0):+.2f} / {ict_metrics.get('worst_trade_r', 0):+.2f}</td>
            </tr>
            <tr>
                <td>IBS+RSI Mean Reversion</td>
                <td>{don_metrics.get('total_trades', 0)}</td>
                <td>{don_metrics.get('win_rate', 0)}%</td>
                <td class="{'win' if don_metrics.get('total_r', 0) > 0 else 'loss'}">{don_metrics.get('total_r', 0):+.2f}R</td>
                <td>{don_metrics.get('avg_r', 0):+.3f}</td>
                <td>{don_metrics.get('profit_factor', 0):.2f}</td>
                <td>{don_metrics.get('best_trade_r', 0):+.2f} / {don_metrics.get('worst_trade_r', 0):+.2f}</td>
            </tr>
        </table>
    </div>
"""

    # Add trades table
    if trades:
        html += """
    <div class="section">
        <h2>Trade Log</h2>
        <table>
            <tr>
                <th>Date</th>
                <th>Symbol</th>
                <th>Strategy</th>
                <th>Entry</th>
                <th>Exit</th>
                <th>Reason</th>
                <th>R Multiple</th>
                <th>P&L %</th>
            </tr>
"""
        for t in trades:
            r_class = 'win' if t['r_multiple'] > 0 else 'loss'
            html += f"""
            <tr>
                <td>{t['date']}</td>
                <td>{t['symbol']}</td>
                <td>{t['strategy']}</td>
                <td>${t['entry_price']:.2f}</td>
                <td>${t['exit_price']:.2f}</td>
                <td>{t['exit_reason']}</td>
                <td class="{r_class}">{t['r_multiple']:+.2f}R</td>
                <td class="{r_class}">{t['pnl_pct']:+.2f}%</td>
            </tr>
"""
        html += """
        </table>
    </div>
"""

    html += f"""
    <footer>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </footer>
</body>
</html>
"""
    return html


# -----------------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="TOTD Backtest - Trade of the Day simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 100 symbols
  python scripts/backtest_totd.py --start 2025-01-01 --end 2025-03-31 --cap 100

  # Full 900-symbol run
  python scripts/backtest_totd.py --start 2025-01-01 --end 2025-12-31 --cap 900

  # With stricter confidence threshold
  python scripts/backtest_totd.py --start 2025-01-01 --end 2025-12-31 --min-conf 0.65
        """,
    )
    ap.add_argument("--universe", type=str, default=str(DEFAULT_UNIVERSE),
                    help="Path to universe CSV (default: optionable_liquid_900.csv)")
    ap.add_argument("--start", type=str, required=True,
                    help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", type=str, required=True,
                    help="End date (YYYY-MM-DD)")
    ap.add_argument("--cap", type=int, default=900,
                    help="Limit universe to N symbols (default: 900)")
    ap.add_argument("--dotenv", type=str, default="./.env",
                    help="Path to .env file")
    ap.add_argument("--min-conf", type=float, default=0.60,
                    help="Minimum confidence for TOTD selection (default: 0.60)")
    ap.add_argument("--min-adv-usd", type=float, default=5_000_000,
                    help="Minimum 60-day ADV in USD (default: 5,000,000)")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Output directory for reports (default: reports/totd_YYYY)")
    ap.add_argument("--no-filters", action="store_true",
                    help="Disable regime/earnings filters")
    ap.add_argument("--no-ensure-top3", action="store_true",
                    help="Don't fill Top-3 from remaining signals")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Verbose output")

    args = ap.parse_args()

    # Load environment
    dotenv_path = Path(args.dotenv)
    if dotenv_path.exists():
        loaded = load_env(dotenv_path)
        if args.verbose:
            print(f"Loaded {len(loaded)} env vars from {dotenv_path}")

    # Check API key
    if not os.getenv("POLYGON_API_KEY"):
        # Try alternate location
        alt_env = Path("C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env")
        if alt_env.exists():
            load_env(alt_env)

    if not os.getenv("POLYGON_API_KEY"):
        print("Error: POLYGON_API_KEY not set", file=sys.stderr)
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

    print(f"Loaded {len(symbols)} symbols from {universe_path}")

    # Run backtest
    trades, _ = run_backtest(
        symbols=symbols,
        start=args.start,
        end=args.end,
        min_conf=args.min_conf,
        min_adv_usd=args.min_adv_usd,
        use_filters=not args.no_filters,
        ensure_top3=not args.no_ensure_top3,
        verbose=args.verbose,
    )

    # Compute and display metrics
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    all_metrics = compute_metrics(trades)
    strat_metrics = compute_metrics_by_strategy(trades)

    print(f"\nCombined: {all_metrics['total_trades']} trades")
    print(f"  Win Rate: {all_metrics['win_rate']}%")
    print(f"  Total R:  {all_metrics['total_r']:+.2f}")
    print(f"  Avg R:    {all_metrics['avg_r']:+.3f}")
    print(f"  PF:       {all_metrics['profit_factor']:.2f}")

    for strat, metrics in strat_metrics.items():
        print(f"\n{strat.upper()}: {metrics['total_trades']} trades")
        print(f"  Win Rate: {metrics['win_rate']}%")
        print(f"  Total R:  {metrics['total_r']:+.2f}")
        print(f"  Avg R:    {metrics['avg_r']:+.3f}")

    # Write reports
    year = args.start[:4]
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "reports" / f"totd_{year}"
    write_reports(trades, output_dir, args.start, args.end)

    return 0


if __name__ == "__main__":
    sys.exit(main())




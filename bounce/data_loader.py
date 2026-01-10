"""
Data Loader Module for Bounce Analysis

Loads OHLCV data for bounce analysis using:
- Polygon (primary, split-adjusted)
- yfinance/Stooq (fallback + validation)

Reuses existing data providers from data/providers/
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.universe.loader import load_universe
from data.providers.polygon_eod import fetch_daily_bars_polygon
from data.providers.multi_source import fetch_daily_bars_resilient

# Try to import free providers for validation
try:
    from data.providers.stooq_eod import fetch_daily_bars as fetch_stooq
    HAS_STOOQ = True
except ImportError:
    HAS_STOOQ = False

try:
    from data.providers.yfinance_eod import fetch_daily_bars as fetch_yfinance
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


def load_ticker_data(
    symbol: str,
    start_date: str,
    end_date: str,
    cache_dir: Optional[Path] = None,
    provider_order: List[str] = None,
) -> Tuple[Optional[pd.DataFrame], Dict]:
    """
    Load OHLCV data for a single ticker.

    Args:
        symbol: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        cache_dir: Directory for cached data
        provider_order: Order of providers to try (default: yfinance first if no Polygon API key)

    Returns:
        df: DataFrame with columns [date, open, high, low, close, volume] or None
        metadata: {source_used, adjustment_basis, bars_count, date_range, error}
    """
    if provider_order is None:
        # Check if Polygon API key is available
        import os
        if os.getenv('POLYGON_API_KEY'):
            provider_order = ["polygon", "yfinance", "stooq"]
        else:
            # Use yfinance first when no Polygon API key
            provider_order = ["yfinance", "stooq", "polygon"]

    if cache_dir is None:
        cache_dir = PROJECT_ROOT / "cache" / "polygon"

    metadata = {
        "symbol": symbol,
        "source_used": None,
        "adjustment_basis": "split_adjusted",
        "bars_count": 0,
        "date_range": (None, None),
        "error": None,
    }

    df = None

    # Try resilient fetcher first (handles fallback internally)
    try:
        df = fetch_daily_bars_resilient(
            symbol=symbol,
            start=start_date,
            end=end_date,
            cache_dir=cache_dir,
            provider_order=provider_order,
        )

        if df is not None and len(df) > 0:
            # Normalize column names
            df = _normalize_columns(df)

            # Ensure date column - handle timezone-aware timestamps
            if 'date' not in df.columns and 'timestamp' in df.columns:
                # Convert timezone-aware timestamps to UTC then to date
                ts = pd.to_datetime(df['timestamp'], utc=True)
                df['date'] = ts.dt.date
            elif 'date' not in df.columns:
                ts = pd.to_datetime(df.index, utc=True)
                df['date'] = ts.date

            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)

            metadata["source_used"] = "polygon"  # Primary assumed
            metadata["bars_count"] = len(df)
            metadata["date_range"] = (
                df['date'].min() if len(df) > 0 else None,
                df['date'].max() if len(df) > 0 else None,
            )

            return df, metadata

    except Exception as e:
        metadata["error"] = str(e)

    # Fallback: try Polygon directly
    try:
        df = fetch_daily_bars_polygon(
            symbol=symbol,
            start=start_date,
            end=end_date,
            cache_dir=cache_dir,
        )

        if df is not None and len(df) > 0:
            df = _normalize_columns(df)
            if 'date' not in df.columns and 'timestamp' in df.columns:
                # Convert timezone-aware timestamps to UTC then to date
                ts = pd.to_datetime(df['timestamp'], utc=True)
                df['date'] = ts.dt.date
            df = df.sort_values('date').reset_index(drop=True)

            metadata["source_used"] = "polygon"
            metadata["bars_count"] = len(df)
            metadata["date_range"] = (df['date'].min(), df['date'].max())
            metadata["error"] = None

            return df, metadata

    except Exception as e:
        metadata["error"] = str(e)

    return None, metadata


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase standard."""
    df = df.copy()

    # Standard column mapping
    col_map = {
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume',
        'Timestamp': 'timestamp',
        'Date': 'date',
        'OPEN': 'open',
        'HIGH': 'high',
        'LOW': 'low',
        'CLOSE': 'close',
        'VOLUME': 'volume',
    }

    df.columns = [col_map.get(c, c.lower()) for c in df.columns]

    # Ensure required columns exist
    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            if col == 'volume':
                df['volume'] = 0
            else:
                df[col] = np.nan

    return df


def validate_against_stooq(
    df_primary: pd.DataFrame,
    symbol: str,
    start_date: str,
    end_date: str,
) -> Dict:
    """
    Cross-validate primary data against Stooq (free source).

    Args:
        df_primary: Primary DataFrame (from Polygon)
        symbol: Stock ticker
        start_date: Start date
        end_date: End date

    Returns:
        {
            "validated": bool,
            "mismatch_flagged": bool,
            "rejected": bool,
            "mismatch_pct": float,
            "overlap_days": int,
            "details": str,
        }
    """
    result = {
        "validated": False,
        "mismatch_flagged": False,
        "rejected": False,
        "mismatch_pct": 0.0,
        "overlap_days": 0,
        "details": "",
    }

    if not HAS_STOOQ:
        result["details"] = "Stooq provider not available"
        return result

    try:
        df_stooq = fetch_stooq(symbol, start_date, end_date)

        if df_stooq is None or len(df_stooq) == 0:
            result["details"] = "No Stooq data available"
            return result

        df_stooq = _normalize_columns(df_stooq)

        # Ensure date columns
        if 'date' not in df_primary.columns:
            df_primary = df_primary.copy()
            df_primary['date'] = pd.to_datetime(df_primary['timestamp']).dt.date
        if 'date' not in df_stooq.columns:
            df_stooq['date'] = pd.to_datetime(df_stooq.index).date

        # Find overlapping dates
        primary_dates = set(pd.to_datetime(df_primary['date']).dt.date)
        stooq_dates = set(pd.to_datetime(df_stooq['date']).dt.date)
        overlap_dates = primary_dates & stooq_dates

        result["overlap_days"] = len(overlap_dates)

        if len(overlap_dates) < 10:
            result["details"] = f"Insufficient overlap ({len(overlap_dates)} days)"
            return result

        # Merge on date
        df_primary['date_key'] = pd.to_datetime(df_primary['date']).dt.date
        df_stooq['date_key'] = pd.to_datetime(df_stooq['date']).dt.date

        merged = pd.merge(
            df_primary[['date_key', 'close']],
            df_stooq[['date_key', 'close']],
            on='date_key',
            suffixes=('_primary', '_stooq'),
        )

        # Calculate mismatch
        merged['pct_diff'] = np.abs(
            merged['close_primary'] - merged['close_stooq']
        ) / merged['close_primary']

        # Count days with >5% mismatch
        mismatch_days = (merged['pct_diff'] > 0.05).sum()
        mismatch_pct = mismatch_days / len(merged) if len(merged) > 0 else 0

        result["mismatch_pct"] = mismatch_pct
        result["validated"] = True

        # Flag if >5% of days have >5% mismatch
        if mismatch_pct > 0.05:
            result["mismatch_flagged"] = True
            result["details"] = f"Flagged: {mismatch_pct:.1%} of days have >5% price diff"

        # Reject if >20% of days have >5% mismatch
        if mismatch_pct > 0.20:
            result["rejected"] = True
            result["details"] = f"REJECTED: {mismatch_pct:.1%} of days have >5% price diff"

        return result

    except Exception as e:
        result["details"] = f"Validation error: {e}"
        return result


def load_universe_data(
    symbols: Optional[List[str]] = None,
    years: int = 10,
    max_workers: int = 4,
    validate_sample: int = 25,
    cache_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Load data for entire universe with parallel fetching.

    Args:
        symbols: List of symbols (default: load from universe file)
        years: Number of years of history
        max_workers: Parallel workers for fetching
        validate_sample: Number of tickers to cross-validate (seed=42)
        cache_dir: Cache directory
        verbose: Print progress

    Returns:
        ticker_data: Dict[symbol, DataFrame]
        quality_report: DataFrame with per-ticker quality metrics
    """
    if symbols is None:
        symbols = load_universe(cap=900)

    if cache_dir is None:
        cache_dir = PROJECT_ROOT / "cache" / "polygon"

    # Calculate date range
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=years * 365 + 30)).strftime("%Y-%m-%d")

    if verbose:
        print(f"Loading data for {len(symbols)} tickers...")
        print(f"Date range: {start_date} to {end_date} ({years} years)")

    ticker_data = {}
    quality_records = []

    # Select validation sample (deterministic with seed=42)
    random.seed(42)
    validation_symbols = set(random.sample(symbols, min(validate_sample, len(symbols))))

    def fetch_ticker(symbol: str) -> Tuple[str, Optional[pd.DataFrame], Dict]:
        df, metadata = load_ticker_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            cache_dir=cache_dir,
        )

        # Run validation for sample
        validation_result = None
        if symbol in validation_symbols and df is not None:
            validation_result = validate_against_stooq(
                df, symbol, start_date, end_date
            )

        return symbol, df, metadata, validation_result

    # Parallel fetch
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_ticker, s): s for s in symbols}

        for future in as_completed(futures):
            symbol, df, metadata, validation = future.result()
            completed += 1

            if df is not None and len(df) > 0:
                ticker_data[symbol] = df

            # Build quality record
            record = {
                "symbol": symbol,
                "source_used": metadata.get("source_used"),
                "bars_count": metadata.get("bars_count", 0),
                "start_date": metadata.get("date_range", (None, None))[0],
                "end_date": metadata.get("date_range", (None, None))[1],
                "error": metadata.get("error"),
                "validated": validation is not None,
                "mismatch_flagged": validation.get("mismatch_flagged") if validation else None,
                "rejected": validation.get("rejected") if validation else None,
                "mismatch_pct": validation.get("mismatch_pct") if validation else None,
            }
            quality_records.append(record)

            if verbose and completed % 100 == 0:
                print(f"  Loaded {completed}/{len(symbols)} tickers...")

    quality_report = pd.DataFrame(quality_records)

    if verbose:
        print("\nData Loading Complete:")
        print(f"  Tickers attempted: {len(symbols)}")
        print(f"  Tickers loaded: {len(ticker_data)}")
        print(f"  Tickers failed: {len(symbols) - len(ticker_data)}")

        if 'validated' in quality_report.columns:
            validated = quality_report['validated'].sum()
            flagged = quality_report['mismatch_flagged'].sum() if 'mismatch_flagged' in quality_report else 0
            rejected = quality_report['rejected'].sum() if 'rejected' in quality_report else 0
            print(f"  Validated sample: {validated}/{validate_sample}")
            print(f"  Mismatch flagged: {flagged}")
            print(f"  Rejected (bad data): {rejected}")

    return ticker_data, quality_report


def get_data_health_summary(quality_report: pd.DataFrame) -> Dict:
    """
    Generate DATA HEALTH summary block.

    Args:
        quality_report: Quality report DataFrame from load_universe_data

    Returns:
        Dict with health metrics
    """
    total = len(quality_report)
    loaded = quality_report['bars_count'].gt(0).sum()
    failed = total - loaded

    # Source breakdown
    polygon_only = (quality_report['source_used'] == 'polygon').sum()
    fallback_used = (quality_report['source_used'].isin(['yfinance', 'stooq'])).sum()
    mixed = (quality_report['source_used'] == 'mixed').sum()

    # Validation stats
    validated = quality_report['validated'].sum() if 'validated' in quality_report.columns else 0
    flagged = quality_report['mismatch_flagged'].sum() if 'mismatch_flagged' in quality_report.columns else 0
    rejected = quality_report['rejected'].sum() if 'rejected' in quality_report.columns else 0

    # Coverage stats
    bars = quality_report['bars_count']
    years_coverage = bars / 252  # Approximate trading days per year

    return {
        "tickers_attempted": total,
        "tickers_loaded": loaded,
        "tickers_failed": failed,
        "polygon_only": polygon_only,
        "fallback_used": fallback_used,
        "mixed_sources": mixed,
        "validated_sample": validated,
        "mismatch_flagged": flagged,
        "rejected_bad_data": rejected,
        "coverage_years_min": years_coverage.min() if len(years_coverage) > 0 else 0,
        "coverage_years_median": years_coverage.median() if len(years_coverage) > 0 else 0,
        "coverage_years_max": years_coverage.max() if len(years_coverage) > 0 else 0,
        "earliest_date": quality_report['start_date'].dropna().min() if 'start_date' in quality_report.columns else None,
        "latest_date": quality_report['end_date'].dropna().max() if 'end_date' in quality_report.columns else None,
    }

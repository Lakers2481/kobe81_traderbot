"""
US Treasury Yield Curve Data Provider

FREE - No API key required
Source: https://home.treasury.gov/resource-center/data-chart-center/interest-rates/

Provides:
- Daily Treasury yield curve rates (all tenors)
- Historical yield curve data
- Yield curve inversions detection
- Real-time par yield curve

Author: Kobe Trading System
Created: 2026-01-04
"""

import os
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from io import StringIO

import pandas as pd
import requests

from core.structured_log import get_logger

logger = get_logger(__name__)


# Treasury tenors in order
TENORS = ["1M", "2M", "3M", "4M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
TENOR_MONTHS = {
    "1M": 1, "2M": 2, "3M": 3, "4M": 4, "6M": 6,
    "1Y": 12, "2Y": 24, "3Y": 36, "5Y": 60, "7Y": 84,
    "10Y": 120, "20Y": 240, "30Y": 360
}


class TreasuryYieldProvider:
    """
    US Treasury yield curve data provider.

    Features:
    - Direct fetch from Treasury.gov XML feeds
    - CSV historical data downloads
    - 24-hour caching
    - Automatic inversion detection
    """

    # Treasury.gov XML feed for daily rates
    DAILY_XML_URL = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv"

    # Alternative: Treasury API (also free)
    TREASURY_API_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/avg_interest_rates"

    CACHE_DIR = Path("data/cache/treasury")
    CACHE_TTL_HOURS = 24

    def __init__(self):
        """Initialize Treasury yield provider."""
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, year: int) -> Path:
        """Get cache file path for a year."""
        return self.CACHE_DIR / f"yields_{year}.csv"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file exists and is fresh."""
        if not cache_path.exists():
            return False

        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age_hours = (datetime.now() - mtime).total_seconds() / 3600
        return age_hours < self.CACHE_TTL_HOURS

    def get_yields_for_year(
        self,
        year: int,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch Treasury yields for a specific year.

        Args:
            year: Year to fetch (e.g., 2024)
            force_refresh: Bypass cache

        Returns:
            DataFrame with date index and tenor columns
        """
        cache_path = self._cache_path(year)

        # Check cache
        if not force_refresh and self._is_cache_valid(cache_path):
            logger.debug(f"Treasury cache hit: {year}")
            df = pd.read_csv(cache_path, parse_dates=["Date"])
            df.set_index("Date", inplace=True)
            return df

        # Fetch from Treasury.gov
        url = self.DAILY_XML_URL.format(year=year)

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Parse CSV
            df = pd.read_csv(StringIO(response.text))

            # Standardize column names
            df.columns = [c.strip() for c in df.columns]

            # Find and rename date column
            date_col = None
            for col in df.columns:
                if "date" in col.lower():
                    date_col = col
                    break

            if date_col:
                df.rename(columns={date_col: "Date"}, inplace=True)
                df["Date"] = pd.to_datetime(df["Date"])
                df.set_index("Date", inplace=True)

            # Rename tenor columns to standard format
            rename_map = {}
            for col in df.columns:
                col_lower = col.lower()
                if "1 mo" in col_lower or "1 month" in col_lower:
                    rename_map[col] = "1M"
                elif "2 mo" in col_lower or "2 month" in col_lower:
                    rename_map[col] = "2M"
                elif "3 mo" in col_lower or "3 month" in col_lower:
                    rename_map[col] = "3M"
                elif "4 mo" in col_lower or "4 month" in col_lower:
                    rename_map[col] = "4M"
                elif "6 mo" in col_lower or "6 month" in col_lower:
                    rename_map[col] = "6M"
                elif "1 yr" in col_lower or "1 year" in col_lower:
                    rename_map[col] = "1Y"
                elif "2 yr" in col_lower or "2 year" in col_lower:
                    rename_map[col] = "2Y"
                elif "3 yr" in col_lower or "3 year" in col_lower:
                    rename_map[col] = "3Y"
                elif "5 yr" in col_lower or "5 year" in col_lower:
                    rename_map[col] = "5Y"
                elif "7 yr" in col_lower or "7 year" in col_lower:
                    rename_map[col] = "7Y"
                elif "10 yr" in col_lower or "10 year" in col_lower:
                    rename_map[col] = "10Y"
                elif "20 yr" in col_lower or "20 year" in col_lower:
                    rename_map[col] = "20Y"
                elif "30 yr" in col_lower or "30 year" in col_lower:
                    rename_map[col] = "30Y"

            df.rename(columns=rename_map, inplace=True)

            # Keep only tenor columns
            valid_cols = [c for c in df.columns if c in TENORS]
            df = df[valid_cols]

            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Sort by date
            df.sort_index(inplace=True)

            # Cache the data
            df.to_csv(cache_path)
            logger.info(f"Treasury yields fetched for {year}: {len(df)} days")

            return df

        except requests.RequestException as e:
            logger.error(f"Treasury fetch failed for {year}: {e}")

            # Fall back to cache if available
            if cache_path.exists():
                logger.warning(f"Using stale cache for {year}")
                df = pd.read_csv(cache_path, parse_dates=["Date"])
                df.set_index("Date", inplace=True)
                return df

            return pd.DataFrame()

    def get_yields(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch Treasury yields for a date range.

        Args:
            start_date: Start date (YYYY-MM-DD), defaults to 5 years ago
            end_date: End date (YYYY-MM-DD), defaults to today
            force_refresh: Bypass cache

        Returns:
            DataFrame with date index and tenor columns
        """
        if not start_date:
            start_date = (datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Fetch data for each year
        all_data = []
        for year in range(start_dt.year, end_dt.year + 1):
            df = self.get_yields_for_year(year, force_refresh)
            if not df.empty:
                all_data.append(df)

        if not all_data:
            return pd.DataFrame()

        # Combine all years
        combined = pd.concat(all_data)
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)

        # Filter to date range
        combined = combined[(combined.index >= start_dt) & (combined.index <= end_dt)]

        return combined

    def get_latest_curve(self) -> Dict[str, float]:
        """
        Get the most recent yield curve.

        Returns:
            Dict mapping tenor to yield (e.g., {'1M': 5.25, '10Y': 4.25})
        """
        df = self.get_yields_for_year(datetime.now().year)

        if df.empty:
            # Try previous year if current year has no data
            df = self.get_yields_for_year(datetime.now().year - 1)

        if df.empty:
            return {}

        latest = df.iloc[-1]
        return {tenor: float(latest[tenor]) for tenor in latest.index if pd.notna(latest[tenor])}

    def get_curve_at_date(self, date: str) -> Dict[str, float]:
        """
        Get yield curve for a specific date.

        Args:
            date: Date string (YYYY-MM-DD)

        Returns:
            Dict mapping tenor to yield
        """
        dt = pd.to_datetime(date)
        df = self.get_yields_for_year(dt.year)

        if df.empty:
            return {}

        # Find the closest date <= requested date
        valid_dates = df.index[df.index <= dt]
        if len(valid_dates) == 0:
            return {}

        row = df.loc[valid_dates[-1]]
        return {tenor: float(row[tenor]) for tenor in row.index if pd.notna(row[tenor])}

    def calculate_spreads(self, curve: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate common yield curve spreads.

        Args:
            curve: Yield curve dict from get_latest_curve()

        Returns:
            Dict with spread calculations
        """
        spreads = {}

        # 10Y - 2Y spread (most watched)
        if "10Y" in curve and "2Y" in curve:
            spreads["10Y_2Y"] = curve["10Y"] - curve["2Y"]

        # 10Y - 3M spread (recession indicator)
        if "10Y" in curve and "3M" in curve:
            spreads["10Y_3M"] = curve["10Y"] - curve["3M"]

        # 2Y - 3M spread (near-term)
        if "2Y" in curve and "3M" in curve:
            spreads["2Y_3M"] = curve["2Y"] - curve["3M"]

        # 30Y - 10Y spread (long end)
        if "30Y" in curve and "10Y" in curve:
            spreads["30Y_10Y"] = curve["30Y"] - curve["10Y"]

        # 5Y - 2Y spread (belly)
        if "5Y" in curve and "2Y" in curve:
            spreads["5Y_2Y"] = curve["5Y"] - curve["2Y"]

        return spreads

    def detect_inversions(self, curve: Dict[str, float]) -> Dict[str, Any]:
        """
        Detect yield curve inversions.

        Args:
            curve: Yield curve dict

        Returns:
            Dict with inversion analysis
        """
        spreads = self.calculate_spreads(curve)

        inversions = []
        for spread_name, value in spreads.items():
            if value < 0:
                inversions.append(spread_name)

        # Severity based on how many inversions
        if len(inversions) == 0:
            severity = "NORMAL"
        elif len(inversions) <= 2:
            severity = "MILD_INVERSION"
        else:
            severity = "SEVERE_INVERSION"

        # Special case: 10Y-2Y is most important
        is_classic_inversion = "10Y_2Y" in inversions

        return {
            "spreads": spreads,
            "inversions": inversions,
            "severity": severity,
            "is_classic_inversion": is_classic_inversion,
            "curve_shape": self._classify_curve_shape(curve),
        }

    def _classify_curve_shape(self, curve: Dict[str, float]) -> str:
        """
        Classify the yield curve shape.

        Returns:
            One of: NORMAL, FLAT, INVERTED, HUMPED, STEEP
        """
        if len(curve) < 3:
            return "UNKNOWN"

        short = curve.get("3M") or curve.get("1M") or curve.get("6M")
        mid = curve.get("2Y") or curve.get("5Y")
        long = curve.get("10Y") or curve.get("30Y")

        if short is None or mid is None or long is None:
            return "UNKNOWN"

        short_mid = mid - short
        mid_long = long - mid
        short_long = long - short

        # Inverted: long-term rates lower than short-term
        if short_long < -0.25:
            return "INVERTED"

        # Flat: small difference across curve
        if abs(short_long) < 0.50:
            return "FLAT"

        # Humped: middle higher than both ends
        if mid > short and mid > long:
            return "HUMPED"

        # Steep: large positive slope
        if short_long > 2.0:
            return "STEEP"

        # Normal: gradual upward slope
        return "NORMAL"

    def get_curve_history(
        self,
        tenor: str = "10Y",
        days: int = 252
    ) -> pd.Series:
        """
        Get historical rates for a specific tenor.

        Args:
            tenor: Tenor to fetch (e.g., '10Y')
            days: Number of trading days

        Returns:
            Series with date index and yield values
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(days * 1.5))  # Account for weekends

        df = self.get_yields(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

        if df.empty or tenor not in df.columns:
            return pd.Series(dtype=float)

        return df[tenor].dropna().tail(days)


# Singleton instance
_provider: Optional[TreasuryYieldProvider] = None


def get_treasury_provider() -> TreasuryYieldProvider:
    """Get or create singleton Treasury provider."""
    global _provider
    if _provider is None:
        _provider = TreasuryYieldProvider()
    return _provider


# Convenience functions
def get_yield_curve() -> Dict[str, float]:
    """Get latest Treasury yield curve."""
    return get_treasury_provider().get_latest_curve()


def get_curve_spreads() -> Dict[str, float]:
    """Get yield curve spreads."""
    curve = get_treasury_provider().get_latest_curve()
    return get_treasury_provider().calculate_spreads(curve)


def detect_inversions() -> Dict[str, Any]:
    """Detect yield curve inversions."""
    curve = get_treasury_provider().get_latest_curve()
    return get_treasury_provider().detect_inversions(curve)


if __name__ == "__main__":
    # Demo usage
    provider = TreasuryYieldProvider()

    print("=== Treasury Yield Curve Provider Demo ===\n")

    # Latest curve
    print("Latest Yield Curve:")
    curve = provider.get_latest_curve()
    for tenor in TENORS:
        if tenor in curve:
            print(f"  {tenor:>4s}: {curve[tenor]:.2f}%")

    print("\nSpreads:")
    spreads = provider.calculate_spreads(curve)
    for name, value in sorted(spreads.items()):
        status = "INVERTED" if value < 0 else "normal"
        print(f"  {name}: {value:+.2f}% ({status})")

    print("\nInversion Analysis:")
    analysis = provider.detect_inversions(curve)
    print(f"  Curve Shape: {analysis['curve_shape']}")
    print(f"  Severity: {analysis['severity']}")
    print(f"  Classic 10Y-2Y Inversion: {analysis['is_classic_inversion']}")
    if analysis["inversions"]:
        print(f"  Inverted Spreads: {', '.join(analysis['inversions'])}")

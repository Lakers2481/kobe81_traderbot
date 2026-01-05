"""
FRED (Federal Reserve Economic Data) Macro Data Provider

FREE API - Requires free registration at https://fred.stlouisfed.org/docs/api/api_key.html
Set FRED_API_KEY environment variable with your API key.

Provides:
- Interest rates (Fed Funds, 10Y, 2Y, etc.)
- Inflation (CPI, PCE)
- GDP growth
- Unemployment
- Leading indicators

Author: Kobe Trading System
Created: 2026-01-04
"""

import os
import json

# Load .env BEFORE checking environment variables
from pathlib import Path
try:
    from dotenv import load_dotenv
    # Load from project root
    _env_path = Path(__file__).parent.parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass  # dotenv not installed, rely on system env vars
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd

from core.structured_log import get_logger

logger = get_logger(__name__)

# Try to import fredapi
_fredapi_available = False
try:
    from fredapi import Fred
    _fredapi_available = True
except ImportError:
    logger.warning("fredapi not installed. Install with: pip install fredapi")


# FRED Series IDs for key macro indicators
FRED_SERIES = {
    # Interest Rates
    "FEDFUNDS": "Federal Funds Effective Rate",
    "DGS10": "10-Year Treasury Constant Maturity",
    "DGS2": "2-Year Treasury Constant Maturity",
    "DGS5": "5-Year Treasury Constant Maturity",
    "DGS30": "30-Year Treasury Constant Maturity",
    "DGS3MO": "3-Month Treasury Bill",
    "T10Y2Y": "10Y-2Y Treasury Spread",
    "T10Y3M": "10Y-3M Treasury Spread",

    # Inflation
    "CPIAUCSL": "Consumer Price Index (All Urban)",
    "PCEPI": "Personal Consumption Expenditures Price Index",
    "CPILFESL": "Core CPI (Less Food & Energy)",
    "T5YIE": "5-Year Breakeven Inflation",
    "T10YIE": "10-Year Breakeven Inflation",

    # GDP & Growth
    "GDP": "Gross Domestic Product",
    "GDPC1": "Real Gross Domestic Product",
    "A191RL1Q225SBEA": "Real GDP Growth Rate",

    # Employment
    "UNRATE": "Unemployment Rate",
    "PAYEMS": "Total Nonfarm Payrolls",
    "ICSA": "Initial Jobless Claims",
    "CCSA": "Continued Jobless Claims",

    # Leading Indicators
    "USSLIND": "Leading Index for the US",
    "UMCSENT": "University of Michigan Consumer Sentiment",
    "RSXFS": "Retail Sales Ex Food Services",

    # Financial Conditions
    "BAMLH0A0HYM2": "High Yield Option-Adjusted Spread",
    "VIXCLS": "CBOE Volatility Index",
    "DTWEXBGS": "Trade Weighted US Dollar Index",
}

# Core macro indicators for daily monitoring
CORE_INDICATORS = [
    "FEDFUNDS", "DGS10", "DGS2", "T10Y2Y",
    "CPIAUCSL", "UNRATE", "VIXCLS"
]


class FREDMacroProvider:
    """
    FRED API client for macro economic data.

    Features:
    - Uses official fredapi library for authentication
    - 24-hour caching to respect API limits
    - Fallback to cached data on API failure
    - DataFrame output with proper indexing
    """

    CACHE_DIR = Path("data/cache/fred")
    CACHE_TTL_HOURS = 24

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FRED provider.

        Args:
            api_key: FRED API key. If not provided, uses env var FRED_API_KEY.
                    Get free key at https://fred.stlouisfed.org/docs/api/api_key.html
        """
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._fred = None

        if not self.api_key:
            logger.warning(
                "No FRED API key found. Get free key at: "
                "https://fred.stlouisfed.org/docs/api/api_key.html "
                "and set FRED_API_KEY environment variable."
            )
        elif _fredapi_available:
            self._fred = Fred(api_key=self.api_key)
            logger.info("FRED provider initialized with API key")

    def _cache_path(self, series_id: str) -> Path:
        """Get cache file path for a series."""
        return self.CACHE_DIR / f"{series_id.lower()}.json"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file exists and is fresh."""
        if not cache_path.exists():
            return False

        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age_hours = (datetime.now() - mtime).total_seconds() / 3600
        return age_hours < self.CACHE_TTL_HOURS

    def _load_cache(self, cache_path: Path) -> Optional[Dict]:
        """Load data from cache file."""
        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")
            return None

    def _save_cache(self, cache_path: Path, data: Dict) -> None:
        """Save data to cache file."""
        try:
            with open(cache_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

    def get_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch FRED series data.

        Args:
            series_id: FRED series ID (e.g., 'FEDFUNDS', 'DGS10')
            start_date: Start date (YYYY-MM-DD), defaults to 5 years ago
            end_date: End date (YYYY-MM-DD), defaults to today
            force_refresh: Bypass cache and fetch fresh data

        Returns:
            DataFrame with columns: date, value, series_id
        """
        series_id = series_id.upper()
        cache_path = self._cache_path(series_id)

        # Check cache first
        if not force_refresh and self._is_cache_valid(cache_path):
            cached = self._load_cache(cache_path)
            if cached and "observations" in cached:
                logger.debug(f"FRED cache hit: {series_id}")
                df = pd.DataFrame(cached["observations"])
                df["date"] = pd.to_datetime(df["date"])
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                df["series_id"] = series_id
                return df

        # Check if we can fetch from API
        if not self._fred:
            logger.error(
                f"Cannot fetch {series_id}: No FRED API key. "
                "Get free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
            )
            # Try stale cache
            if cache_path.exists():
                cached = self._load_cache(cache_path)
                if cached and "observations" in cached:
                    logger.warning(f"Using stale cache for {series_id}")
                    df = pd.DataFrame(cached["observations"])
                    df["date"] = pd.to_datetime(df["date"])
                    df["value"] = pd.to_numeric(df["value"], errors="coerce")
                    df["series_id"] = series_id
                    return df
            return pd.DataFrame(columns=["date", "value", "series_id"])

        # Parse dates
        if not start_date:
            start_dt = datetime.now() - timedelta(days=5*365)
        else:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")

        if not end_date:
            end_dt = datetime.now()
        else:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        try:
            # Use fredapi library
            series = self._fred.get_series(
                series_id,
                observation_start=start_dt,
                observation_end=end_dt
            )

            if series is None or series.empty:
                logger.warning(f"FRED returned no data for {series_id}")
                return pd.DataFrame(columns=["date", "value", "series_id"])

            # Convert to DataFrame
            df = pd.DataFrame({
                "date": series.index,
                "value": series.values,
                "series_id": series_id
            })
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")

            # Cache the response
            cache_data = {
                "observations": df[["date", "value"]].assign(
                    date=df["date"].dt.strftime("%Y-%m-%d")
                ).to_dict("records")
            }
            self._save_cache(cache_path, cache_data)
            logger.info(f"FRED fetched {series_id}: {len(df)} observations")

            return df

        except Exception as e:
            logger.error(f"FRED API request failed for {series_id}: {e}")

            # Fall back to stale cache
            if cache_path.exists():
                cached = self._load_cache(cache_path)
                if cached and "observations" in cached:
                    logger.warning(f"Using stale cache for {series_id}")
                    df = pd.DataFrame(cached["observations"])
                    df["date"] = pd.to_datetime(df["date"])
                    df["value"] = pd.to_numeric(df["value"], errors="coerce")
                    df["series_id"] = series_id
                    return df

            return pd.DataFrame(columns=["date", "value", "series_id"])

    def get_multiple_series(
        self,
        series_ids: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch multiple FRED series and combine into single DataFrame.

        Args:
            series_ids: List of FRED series IDs
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            force_refresh: Bypass cache

        Returns:
            DataFrame with series as columns, date as index
        """
        all_data = []

        for series_id in series_ids:
            df = self.get_series(series_id, start_date, end_date, force_refresh)
            if not df.empty:
                df = df[["date", "value"]].rename(columns={"value": series_id})
                all_data.append(df.set_index("date"))

        if not all_data:
            return pd.DataFrame()

        # Combine all series
        combined = all_data[0]
        for df in all_data[1:]:
            combined = combined.join(df, how="outer")

        combined = combined.sort_index()
        return combined

    def get_core_indicators(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch all core macro indicators.

        Returns:
            DataFrame with core indicators as columns
        """
        return self.get_multiple_series(
            CORE_INDICATORS, start_date, end_date, force_refresh
        )

    def get_yield_curve(
        self,
        as_of_date: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Get yield curve snapshot for a specific date.

        Args:
            as_of_date: Date (YYYY-MM-DD), defaults to latest available

        Returns:
            Dict mapping tenor to yield (e.g., {'3M': 5.25, '2Y': 4.50, ...})
        """
        tenors = {
            "DGS3MO": "3M",
            "DGS2": "2Y",
            "DGS5": "5Y",
            "DGS10": "10Y",
            "DGS30": "30Y",
        }

        curve = {}
        for series_id, tenor in tenors.items():
            df = self.get_series(series_id)
            if not df.empty:
                if as_of_date:
                    row = df[df["date"] <= as_of_date].iloc[-1] if len(df) > 0 else None
                else:
                    row = df.iloc[-1] if len(df) > 0 else None

                if row is not None and pd.notna(row["value"]):
                    curve[tenor] = float(row["value"])

        return curve

    def get_yield_curve_slope(self) -> Dict[str, float]:
        """
        Calculate yield curve slopes (spread between tenors).

        Returns:
            Dict with '10Y-2Y', '10Y-3M' spreads and inversion flag
        """
        curve = self.get_yield_curve()

        result = {
            "10Y_2Y_spread": None,
            "10Y_3M_spread": None,
            "is_inverted": False,
        }

        if "10Y" in curve and "2Y" in curve:
            result["10Y_2Y_spread"] = curve["10Y"] - curve["2Y"]
            result["is_inverted"] = result["10Y_2Y_spread"] < 0

        if "10Y" in curve and "3M" in curve:
            result["10Y_3M_spread"] = curve["10Y"] - curve["3M"]

        return result

    def get_macro_regime(self) -> Dict[str, Any]:
        """
        Determine current macro regime based on key indicators.

        Returns:
            Dict with regime classification and supporting data
        """
        # Get latest data for key indicators
        df = self.get_core_indicators()

        if df.empty:
            return {"regime": "UNKNOWN", "confidence": 0.0, "indicators": {}}

        latest = df.iloc[-1]

        # Get yield curve data
        slope = self.get_yield_curve_slope()

        # Classify regime
        regime = "NEUTRAL"
        signals = []

        # Yield curve signal
        if slope.get("is_inverted"):
            signals.append("INVERTED_CURVE")
            regime = "CONTRACTIONARY"

        # VIX signal (if available)
        if "VIXCLS" in latest and pd.notna(latest["VIXCLS"]):
            vix = latest["VIXCLS"]
            if vix > 25:
                signals.append("HIGH_VIX")
                regime = "RISK_OFF"
            elif vix < 15:
                signals.append("LOW_VIX")
                if regime == "NEUTRAL":
                    regime = "RISK_ON"

        # Fed funds trend (rising vs falling)
        if "FEDFUNDS" in df.columns:
            ff = df["FEDFUNDS"].dropna()
            if len(ff) >= 60:  # ~3 months
                ff_change = ff.iloc[-1] - ff.iloc[-60]
                if ff_change > 0.5:
                    signals.append("TIGHTENING")
                elif ff_change < -0.5:
                    signals.append("EASING")
                    if regime in ["NEUTRAL", "RISK_ON"]:
                        regime = "EXPANSIONARY"

        return {
            "regime": regime,
            "confidence": len(signals) / 3.0,  # Max 3 signals
            "signals": signals,
            "indicators": {
                "yield_curve_slope": slope,
                "latest_values": latest.to_dict() if hasattr(latest, "to_dict") else {},
            },
            "timestamp": datetime.now().isoformat(),
        }


# Singleton instance
_provider: Optional[FREDMacroProvider] = None


def get_fred_provider() -> FREDMacroProvider:
    """Get or create singleton FRED provider."""
    global _provider
    if _provider is None:
        _provider = FREDMacroProvider()
    return _provider


# Convenience functions
def get_macro_regime() -> Dict[str, Any]:
    """Get current macro regime assessment."""
    return get_fred_provider().get_macro_regime()


def get_yield_curve() -> Dict[str, float]:
    """Get current yield curve snapshot."""
    return get_fred_provider().get_yield_curve()


def get_yield_curve_slope() -> Dict[str, float]:
    """Get yield curve slopes and inversion status."""
    return get_fred_provider().get_yield_curve_slope()


if __name__ == "__main__":
    # Demo usage
    provider = FREDMacroProvider()

    print("=== FRED Macro Provider Demo ===\n")

    # Yield curve
    print("Yield Curve:")
    curve = provider.get_yield_curve()
    for tenor, rate in sorted(curve.items(), key=lambda x: {"3M": 0, "2Y": 1, "5Y": 2, "10Y": 3, "30Y": 4}.get(x[0], 99)):
        print(f"  {tenor}: {rate:.2f}%")

    print("\nYield Curve Slopes:")
    slopes = provider.get_yield_curve_slope()
    print(f"  10Y-2Y Spread: {slopes.get('10Y_2Y_spread', 'N/A')}")
    print(f"  10Y-3M Spread: {slopes.get('10Y_3M_spread', 'N/A')}")
    print(f"  Inverted: {slopes.get('is_inverted', 'N/A')}")

    print("\nMacro Regime:")
    regime = provider.get_macro_regime()
    print(f"  Regime: {regime['regime']}")
    print(f"  Confidence: {regime['confidence']:.1%}")
    print(f"  Signals: {regime['signals']}")

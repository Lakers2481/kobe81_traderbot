"""
CFTC Commitment of Traders (COT) Data Provider

FREE - No API key required
Source: https://www.cftc.gov/MarketReports/CommitmentsofTraders/

Provides:
- Weekly COT reports (Futures Only + Combined)
- Commercial vs Non-commercial positioning
- Net positions and changes
- Extreme positioning indicators (risk-on/off signals)

Published every Friday (data as of Tuesday)

Author: Kobe Trading System
Created: 2026-01-04
"""

import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from io import BytesIO, StringIO

import pandas as pd
import requests

from core.structured_log import get_logger

logger = get_logger(__name__)


# Key futures contracts for market sentiment
COT_CONTRACTS = {
    # Equity Index Futures
    "E-MINI S&P 500": {"category": "equity", "ticker": "ES"},
    "E-MINI NASDAQ-100": {"category": "equity", "ticker": "NQ"},
    "E-MINI DOW ($5)": {"category": "equity", "ticker": "YM"},
    "E-MINI RUSSELL 2000": {"category": "equity", "ticker": "RTY"},
    "VIX FUTURES": {"category": "volatility", "ticker": "VX"},

    # Treasury Futures
    "10-YEAR T-NOTE": {"category": "rates", "ticker": "ZN"},
    "2-YEAR T-NOTE": {"category": "rates", "ticker": "ZT"},
    "5-YEAR T-NOTE": {"category": "rates", "ticker": "ZF"},
    "30-YEAR T-BOND": {"category": "rates", "ticker": "ZB"},
    "ULTRA T-BOND": {"category": "rates", "ticker": "UB"},

    # Currencies
    "EURO FX": {"category": "fx", "ticker": "6E"},
    "JAPANESE YEN": {"category": "fx", "ticker": "6J"},
    "BRITISH POUND": {"category": "fx", "ticker": "6B"},
    "U.S. DOLLAR INDEX": {"category": "fx", "ticker": "DX"},

    # Commodities
    "GOLD": {"category": "metals", "ticker": "GC"},
    "SILVER": {"category": "metals", "ticker": "SI"},
    "COPPER": {"category": "metals", "ticker": "HG"},
    "CRUDE OIL": {"category": "energy", "ticker": "CL"},
    "NATURAL GAS": {"category": "energy", "ticker": "NG"},
}

# Core contracts for sentiment analysis
CORE_CONTRACTS = [
    "E-MINI S&P 500",
    "VIX FUTURES",
    "10-YEAR T-NOTE",
    "GOLD",
    "CRUDE OIL",
    "U.S. DOLLAR INDEX"
]


class CFTCCOTProvider:
    """
    CFTC Commitment of Traders (COT) data provider.

    Features:
    - Download historical COT data (back to 2006)
    - Parse futures-only and combined reports
    - Calculate net positions and extremes
    - Sentiment indicators for trading
    """

    # CFTC COT data URLs
    BASE_URL = "https://www.cftc.gov/files/dea/history"

    # Current year files
    FUTURES_ONLY_URL = "{base}/fut_fin_txt_{year}.zip"  # Financial futures
    COMBINED_URL = "{base}/com_fin_txt_{year}.zip"  # Combined (futures + options)
    LEGACY_URL = "{base}/deacot{year}.zip"  # Legacy format (commodities)

    CACHE_DIR = Path("data/cache/cftc_cot")
    CACHE_TTL_HOURS = 24 * 7  # Weekly data, cache for 7 days

    def __init__(self):
        """Initialize CFTC COT provider."""
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, report_type: str, year: int) -> Path:
        """Get cache file path for a report."""
        return self.CACHE_DIR / f"cot_{report_type}_{year}.parquet"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file exists and is fresh."""
        if not cache_path.exists():
            return False

        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age_hours = (datetime.now() - mtime).total_seconds() / 3600
        return age_hours < self.CACHE_TTL_HOURS

    def download_cot_report(
        self,
        year: int,
        report_type: str = "financial",
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Download COT report for a specific year.

        Args:
            year: Year to fetch (2006-present)
            report_type: 'financial' or 'legacy'
            force_refresh: Bypass cache

        Returns:
            DataFrame with COT data
        """
        cache_path = self._cache_path(report_type, year)

        # Check cache
        if not force_refresh and self._is_cache_valid(cache_path):
            logger.debug(f"COT cache hit: {report_type} {year}")
            return pd.read_parquet(cache_path)

        # Check if we've already failed for this year (avoid repeated 404s)
        global _cot_download_errors
        error_key = f"{year}_{report_type}"
        if error_key in _cot_download_errors:
            # Already failed, try stale cache silently
            if cache_path.exists():
                return pd.read_parquet(cache_path)
            return pd.DataFrame()

        # Build URL
        if report_type == "financial":
            url = self.FUTURES_ONLY_URL.format(base=self.BASE_URL, year=year)
        else:
            url = self.LEGACY_URL.format(base=self.BASE_URL, year=year)

        try:
            logger.info(f"Downloading COT {report_type} {year} from {url}")
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            # Extract ZIP
            with zipfile.ZipFile(BytesIO(response.content)) as zf:
                # Find the txt file in the zip
                txt_files = [f for f in zf.namelist() if f.endswith('.txt')]
                if not txt_files:
                    logger.error(f"No txt file found in COT zip for {year}")
                    return pd.DataFrame()

                with zf.open(txt_files[0]) as f:
                    content = f.read().decode('utf-8', errors='ignore')

            # Parse CSV
            df = pd.read_csv(StringIO(content))

            # Standardize column names
            df.columns = [c.strip() for c in df.columns]

            # Parse date column
            date_col = None
            for col in df.columns:
                if 'date' in col.lower():
                    date_col = col
                    break

            if date_col:
                df['report_date'] = pd.to_datetime(df[date_col], errors='coerce')

            # Cache as parquet
            df.to_parquet(cache_path)
            logger.info(f"COT {report_type} {year}: {len(df)} rows cached")

            return df

        except requests.RequestException as e:
            # Use module-level error tracking to avoid repeated logging
            # (global already declared above)
            error_key = f"{year}_{report_type}"
            if error_key not in _cot_download_errors:
                # For 404 on current year, this is expected early in year
                if '404' in str(e) and year == datetime.now().year:
                    logger.debug(f"COT {year} not yet published (expected early in year)")
                else:
                    logger.warning(f"COT download failed for {year}: {e}")
                _cot_download_errors.add(error_key)

            # Fall back to cache
            if cache_path.exists():
                logger.warning(f"Using stale cache for COT {year}")
                return pd.read_parquet(cache_path)

            return pd.DataFrame()

    def get_cot_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        contracts: Optional[List[str]] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Get COT data for a date range and specific contracts.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            contracts: List of contract names to filter (None = all)
            force_refresh: Bypass cache

        Returns:
            DataFrame with COT positioning data
        """
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365*3)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Download all needed years
        all_data = []
        current_year = datetime.now().year
        for year in range(start_dt.year, end_dt.year + 1):
            df = self.download_cot_report(year, "financial", force_refresh)
            if not df.empty:
                all_data.append(df)
            elif year == current_year:
                # Current year not available yet, skip silently (expected early in year)
                pass

        if not all_data:
            # If no data at all, try previous year as fallback
            fallback_year = current_year - 1
            df = self.download_cot_report(fallback_year, "financial", force_refresh)
            if not df.empty:
                all_data.append(df)
                logger.info(f"Using {fallback_year} COT data (current year not published yet)")

        if not all_data:
            return pd.DataFrame()

        # Combine all years
        combined = pd.concat(all_data, ignore_index=True)

        # Filter by date
        if 'report_date' in combined.columns:
            combined = combined[
                (combined['report_date'] >= start_dt) &
                (combined['report_date'] <= end_dt)
            ]

        # Filter by contracts
        if contracts:
            name_col = None
            for col in combined.columns:
                if 'name' in col.lower() or 'market' in col.lower():
                    name_col = col
                    break

            if name_col:
                pattern = '|'.join(contracts)
                combined = combined[combined[name_col].str.contains(pattern, case=False, na=False)]

        combined.sort_values('report_date', inplace=True)
        return combined

    def get_positioning(
        self,
        contract_name: str,
        lookback_weeks: int = 52
    ) -> Dict[str, Any]:
        """
        Get current positioning for a specific contract.

        Args:
            contract_name: Contract name (e.g., 'E-MINI S&P 500')
            lookback_weeks: Weeks of history for percentile calculations

        Returns:
            Dict with positioning data and extremes
        """
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(weeks=lookback_weeks + 4)).strftime("%Y-%m-%d")

        df = self.get_cot_data(start_date, end_date, [contract_name])

        if df.empty:
            return {"error": f"No data found for {contract_name}"}

        # Find the relevant columns
        cols = df.columns.tolist()

        # Look for commercial/non-commercial positions
        comm_long_col = None
        comm_short_col = None
        noncomm_long_col = None
        noncomm_short_col = None

        for col in cols:
            col_lower = col.lower()
            if 'commercial' in col_lower and 'non' not in col_lower:
                if 'long' in col_lower:
                    comm_long_col = col
                elif 'short' in col_lower:
                    comm_short_col = col
            elif 'noncommercial' in col_lower or 'non-commercial' in col_lower:
                if 'long' in col_lower:
                    noncomm_long_col = col
                elif 'short' in col_lower:
                    noncomm_short_col = col

        # Calculate net positions
        result = {
            "contract": contract_name,
            "report_date": df['report_date'].max().strftime("%Y-%m-%d") if 'report_date' in df.columns else None,
        }

        # Commercial net position (hedgers)
        if comm_long_col and comm_short_col:
            df['comm_net'] = df[comm_long_col] - df[comm_short_col]
            latest_comm = df['comm_net'].iloc[-1]
            result['commercial_net'] = int(latest_comm)
            result['commercial_percentile'] = self._calculate_percentile(df['comm_net'])

        # Non-commercial net position (speculators)
        if noncomm_long_col and noncomm_short_col:
            df['spec_net'] = df[noncomm_long_col] - df[noncomm_short_col]
            latest_spec = df['spec_net'].iloc[-1]
            result['speculator_net'] = int(latest_spec)
            result['speculator_percentile'] = self._calculate_percentile(df['spec_net'])

        # Determine sentiment
        spec_pct = result.get('speculator_percentile', 50)
        if spec_pct >= 80:
            result['sentiment'] = 'EXTREMELY_BULLISH'
            result['contrarian_signal'] = 'BEARISH'  # Crowded long
        elif spec_pct >= 60:
            result['sentiment'] = 'BULLISH'
            result['contrarian_signal'] = 'NEUTRAL'
        elif spec_pct <= 20:
            result['sentiment'] = 'EXTREMELY_BEARISH'
            result['contrarian_signal'] = 'BULLISH'  # Crowded short
        elif spec_pct <= 40:
            result['sentiment'] = 'BEARISH'
            result['contrarian_signal'] = 'NEUTRAL'
        else:
            result['sentiment'] = 'NEUTRAL'
            result['contrarian_signal'] = 'NEUTRAL'

        return result

    def _calculate_percentile(self, series: pd.Series) -> float:
        """Calculate percentile of latest value in historical distribution."""
        latest = series.iloc[-1]
        return float((series < latest).sum() / len(series) * 100)

    def get_market_sentiment(self) -> Dict[str, Any]:
        """
        Get overall market sentiment from key COT indicators.

        Returns:
            Dict with aggregated sentiment analysis
        """
        sentiments = {}
        signals = []

        for contract in CORE_CONTRACTS:
            pos = self.get_positioning(contract)
            if 'error' not in pos:
                sentiments[contract] = pos

                # Collect extreme signals
                if pos.get('contrarian_signal') == 'BULLISH':
                    signals.append(f"{contract}: OVERSOLD (contrarian buy)")
                elif pos.get('contrarian_signal') == 'BEARISH':
                    signals.append(f"{contract}: OVERBOUGHT (contrarian sell)")

        # Aggregate equity sentiment
        equity_sentiment = 'NEUTRAL'
        if 'E-MINI S&P 500' in sentiments:
            equity_sentiment = sentiments['E-MINI S&P 500'].get('sentiment', 'NEUTRAL')

        # VIX positioning (inverse logic)
        vix_sentiment = 'NEUTRAL'
        if 'VIX FUTURES' in sentiments:
            vix_spec = sentiments['VIX FUTURES'].get('speculator_percentile', 50)
            if vix_spec >= 80:  # High VIX longs = fear
                vix_sentiment = 'FEARFUL'
            elif vix_spec <= 20:  # Low VIX longs = complacent
                vix_sentiment = 'COMPLACENT'

        # Dollar sentiment
        dollar_sentiment = 'NEUTRAL'
        if 'U.S. DOLLAR INDEX' in sentiments:
            dollar_sentiment = sentiments['U.S. DOLLAR INDEX'].get('sentiment', 'NEUTRAL')

        return {
            'timestamp': datetime.now().isoformat(),
            'equity_sentiment': equity_sentiment,
            'vix_sentiment': vix_sentiment,
            'dollar_sentiment': dollar_sentiment,
            'extreme_signals': signals,
            'positions': sentiments,
            'overall': self._aggregate_sentiment([
                equity_sentiment, vix_sentiment, dollar_sentiment
            ])
        }

    def _aggregate_sentiment(self, sentiments: List[str]) -> str:
        """Aggregate multiple sentiment readings."""
        bullish = sum(1 for s in sentiments if 'BULLISH' in s)
        bearish = sum(1 for s in sentiments if 'BEARISH' in s)

        if bullish >= 2:
            return 'RISK_ON'
        elif bearish >= 2:
            return 'RISK_OFF'
        else:
            return 'NEUTRAL'


# Singleton instance
_provider: Optional[CFTCCOTProvider] = None

# Module-level error tracking to avoid repeated failed downloads
_cot_download_errors: set = set()


def get_cot_provider() -> CFTCCOTProvider:
    """Get or create singleton COT provider."""
    global _provider
    if _provider is None:
        _provider = CFTCCOTProvider()
    return _provider


# Convenience functions
def get_market_sentiment() -> Dict[str, Any]:
    """Get COT-based market sentiment analysis."""
    return get_cot_provider().get_market_sentiment()


def get_positioning(contract: str) -> Dict[str, Any]:
    """Get positioning for a specific contract."""
    return get_cot_provider().get_positioning(contract)


if __name__ == "__main__":
    # Demo usage
    provider = CFTCCOTProvider()

    print("=== CFTC COT Provider Demo ===\n")

    print("Fetching S&P 500 E-mini positioning...")
    pos = provider.get_positioning("E-MINI S&P 500")

    if 'error' not in pos:
        print("\nS&P 500 E-mini Futures:")
        print(f"  Report Date: {pos.get('report_date', 'N/A')}")
        print(f"  Commercial Net: {pos.get('commercial_net', 'N/A'):,}")
        print(f"  Commercial Percentile: {pos.get('commercial_percentile', 'N/A'):.1f}%")
        print(f"  Speculator Net: {pos.get('speculator_net', 'N/A'):,}")
        print(f"  Speculator Percentile: {pos.get('speculator_percentile', 'N/A'):.1f}%")
        print(f"  Sentiment: {pos.get('sentiment', 'N/A')}")
        print(f"  Contrarian Signal: {pos.get('contrarian_signal', 'N/A')}")
    else:
        print(f"  Error: {pos['error']}")

    print("\n\nOverall Market Sentiment:")
    sentiment = provider.get_market_sentiment()
    print(f"  Equity: {sentiment.get('equity_sentiment', 'N/A')}")
    print(f"  VIX: {sentiment.get('vix_sentiment', 'N/A')}")
    print(f"  Dollar: {sentiment.get('dollar_sentiment', 'N/A')}")
    print(f"  Overall: {sentiment.get('overall', 'N/A')}")

    if sentiment.get('extreme_signals'):
        print("\n  Extreme Signals:")
        for sig in sentiment['extreme_signals']:
            print(f"    - {sig}")

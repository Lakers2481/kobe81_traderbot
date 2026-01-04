"""
BEA (Bureau of Economic Analysis) Macro Data Provider

FREE API - Requires free registration at https://www.bea.gov/resources/for-developers

Provides:
- GDP (Gross Domestic Product)
- PCE (Personal Consumption Expenditures)
- Personal Income
- Corporate Profits
- Trade Balance

Author: Kobe Trading System
Created: 2026-01-04
"""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
import requests

from core.structured_log import get_logger

logger = get_logger(__name__)


# Key BEA datasets and tables
BEA_DATASETS = {
    'NIPA': {
        'GDP': {'TableName': 'T10101', 'Frequency': 'Q'},  # GDP components
        'PCE': {'TableName': 'T20801', 'Frequency': 'M'},  # PCE
        'PERSONAL_INCOME': {'TableName': 'T20100', 'Frequency': 'M'},
        'CORPORATE_PROFITS': {'TableName': 'T16500', 'Frequency': 'Q'},
    },
    'ITA': {
        'TRADE_BALANCE': {'Indicator': 'Balance', 'AreaOrCountry': 'AllCountries'},
    }
}


class BEAMacroProvider:
    """
    BEA API client for macro economic data.

    Features:
    - GDP and components
    - Personal consumption expenditures
    - Personal income data
    - 24-hour caching
    """

    BASE_URL = "https://apps.bea.gov/api/data"
    CACHE_DIR = Path("data/cache/bea")
    CACHE_TTL_HOURS = 24

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize BEA provider.

        Args:
            api_key: BEA API key. If not provided, uses env var BEA_API_KEY
        """
        self.api_key = api_key or os.getenv("BEA_API_KEY")
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if not self.api_key:
            logger.warning("No BEA API key - get free key at https://www.bea.gov/resources/for-developers")

    def _cache_path(self, dataset: str, table: str) -> Path:
        """Get cache file path."""
        return self.CACHE_DIR / f"bea_{dataset}_{table}.json"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache is fresh."""
        if not cache_path.exists():
            return False
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return (datetime.now() - mtime).total_seconds() / 3600 < self.CACHE_TTL_HOURS

    def _load_cache(self, cache_path: Path) -> Optional[Dict]:
        """Load from cache."""
        try:
            with open(cache_path) as f:
                return json.load(f)
        except Exception:
            return None

    def _save_cache(self, cache_path: Path, data: Dict) -> None:
        """Save to cache."""
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")

    def get_nipa_data(
        self,
        table_name: str,
        frequency: str = 'Q',
        years: Optional[List[int]] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch NIPA (National Income and Product Accounts) data.

        Args:
            table_name: BEA table name (e.g., 'T10101' for GDP)
            frequency: 'Q' for quarterly, 'M' for monthly
            years: List of years to fetch (defaults to last 5 years)
            force_refresh: Bypass cache

        Returns:
            DataFrame with economic data
        """
        if not self.api_key:
            logger.error("BEA API key required")
            return pd.DataFrame()

        cache_path = self._cache_path('NIPA', table_name)

        if not force_refresh and self._is_cache_valid(cache_path):
            cached = self._load_cache(cache_path)
            if cached and 'data' in cached:
                return pd.DataFrame(cached['data'])

        if years is None:
            current_year = datetime.now().year
            years = list(range(current_year - 5, current_year + 1))

        year_str = ','.join(str(y) for y in years)

        params = {
            'UserID': self.api_key,
            'method': 'GetData',
            'datasetname': 'NIPA',
            'TableName': table_name,
            'Frequency': frequency,
            'Year': year_str,
            'ResultFormat': 'JSON'
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if 'BEAAPI' in data and 'Results' in data['BEAAPI']:
                results = data['BEAAPI']['Results']
                if 'Data' in results:
                    df = pd.DataFrame(results['Data'])
                    self._save_cache(cache_path, {'data': results['Data']})
                    logger.info(f"BEA {table_name}: {len(df)} rows fetched")
                    return df

            logger.warning(f"BEA API returned no data for {table_name}")
            return pd.DataFrame()

        except requests.RequestException as e:
            logger.error(f"BEA API error: {e}")
            cached = self._load_cache(cache_path)
            if cached and 'data' in cached:
                logger.warning("Using stale cache")
                return pd.DataFrame(cached['data'])
            return pd.DataFrame()

    def get_gdp(self, years: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Get GDP data and growth rate.

        Returns:
            Dict with GDP values and growth rate
        """
        df = self.get_nipa_data('T10101', frequency='Q', years=years)

        if df.empty:
            return {'error': 'No GDP data available'}

        # Filter for Real GDP line
        gdp_df = df[df['SeriesCode'] == 'A191RL1']

        if gdp_df.empty:
            # Try alternative
            gdp_df = df[df['LineDescription'].str.contains('Gross domestic product', case=False, na=False)]

        if gdp_df.empty:
            return {'error': 'GDP line not found in data'}

        # Sort by time period
        gdp_df = gdp_df.sort_values('TimePeriod')

        latest = gdp_df.iloc[-1] if len(gdp_df) > 0 else None

        if latest is None:
            return {'error': 'No GDP data'}

        result = {
            'period': latest.get('TimePeriod', 'Unknown'),
            'value': float(latest.get('DataValue', 0)),
            'description': latest.get('LineDescription', 'GDP'),
            'unit': 'Billions of chained 2017 dollars',
        }

        # Calculate QoQ growth
        if len(gdp_df) >= 2:
            prev = float(gdp_df.iloc[-2]['DataValue'])
            curr = float(latest['DataValue'])
            if prev > 0:
                result['qoq_growth'] = (curr - prev) / prev * 100

        # YoY growth
        if len(gdp_df) >= 5:
            yoy_prev = float(gdp_df.iloc[-5]['DataValue'])
            curr = float(latest['DataValue'])
            if yoy_prev > 0:
                result['yoy_growth'] = (curr - yoy_prev) / yoy_prev * 100

        return result

    def get_pce(self) -> Dict[str, Any]:
        """
        Get Personal Consumption Expenditures data.

        Returns:
            Dict with PCE values
        """
        df = self.get_nipa_data('T20801', frequency='M')

        if df.empty:
            return {'error': 'No PCE data available'}

        # Get total PCE
        pce_df = df[df['LineDescription'].str.contains('Personal consumption', case=False, na=False)]

        if pce_df.empty:
            return {'error': 'PCE line not found'}

        pce_df = pce_df.sort_values('TimePeriod')
        latest = pce_df.iloc[-1] if len(pce_df) > 0 else None

        if latest is None:
            return {'error': 'No PCE data'}

        return {
            'period': latest.get('TimePeriod', 'Unknown'),
            'value': float(latest.get('DataValue', 0)),
            'description': latest.get('LineDescription', 'PCE'),
        }

    def get_macro_summary(self) -> Dict[str, Any]:
        """
        Get summary of key BEA macro indicators.

        Returns:
            Dict with all key indicators
        """
        return {
            'gdp': self.get_gdp(),
            'pce': self.get_pce(),
            'timestamp': datetime.now().isoformat(),
        }


# Singleton
_provider: Optional[BEAMacroProvider] = None


def get_bea_provider() -> BEAMacroProvider:
    """Get or create singleton BEA provider."""
    global _provider
    if _provider is None:
        _provider = BEAMacroProvider()
    return _provider


def get_gdp() -> Dict[str, Any]:
    """Get current GDP data."""
    return get_bea_provider().get_gdp()


def get_pce() -> Dict[str, Any]:
    """Get current PCE data."""
    return get_bea_provider().get_pce()


if __name__ == "__main__":
    # Demo
    print("=== BEA Macro Provider Demo ===\n")

    provider = BEAMacroProvider()

    print("GDP Data:")
    gdp = provider.get_gdp()
    if 'error' not in gdp:
        print(f"  Period: {gdp.get('period')}")
        print(f"  Value: ${gdp.get('value', 0):,.0f}B")
        if 'qoq_growth' in gdp:
            print(f"  QoQ Growth: {gdp['qoq_growth']:.1f}%")
        if 'yoy_growth' in gdp:
            print(f"  YoY Growth: {gdp['yoy_growth']:.1f}%")
    else:
        print(f"  {gdp['error']}")

    print("\nPCE Data:")
    pce = provider.get_pce()
    if 'error' not in pce:
        print(f"  Period: {pce.get('period')}")
        print(f"  Value: ${pce.get('value', 0):,.0f}")
    else:
        print(f"  {pce['error']}")

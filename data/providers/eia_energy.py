"""
EIA (Energy Information Administration) Energy Data Provider

FREE API - Requires free registration at https://www.eia.gov/opendata/

Provides:
- Crude Oil prices (WTI, Brent)
- Natural Gas prices
- Gasoline prices
- Petroleum inventories
- Energy production data

Author: Kobe Trading System
Created: 2026-01-04
"""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import requests

from core.structured_log import get_logger

logger = get_logger(__name__)


# Key EIA series IDs
EIA_SERIES = {
    # Crude Oil
    'WTI_SPOT': 'PET.RWTC.D',  # WTI Cushing spot price, daily
    'BRENT_SPOT': 'PET.RBRTE.D',  # Brent spot price, daily
    'WTI_FUTURES': 'PET.RCLC1.D',  # WTI futures front month

    # Natural Gas
    'HENRY_HUB': 'NG.RNGWHHD.D',  # Henry Hub natural gas spot
    'NG_FUTURES': 'NG.RNGC1.D',  # Natural gas futures

    # Refined Products
    'GASOLINE_REGULAR': 'PET.EMM_EPM0_PTE_NUS_DPG.W',  # Regular gasoline price
    'DIESEL': 'PET.EMD_EPD2D_PTE_NUS_DPG.W',  # Diesel price
    'HEATING_OIL': 'PET.EER_EPLLPA_PF4_Y35NY_DPG.D',  # Heating oil

    # Inventories
    'CRUDE_INVENTORY': 'PET.WCESTUS1.W',  # Crude oil stocks
    'GASOLINE_INVENTORY': 'PET.WGTSTUS1.W',  # Gasoline stocks
    'DISTILLATE_INVENTORY': 'PET.WDISTUS1.W',  # Distillate stocks

    # Production
    'US_CRUDE_PRODUCTION': 'PET.WCRFPUS2.W',  # US crude production
}


class EIAEnergyProvider:
    """
    EIA API client for energy data.

    Features:
    - Crude oil and natural gas prices
    - Petroleum inventories
    - Energy production data
    - 24-hour caching
    """

    BASE_URL = "https://api.eia.gov/v2"
    CACHE_DIR = Path("data/cache/eia")
    CACHE_TTL_HOURS = 24

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize EIA provider.

        Args:
            api_key: EIA API key. If not provided, uses env var EIA_API_KEY
        """
        self.api_key = api_key or os.getenv("EIA_API_KEY")
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if not self.api_key:
            logger.warning("No EIA API key - get free key at https://www.eia.gov/opendata/")

    def _cache_path(self, series_id: str) -> Path:
        """Get cache file path."""
        safe_id = series_id.replace('.', '_').replace('/', '_')
        return self.CACHE_DIR / f"eia_{safe_id}.json"

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

    def get_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch EIA data series.

        Args:
            series_id: EIA series ID (e.g., 'PET.RWTC.D')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            force_refresh: Bypass cache

        Returns:
            DataFrame with date and value columns
        """
        if not self.api_key:
            logger.error("EIA API key required")
            return pd.DataFrame()

        cache_path = self._cache_path(series_id)

        if not force_refresh and self._is_cache_valid(cache_path):
            cached = self._load_cache(cache_path)
            if cached and 'data' in cached:
                df = pd.DataFrame(cached['data'])
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                return df

        # Parse series ID to determine route
        parts = series_id.split('.')
        if len(parts) >= 2:
            facet = parts[0]
            product = parts[1] if len(parts) > 1 else ''

        # Build URL for v2 API
        url = f"{self.BASE_URL}/seriesid/{series_id}"

        params = {
            'api_key': self.api_key,
            'out': 'json',
        }

        if start_date:
            params['start'] = start_date
        else:
            # Default to 2 years of data
            params['start'] = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

        if end_date:
            params['end'] = end_date

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if 'response' in data and 'data' in data['response']:
                records = data['response']['data']
                df = pd.DataFrame(records)

                # Standardize columns
                if 'period' in df.columns:
                    df['date'] = pd.to_datetime(df['period'])
                if 'value' not in df.columns and 'price' in df.columns:
                    df['value'] = df['price']

                df = df.sort_values('date', ascending=False)

                # Cache
                cache_data = df.to_dict('records')
                self._save_cache(cache_path, {'data': cache_data})

                logger.info(f"EIA {series_id}: {len(df)} rows fetched")
                return df

            logger.warning(f"EIA API returned no data for {series_id}")
            return pd.DataFrame()

        except requests.RequestException as e:
            logger.error(f"EIA API error: {e}")
            cached = self._load_cache(cache_path)
            if cached and 'data' in cached:
                logger.warning("Using stale cache")
                df = pd.DataFrame(cached['data'])
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                return df
            return pd.DataFrame()

    def get_crude_prices(self) -> Dict[str, Any]:
        """
        Get current crude oil prices.

        Returns:
            Dict with WTI and Brent prices
        """
        result = {'timestamp': datetime.now().isoformat()}

        # WTI
        wti_df = self.get_series(EIA_SERIES['WTI_SPOT'])
        if not wti_df.empty and 'value' in wti_df.columns:
            latest = wti_df.iloc[0]
            result['wti'] = {
                'price': float(latest['value']) if pd.notna(latest['value']) else None,
                'date': latest.get('date', latest.get('period', '')),
            }

        # Brent
        brent_df = self.get_series(EIA_SERIES['BRENT_SPOT'])
        if not brent_df.empty and 'value' in brent_df.columns:
            latest = brent_df.iloc[0]
            result['brent'] = {
                'price': float(latest['value']) if pd.notna(latest['value']) else None,
                'date': latest.get('date', latest.get('period', '')),
            }

        # Spread
        if 'wti' in result and 'brent' in result:
            if result['wti'].get('price') and result['brent'].get('price'):
                result['brent_wti_spread'] = result['brent']['price'] - result['wti']['price']

        return result

    def get_natural_gas_prices(self) -> Dict[str, Any]:
        """
        Get current natural gas prices.

        Returns:
            Dict with Henry Hub price
        """
        result = {'timestamp': datetime.now().isoformat()}

        ng_df = self.get_series(EIA_SERIES['HENRY_HUB'])
        if not ng_df.empty and 'value' in ng_df.columns:
            latest = ng_df.iloc[0]
            result['henry_hub'] = {
                'price': float(latest['value']) if pd.notna(latest['value']) else None,
                'date': latest.get('date', latest.get('period', '')),
                'unit': '$/MMBtu',
            }

        return result

    def get_inventories(self) -> Dict[str, Any]:
        """
        Get current petroleum inventories.

        Returns:
            Dict with inventory levels
        """
        result = {'timestamp': datetime.now().isoformat()}

        # Crude
        crude_df = self.get_series(EIA_SERIES['CRUDE_INVENTORY'])
        if not crude_df.empty and 'value' in crude_df.columns:
            latest = crude_df.iloc[0]
            result['crude_stocks'] = {
                'value': float(latest['value']) if pd.notna(latest['value']) else None,
                'date': latest.get('date', latest.get('period', '')),
                'unit': 'thousand barrels',
            }

            # Calculate week-over-week change
            if len(crude_df) >= 2:
                prev = float(crude_df.iloc[1]['value']) if pd.notna(crude_df.iloc[1]['value']) else 0
                curr = float(latest['value']) if pd.notna(latest['value']) else 0
                result['crude_stocks']['wow_change'] = curr - prev

        # Gasoline
        gas_df = self.get_series(EIA_SERIES['GASOLINE_INVENTORY'])
        if not gas_df.empty and 'value' in gas_df.columns:
            latest = gas_df.iloc[0]
            result['gasoline_stocks'] = {
                'value': float(latest['value']) if pd.notna(latest['value']) else None,
                'date': latest.get('date', latest.get('period', '')),
                'unit': 'thousand barrels',
            }

        return result

    def get_energy_summary(self) -> Dict[str, Any]:
        """
        Get summary of all key energy indicators.

        Returns:
            Dict with prices and inventories
        """
        return {
            'crude': self.get_crude_prices(),
            'natural_gas': self.get_natural_gas_prices(),
            'inventories': self.get_inventories(),
            'timestamp': datetime.now().isoformat(),
        }

    def get_energy_regime(self) -> Dict[str, Any]:
        """
        Determine energy market regime.

        Returns:
            Dict with regime classification
        """
        crude = self.get_crude_prices()
        ng = self.get_natural_gas_prices()

        regime = 'NEUTRAL'
        signals = []

        # WTI regime
        wti_price = crude.get('wti', {}).get('price')
        if wti_price:
            if wti_price > 90:
                signals.append('HIGH_OIL')
                regime = 'INFLATIONARY'
            elif wti_price < 50:
                signals.append('LOW_OIL')
                regime = 'DEFLATIONARY'

        # Brent-WTI spread
        spread = crude.get('brent_wti_spread')
        if spread:
            if abs(spread) > 10:
                signals.append('WIDE_SPREAD')

        # Natural gas
        ng_price = ng.get('henry_hub', {}).get('price')
        if ng_price:
            if ng_price > 6:
                signals.append('HIGH_NATGAS')
            elif ng_price < 2:
                signals.append('LOW_NATGAS')

        return {
            'regime': regime,
            'signals': signals,
            'wti_price': wti_price,
            'brent_price': crude.get('brent', {}).get('price'),
            'ng_price': ng_price,
            'timestamp': datetime.now().isoformat(),
        }


# Singleton
_provider: Optional[EIAEnergyProvider] = None


def get_eia_provider() -> EIAEnergyProvider:
    """Get or create singleton EIA provider."""
    global _provider
    if _provider is None:
        _provider = EIAEnergyProvider()
    return _provider


def get_crude_prices() -> Dict[str, Any]:
    """Get current crude oil prices."""
    return get_eia_provider().get_crude_prices()


def get_natural_gas_prices() -> Dict[str, Any]:
    """Get current natural gas prices."""
    return get_eia_provider().get_natural_gas_prices()


def get_energy_regime() -> Dict[str, Any]:
    """Get energy market regime."""
    return get_eia_provider().get_energy_regime()


if __name__ == "__main__":
    # Demo
    print("=== EIA Energy Provider Demo ===\n")

    provider = EIAEnergyProvider()

    print("Crude Oil Prices:")
    crude = provider.get_crude_prices()
    if 'wti' in crude:
        print(f"  WTI: ${crude['wti'].get('price', 'N/A')}")
    if 'brent' in crude:
        print(f"  Brent: ${crude['brent'].get('price', 'N/A')}")
    if 'brent_wti_spread' in crude:
        print(f"  Spread: ${crude['brent_wti_spread']:.2f}")

    print("\nNatural Gas:")
    ng = provider.get_natural_gas_prices()
    if 'henry_hub' in ng:
        print(f"  Henry Hub: ${ng['henry_hub'].get('price', 'N/A')}/MMBtu")

    print("\nInventories:")
    inv = provider.get_inventories()
    if 'crude_stocks' in inv:
        print(f"  Crude Stocks: {inv['crude_stocks'].get('value', 'N/A'):,.0f}k bbl")
        if 'wow_change' in inv['crude_stocks']:
            print(f"  WoW Change: {inv['crude_stocks']['wow_change']:+,.0f}k bbl")

    print("\nEnergy Regime:")
    regime = provider.get_energy_regime()
    print(f"  Regime: {regime.get('regime', 'UNKNOWN')}")
    if regime.get('signals'):
        print(f"  Signals: {', '.join(regime['signals'])}")

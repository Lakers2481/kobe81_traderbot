"""
Macro Feature Engineering Module

Transforms raw macro data into ML-ready features:
- Yield curve slopes and inversions
- Rate change momentum
- Inflation trends
- COT positioning z-scores
- Cross-asset correlations
- Regime indicators

Author: Kobe Trading System
Created: 2026-01-04
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from core.structured_log import get_logger

# Import macro data providers (lazy import to avoid circular deps)
logger = get_logger(__name__)


class MacroFeatureGenerator:
    """
    Generates ML features from macro economic data.

    Features grouped by category:
    1. Yield Curve Features (8 features)
    2. Rate Momentum Features (6 features)
    3. Inflation Features (4 features)
    4. COT Positioning Features (6 features)
    5. Regime Indicators (4 features)
    6. Cross-Asset Features (4 features)

    Total: 32 macro features
    """

    def __init__(self, lookback_days: int = 252):
        """
        Initialize feature generator.

        Args:
            lookback_days: Historical lookback for calculations
        """
        self.lookback_days = lookback_days
        self._cache: Dict[str, Any] = {}
        self._cache_time: Optional[datetime] = None
        self._cache_ttl_minutes = 60

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if self._cache_time is None:
            return False
        age = (datetime.now() - self._cache_time).total_seconds() / 60
        return age < self._cache_ttl_minutes

    def _get_fred_provider(self):
        """Lazy import FRED provider."""
        from data.providers.fred_macro import get_fred_provider
        return get_fred_provider()

    def _get_treasury_provider(self):
        """Lazy import Treasury provider."""
        from data.providers.treasury_yields import get_treasury_provider
        return get_treasury_provider()

    def _get_cot_provider(self):
        """Lazy import COT provider."""
        from data.providers.cftc_cot import get_cot_provider
        return get_cot_provider()

    def get_yield_curve_features(self) -> Dict[str, float]:
        """
        Calculate yield curve features.

        Returns:
            Dict with yield curve feature values
        """
        features = {}

        try:
            treasury = self._get_treasury_provider()
            curve = treasury.get_latest_curve()
            spreads = treasury.calculate_spreads(curve)
            inversions = treasury.detect_inversions(curve)

            # Spread features
            features['yc_10y_2y_spread'] = spreads.get('10Y_2Y', 0.0)
            features['yc_10y_3m_spread'] = spreads.get('10Y_3M', 0.0)
            features['yc_30y_10y_spread'] = spreads.get('30Y_10Y', 0.0)
            features['yc_2y_3m_spread'] = spreads.get('2Y_3M', 0.0)

            # Inversion features (binary)
            features['yc_is_inverted'] = 1.0 if inversions.get('is_classic_inversion') else 0.0
            features['yc_inversion_count'] = float(len(inversions.get('inversions', [])))

            # Curve shape encoding
            shape_map = {'NORMAL': 0, 'FLAT': 1, 'INVERTED': 2, 'HUMPED': 3, 'STEEP': 4}
            features['yc_shape'] = float(shape_map.get(inversions.get('curve_shape', 'UNKNOWN'), -1))

            # Level features
            if curve:
                features['yc_level_short'] = curve.get('3M', curve.get('1M', 0.0))
                features['yc_level_long'] = curve.get('10Y', curve.get('30Y', 0.0))

        except Exception as e:
            logger.warning(f"Yield curve features failed: {e}")
            # Return zeros for all features
            for key in ['yc_10y_2y_spread', 'yc_10y_3m_spread', 'yc_30y_10y_spread',
                        'yc_2y_3m_spread', 'yc_is_inverted', 'yc_inversion_count',
                        'yc_shape', 'yc_level_short', 'yc_level_long']:
                features[key] = 0.0

        return features

    def get_rate_momentum_features(self) -> Dict[str, float]:
        """
        Calculate interest rate momentum features.

        Returns:
            Dict with rate momentum feature values
        """
        features = {}

        try:
            fred = self._get_fred_provider()

            # Get Fed Funds rate history
            ff_df = fred.get_series('FEDFUNDS')
            if not ff_df.empty:
                ff = ff_df.set_index('date')['value'].dropna()

                # Rate changes over different periods
                if len(ff) >= 5:
                    features['rate_change_1w'] = float(ff.iloc[-1] - ff.iloc[-5]) if len(ff) >= 5 else 0.0
                if len(ff) >= 21:
                    features['rate_change_1m'] = float(ff.iloc[-1] - ff.iloc[-21])
                if len(ff) >= 63:
                    features['rate_change_3m'] = float(ff.iloc[-1] - ff.iloc[-63])
                if len(ff) >= 252:
                    features['rate_change_1y'] = float(ff.iloc[-1] - ff.iloc[-252])

                # Rate level
                features['rate_level'] = float(ff.iloc[-1]) if len(ff) > 0 else 0.0

                # Rate percentile (where are we in historical distribution)
                if len(ff) >= 252:
                    features['rate_percentile'] = float(stats.percentileofscore(ff.tail(252), ff.iloc[-1]) / 100)

            # Get 10Y rate momentum
            t10_df = fred.get_series('DGS10')
            if not t10_df.empty:
                t10 = t10_df.set_index('date')['value'].dropna()
                if len(t10) >= 21:
                    features['t10y_momentum_1m'] = float(t10.iloc[-1] - t10.iloc[-21])

        except Exception as e:
            logger.warning(f"Rate momentum features failed: {e}")

        # Fill missing with zeros
        for key in ['rate_change_1w', 'rate_change_1m', 'rate_change_3m', 'rate_change_1y',
                    'rate_level', 'rate_percentile', 't10y_momentum_1m']:
            if key not in features:
                features[key] = 0.0

        return features

    def get_inflation_features(self) -> Dict[str, float]:
        """
        Calculate inflation-related features.

        Returns:
            Dict with inflation feature values
        """
        features = {}

        try:
            fred = self._get_fred_provider()

            # CPI (annual rate)
            cpi_df = fred.get_series('CPIAUCSL')
            if not cpi_df.empty:
                cpi = cpi_df.set_index('date')['value'].dropna()
                if len(cpi) >= 12:
                    # YoY inflation rate
                    yoy_inflation = (cpi.iloc[-1] / cpi.iloc[-12] - 1) * 100
                    features['inflation_yoy'] = float(yoy_inflation)

                    # Inflation momentum (is it accelerating?)
                    if len(cpi) >= 24:
                        prev_yoy = (cpi.iloc[-12] / cpi.iloc[-24] - 1) * 100
                        features['inflation_acceleration'] = float(yoy_inflation - prev_yoy)

            # Breakeven inflation (market expectations)
            t5yie_df = fred.get_series('T5YIE')
            if not t5yie_df.empty:
                t5yie = t5yie_df.set_index('date')['value'].dropna()
                if len(t5yie) > 0:
                    features['inflation_breakeven_5y'] = float(t5yie.iloc[-1])

            t10yie_df = fred.get_series('T10YIE')
            if not t10yie_df.empty:
                t10yie = t10yie_df.set_index('date')['value'].dropna()
                if len(t10yie) > 0:
                    features['inflation_breakeven_10y'] = float(t10yie.iloc[-1])

            # Real rate (10Y nominal - 10Y breakeven)
            if 'inflation_breakeven_10y' in features:
                t10_df = fred.get_series('DGS10')
                if not t10_df.empty:
                    t10 = t10_df.set_index('date')['value'].dropna()
                    if len(t10) > 0:
                        features['real_rate_10y'] = float(t10.iloc[-1] - features['inflation_breakeven_10y'])

        except Exception as e:
            logger.warning(f"Inflation features failed: {e}")

        # Fill missing
        for key in ['inflation_yoy', 'inflation_acceleration', 'inflation_breakeven_5y',
                    'inflation_breakeven_10y', 'real_rate_10y']:
            if key not in features:
                features[key] = 0.0

        return features

    def get_cot_features(self) -> Dict[str, float]:
        """
        Calculate COT positioning features.

        Returns:
            Dict with COT positioning feature values
        """
        features = {}

        try:
            cot = self._get_cot_provider()
            sentiment = cot.get_market_sentiment()

            # Equity positioning
            if 'E-MINI S&P 500' in sentiment.get('positions', {}):
                es_pos = sentiment['positions']['E-MINI S&P 500']
                features['cot_es_spec_pct'] = float(es_pos.get('speculator_percentile', 50)) / 100
                features['cot_es_comm_pct'] = float(es_pos.get('commercial_percentile', 50)) / 100

            # VIX positioning (inverse sentiment)
            if 'VIX FUTURES' in sentiment.get('positions', {}):
                vix_pos = sentiment['positions']['VIX FUTURES']
                features['cot_vix_spec_pct'] = float(vix_pos.get('speculator_percentile', 50)) / 100

            # Gold positioning (risk appetite indicator)
            if 'GOLD' in sentiment.get('positions', {}):
                gold_pos = sentiment['positions']['GOLD']
                features['cot_gold_spec_pct'] = float(gold_pos.get('speculator_percentile', 50)) / 100

            # Dollar positioning
            if 'U.S. DOLLAR INDEX' in sentiment.get('positions', {}):
                dxy_pos = sentiment['positions']['U.S. DOLLAR INDEX']
                features['cot_dxy_spec_pct'] = float(dxy_pos.get('speculator_percentile', 50)) / 100

            # Overall sentiment encoding
            overall_map = {'RISK_ON': 1.0, 'NEUTRAL': 0.0, 'RISK_OFF': -1.0}
            features['cot_overall_sentiment'] = overall_map.get(sentiment.get('overall', 'NEUTRAL'), 0.0)

            # Extreme count (contrarian signal)
            features['cot_extreme_count'] = float(len(sentiment.get('extreme_signals', [])))

        except Exception as e:
            logger.warning(f"COT features failed: {e}")

        # Fill missing
        for key in ['cot_es_spec_pct', 'cot_es_comm_pct', 'cot_vix_spec_pct',
                    'cot_gold_spec_pct', 'cot_dxy_spec_pct', 'cot_overall_sentiment',
                    'cot_extreme_count']:
            if key not in features:
                features[key] = 0.5 if 'pct' in key else 0.0

        return features

    def get_regime_features(self) -> Dict[str, float]:
        """
        Calculate macro regime indicator features.

        Returns:
            Dict with regime feature values
        """
        features = {}

        try:
            fred = self._get_fred_provider()
            regime = fred.get_macro_regime()

            # Regime encoding
            regime_map = {
                'EXPANSIONARY': 2.0,
                'RISK_ON': 1.0,
                'NEUTRAL': 0.0,
                'RISK_OFF': -1.0,
                'CONTRACTIONARY': -2.0,
                'UNKNOWN': 0.0
            }
            features['macro_regime'] = regime_map.get(regime.get('regime', 'UNKNOWN'), 0.0)
            features['macro_regime_confidence'] = float(regime.get('confidence', 0.0))

            # Signal flags (binary)
            signals = regime.get('signals', [])
            features['regime_inverted_curve'] = 1.0 if 'INVERTED_CURVE' in signals else 0.0
            features['regime_high_vix'] = 1.0 if 'HIGH_VIX' in signals else 0.0
            features['regime_tightening'] = 1.0 if 'TIGHTENING' in signals else 0.0
            features['regime_easing'] = 1.0 if 'EASING' in signals else 0.0

        except Exception as e:
            logger.warning(f"Regime features failed: {e}")

        # Fill missing
        for key in ['macro_regime', 'macro_regime_confidence', 'regime_inverted_curve',
                    'regime_high_vix', 'regime_tightening', 'regime_easing']:
            if key not in features:
                features[key] = 0.0

        return features

    def get_vix_features(self) -> Dict[str, float]:
        """
        Calculate VIX-related features.

        Returns:
            Dict with VIX feature values
        """
        features = {}

        try:
            fred = self._get_fred_provider()
            vix_df = fred.get_series('VIXCLS')

            if not vix_df.empty:
                vix = vix_df.set_index('date')['value'].dropna()

                if len(vix) > 0:
                    features['vix_level'] = float(vix.iloc[-1])

                    # VIX regime (based on level)
                    if vix.iloc[-1] > 30:
                        features['vix_regime'] = 2.0  # High fear
                    elif vix.iloc[-1] > 20:
                        features['vix_regime'] = 1.0  # Elevated
                    elif vix.iloc[-1] < 12:
                        features['vix_regime'] = -1.0  # Complacent
                    else:
                        features['vix_regime'] = 0.0  # Normal

                    # VIX percentile
                    if len(vix) >= 252:
                        features['vix_percentile'] = float(
                            stats.percentileofscore(vix.tail(252), vix.iloc[-1]) / 100
                        )

                    # VIX momentum
                    if len(vix) >= 5:
                        features['vix_change_1w'] = float(vix.iloc[-1] - vix.iloc[-5])
                    if len(vix) >= 21:
                        features['vix_change_1m'] = float(vix.iloc[-1] - vix.iloc[-21])

        except Exception as e:
            logger.warning(f"VIX features failed: {e}")

        # Fill missing
        for key in ['vix_level', 'vix_regime', 'vix_percentile', 'vix_change_1w', 'vix_change_1m']:
            if key not in features:
                features[key] = 0.0 if key != 'vix_level' else 15.0  # Default VIX

        return features

    def get_all_features(self, use_cache: bool = True) -> Dict[str, float]:
        """
        Generate all macro features.

        Args:
            use_cache: Use cached features if available

        Returns:
            Dict with all macro features (32 total)
        """
        if use_cache and self._is_cache_valid():
            return self._cache

        features = {}

        # Combine all feature groups
        features.update(self.get_yield_curve_features())
        features.update(self.get_rate_momentum_features())
        features.update(self.get_inflation_features())
        features.update(self.get_cot_features())
        features.update(self.get_regime_features())
        features.update(self.get_vix_features())

        # Add metadata
        features['_timestamp'] = datetime.now().timestamp()
        features['_feature_count'] = len([k for k in features if not k.startswith('_')])

        # Cache
        self._cache = features
        self._cache_time = datetime.now()

        return features

    def get_features_dataframe(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Get all features as a single-row DataFrame.

        Returns:
            DataFrame with features as columns
        """
        features = self.get_all_features(use_cache)

        # Remove metadata columns
        clean_features = {k: v for k, v in features.items() if not k.startswith('_')}

        return pd.DataFrame([clean_features])

    def get_feature_importance_hint(self) -> Dict[str, str]:
        """
        Get hints about which features are most important.

        Returns:
            Dict mapping feature names to importance descriptions
        """
        return {
            'yc_10y_2y_spread': 'HIGH - Classic recession indicator',
            'yc_is_inverted': 'HIGH - Strong recession signal',
            'vix_level': 'HIGH - Fear/risk gauge',
            'cot_es_spec_pct': 'MEDIUM - Crowding indicator',
            'macro_regime': 'HIGH - Overall environment',
            'rate_change_3m': 'MEDIUM - Policy momentum',
            'inflation_yoy': 'MEDIUM - Inflation pressure',
            'real_rate_10y': 'MEDIUM - Real borrowing cost',
        }


# Singleton instance
_generator: Optional[MacroFeatureGenerator] = None


def get_macro_feature_generator() -> MacroFeatureGenerator:
    """Get or create singleton feature generator."""
    global _generator
    if _generator is None:
        _generator = MacroFeatureGenerator()
    return _generator


def get_macro_features() -> Dict[str, float]:
    """Get all macro features (convenience function)."""
    return get_macro_feature_generator().get_all_features()


def get_macro_features_df() -> pd.DataFrame:
    """Get macro features as DataFrame (convenience function)."""
    return get_macro_feature_generator().get_features_dataframe()


if __name__ == "__main__":
    # Demo usage
    generator = MacroFeatureGenerator()

    print("=== Macro Feature Generator Demo ===\n")

    print("Generating all macro features...")
    features = generator.get_all_features()

    print(f"\nTotal features: {features.get('_feature_count', 0)}")

    # Group and display
    print("\n--- Yield Curve Features ---")
    for k, v in sorted(features.items()):
        if k.startswith('yc_'):
            print(f"  {k}: {v:.4f}")

    print("\n--- Rate Momentum Features ---")
    for k, v in sorted(features.items()):
        if k.startswith('rate_') or k.startswith('t10y_'):
            print(f"  {k}: {v:.4f}")

    print("\n--- Inflation Features ---")
    for k, v in sorted(features.items()):
        if k.startswith('inflation_') or k.startswith('real_'):
            print(f"  {k}: {v:.4f}")

    print("\n--- COT Features ---")
    for k, v in sorted(features.items()):
        if k.startswith('cot_'):
            print(f"  {k}: {v:.4f}")

    print("\n--- Regime Features ---")
    for k, v in sorted(features.items()):
        if k.startswith('macro_') or k.startswith('regime_'):
            print(f"  {k}: {v:.4f}")

    print("\n--- VIX Features ---")
    for k, v in sorted(features.items()):
        if k.startswith('vix_'):
            print(f"  {k}: {v:.4f}")

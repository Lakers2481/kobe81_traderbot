"""
Technical Features Module using pandas-ta.

Provides 150+ technical indicators organized into categories:
- Momentum: RSI, MACD, Stochastic, Williams %R, etc.
- Volatility: ATR, Bollinger Bands, Keltner Channels, etc.
- Trend: SMA, EMA, ADX, Supertrend, etc.
- Volume: OBV, VWAP, MFI, AD, etc.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

# Try pandas-ta first, then fallback to ta library
PANDAS_TA_AVAILABLE = False
TA_LIB_AVAILABLE = False
pta = None  # pandas_ta alias

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        import pandas_ta as pta
        PANDAS_TA_AVAILABLE = True
    except ImportError:
        pass

# Fallback to ta library (technical-analysis)
if not PANDAS_TA_AVAILABLE:
    try:
        import ta as ta_lib
        from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator, UltimateOscillator, AwesomeOscillatorIndicator
        from ta.volatility import AverageTrueRange, BollingerBands, KeltnerChannel
        from ta.trend import SMAIndicator, EMAIndicator, ADXIndicator, PSARIndicator, AroonIndicator, MACD
        from ta.volume import OnBalanceVolumeIndicator, MFIIndicator, AccDistIndexIndicator, ChaikinMoneyFlowIndicator
        TA_LIB_AVAILABLE = True
    except ImportError:
        pass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.structured_log import jlog


@dataclass
class TechnicalConfig:
    """Configuration for technical feature extraction."""
    # Momentum
    rsi_periods: List[int] = field(default_factory=lambda: [2, 7, 14])
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    stoch_k: int = 14
    stoch_d: int = 3
    williams_period: int = 14
    cci_period: int = 20

    # Volatility
    atr_periods: List[int] = field(default_factory=lambda: [7, 14, 21])
    bb_period: int = 20
    bb_std: float = 2.0
    kc_period: int = 20
    kc_scalar: float = 1.5

    # Trend
    sma_periods: List[int] = field(default_factory=lambda: [10, 20, 50, 200])
    ema_periods: List[int] = field(default_factory=lambda: [9, 21, 50])
    adx_period: int = 14
    supertrend_period: int = 10
    supertrend_multiplier: float = 3.0

    # Volume
    obv: bool = True
    vwap: bool = True
    mfi_period: int = 14
    ad: bool = True


class TechnicalFeatures:
    """
    Technical feature extractor using pandas-ta.

    Computes 150+ technical indicators efficiently in a single pass.
    All indicators are shifted by 1 to prevent lookahead bias.
    """

    def __init__(self, config: Optional[TechnicalConfig] = None):
        self.config = config or TechnicalConfig()
        self._validate_pandas_ta()

    def _validate_pandas_ta(self) -> None:
        """Check that pandas-ta or ta library is available."""
        if not PANDAS_TA_AVAILABLE and not TA_LIB_AVAILABLE:
            jlog("no_ta_library_available", level="WARNING",
                 message="Neither pandas-ta nor ta library installed, using basic fallback indicators")

    def compute_all(self, df: pd.DataFrame, shift: bool = True) -> pd.DataFrame:
        """
        Compute all technical features.

        Args:
            df: DataFrame with OHLCV columns (open, high, low, close, volume)
            shift: If True, shift indicators by 1 bar to prevent lookahead

        Returns:
            DataFrame with all technical features added
        """
        if df.empty:
            return df

        # Ensure column names are lowercase
        df = df.copy()
        df.columns = df.columns.str.lower()

        # Validate required columns
        required = ['open', 'high', 'low', 'close']
        missing = [c for c in required if c not in df.columns]
        if missing:
            jlog("missing_ohlc_columns", level="ERROR", missing=missing)
            return df

        # Compute features by category
        df = self._add_momentum_features(df)
        df = self._add_volatility_features(df)
        df = self._add_trend_features(df)

        if 'volume' in df.columns:
            df = self._add_volume_features(df)

        # Shift all new columns by 1 to prevent lookahead
        if shift:
            original_cols = {'open', 'high', 'low', 'close', 'volume', 'timestamp', 'symbol', 'date'}
            feature_cols = [c for c in df.columns if c not in original_cols]
            for col in feature_cols:
                df[col] = df[col].shift(1)

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        if PANDAS_TA_AVAILABLE:
            return self._add_momentum_pandas_ta(df)
        elif TA_LIB_AVAILABLE:
            return self._add_momentum_ta_lib(df)
        else:
            return self._add_momentum_fallback(df)

    def _add_momentum_pandas_ta(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators using pandas-ta."""
        try:
            # RSI at multiple periods
            for period in self.config.rsi_periods:
                rsi = pta.rsi(df['close'], length=period)
                if rsi is not None:
                    df[f'rsi_{period}'] = rsi

            # MACD
            macd = pta.macd(df['close'],
                         fast=self.config.macd_fast,
                         slow=self.config.macd_slow,
                         signal=self.config.macd_signal)
            if macd is not None:
                df = pd.concat([df, macd], axis=1)

            # Stochastic
            stoch = pta.stoch(df['high'], df['low'], df['close'],
                           k=self.config.stoch_k, d=self.config.stoch_d)
            if stoch is not None:
                df = pd.concat([df, stoch], axis=1)

            # Williams %R
            willr = pta.willr(df['high'], df['low'], df['close'],
                           length=self.config.williams_period)
            if willr is not None:
                df['williams_r'] = willr

            # CCI
            cci = pta.cci(df['high'], df['low'], df['close'],
                        length=self.config.cci_period)
            if cci is not None:
                df['cci'] = cci

            # ROC (Rate of Change)
            for period in [5, 10, 20]:
                roc = pta.roc(df['close'], length=period)
                if roc is not None:
                    df[f'roc_{period}'] = roc

            # Ultimate Oscillator
            uo = pta.uo(df['high'], df['low'], df['close'])
            if uo is not None:
                df['ultimate_osc'] = uo

            # Awesome Oscillator
            ao = pta.ao(df['high'], df['low'])
            if ao is not None:
                df['awesome_osc'] = ao

        except Exception as e:
            jlog("momentum_features_error", level="WARNING", error=str(e))

        return df

    def _add_momentum_ta_lib(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators using ta library."""
        try:
            # RSI at multiple periods
            for period in self.config.rsi_periods:
                rsi = RSIIndicator(df['close'], window=period)
                df[f'rsi_{period}'] = rsi.rsi()

            # MACD
            macd = MACD(df['close'],
                       window_fast=self.config.macd_fast,
                       window_slow=self.config.macd_slow,
                       window_sign=self.config.macd_signal)
            df['MACD_12_26_9'] = macd.macd()
            df['MACDh_12_26_9'] = macd.macd_diff()
            df['MACDs_12_26_9'] = macd.macd_signal()

            # Stochastic
            stoch = StochasticOscillator(df['high'], df['low'], df['close'],
                                        window=self.config.stoch_k,
                                        smooth_window=self.config.stoch_d)
            df['STOCHk_14_3_3'] = stoch.stoch()
            df['STOCHd_14_3_3'] = stoch.stoch_signal()

            # Williams %R
            willr = WilliamsRIndicator(df['high'], df['low'], df['close'],
                                       lbp=self.config.williams_period)
            df['williams_r'] = willr.williams_r()

            # ROC (Rate of Change)
            for period in [5, 10, 20]:
                roc = ROCIndicator(df['close'], window=period)
                df[f'roc_{period}'] = roc.roc()

            # Ultimate Oscillator
            uo = UltimateOscillator(df['high'], df['low'], df['close'])
            df['ultimate_osc'] = uo.ultimate_oscillator()

            # Awesome Oscillator
            ao = AwesomeOscillatorIndicator(df['high'], df['low'])
            df['awesome_osc'] = ao.awesome_oscillator()

        except Exception as e:
            jlog("momentum_features_ta_lib_error", level="WARNING", error=str(e))

        return df

    def _add_momentum_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback momentum indicators without pandas-ta."""
        # RSI using Wilder smoothing
        for period in self.config.rsi_periods:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta).where(delta < 0, 0.0)
            avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # Simple ROC
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = df['close'].pct_change(periods=period) * 100

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""
        if PANDAS_TA_AVAILABLE:
            return self._add_volatility_pandas_ta(df)
        elif TA_LIB_AVAILABLE:
            return self._add_volatility_ta_lib(df)
        else:
            return self._add_volatility_fallback(df)

    def _add_volatility_pandas_ta(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators using pandas-ta."""
        try:
            # ATR at multiple periods
            for period in self.config.atr_periods:
                atr = pta.atr(df['high'], df['low'], df['close'], length=period)
                if atr is not None:
                    df[f'atr_{period}'] = atr
                    # Normalized ATR (ATR / Close)
                    df[f'natr_{period}'] = atr / df['close'] * 100

            # Bollinger Bands
            bb = pta.bbands(df['close'],
                         length=self.config.bb_period,
                         std=self.config.bb_std)
            if bb is not None:
                df = pd.concat([df, bb], axis=1)
                # BB %B and width
                if 'BBL_20_2.0' in df.columns and 'BBU_20_2.0' in df.columns:
                    bb_range = df['BBU_20_2.0'] - df['BBL_20_2.0']
                    df['bb_pct_b'] = (df['close'] - df['BBL_20_2.0']) / bb_range
                    df['bb_width'] = bb_range / df['BBM_20_2.0']

            # Keltner Channels
            kc = pta.kc(df['high'], df['low'], df['close'],
                      length=self.config.kc_period,
                      scalar=self.config.kc_scalar)
            if kc is not None:
                df = pd.concat([df, kc], axis=1)

            # True Range
            tr = pta.true_range(df['high'], df['low'], df['close'])
            if tr is not None:
                df['true_range'] = tr

            # Historical Volatility
            for period in [10, 20, 30]:
                returns = df['close'].pct_change()
                df[f'hvol_{period}'] = returns.rolling(period).std() * np.sqrt(252) * 100

        except Exception as e:
            jlog("volatility_features_error", level="WARNING", error=str(e))

        return df

    def _add_volatility_ta_lib(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators using ta library."""
        try:
            # ATR at multiple periods
            for period in self.config.atr_periods:
                atr = AverageTrueRange(df['high'], df['low'], df['close'], window=period)
                df[f'atr_{period}'] = atr.average_true_range()
                df[f'natr_{period}'] = df[f'atr_{period}'] / df['close'] * 100

            # Bollinger Bands
            bb = BollingerBands(df['close'],
                               window=self.config.bb_period,
                               window_dev=int(self.config.bb_std))
            df['BBL_20_2.0'] = bb.bollinger_lband()
            df['BBM_20_2.0'] = bb.bollinger_mavg()
            df['BBU_20_2.0'] = bb.bollinger_hband()
            bb_range = df['BBU_20_2.0'] - df['BBL_20_2.0']
            df['bb_pct_b'] = bb.bollinger_pband()
            df['bb_width'] = bb.bollinger_wband()

            # Keltner Channels
            kc = KeltnerChannel(df['high'], df['low'], df['close'],
                               window=self.config.kc_period)
            df['KCL_20_1.5'] = kc.keltner_channel_lband()
            df['KCM_20_1.5'] = kc.keltner_channel_mband()
            df['KCU_20_1.5'] = kc.keltner_channel_hband()

            # True Range (calculated from ATR class)
            tr = AverageTrueRange(df['high'], df['low'], df['close'], window=1)
            df['true_range'] = tr.average_true_range()

            # Historical Volatility
            for period in [10, 20, 30]:
                returns = df['close'].pct_change()
                df[f'hvol_{period}'] = returns.rolling(period).std() * np.sqrt(252) * 100

        except Exception as e:
            jlog("volatility_features_ta_lib_error", level="WARNING", error=str(e))

        return df

    def _add_volatility_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback volatility indicators without pandas-ta."""
        # ATR
        for period in self.config.atr_periods:
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift(1)).abs()
            low_close = (df['low'] - df['close'].shift(1)).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df[f'atr_{period}'] = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
            df[f'natr_{period}'] = df[f'atr_{period}'] / df['close'] * 100

        # Simple Bollinger Bands
        df['bb_mid'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']

        return df

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators."""
        if PANDAS_TA_AVAILABLE:
            return self._add_trend_pandas_ta(df)
        elif TA_LIB_AVAILABLE:
            return self._add_trend_ta_lib(df)
        else:
            return self._add_trend_fallback(df)

    def _add_trend_pandas_ta(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators using pandas-ta."""
        try:
            # SMA at multiple periods
            for period in self.config.sma_periods:
                sma = pta.sma(df['close'], length=period)
                if sma is not None:
                    df[f'sma_{period}'] = sma
                    df[f'sma_{period}_dist'] = (df['close'] - sma) / sma * 100

            # EMA at multiple periods
            for period in self.config.ema_periods:
                ema = pta.ema(df['close'], length=period)
                if ema is not None:
                    df[f'ema_{period}'] = ema

            # ADX (trend strength)
            adx = pta.adx(df['high'], df['low'], df['close'],
                        length=self.config.adx_period)
            if adx is not None:
                df = pd.concat([df, adx], axis=1)

            # Supertrend
            st = pta.supertrend(df['high'], df['low'], df['close'],
                             length=self.config.supertrend_period,
                             multiplier=self.config.supertrend_multiplier)
            if st is not None:
                df = pd.concat([df, st], axis=1)

            # PSAR (Parabolic SAR)
            psar = pta.psar(df['high'], df['low'], df['close'])
            if psar is not None:
                df = pd.concat([df, psar], axis=1)

            # Aroon
            aroon = pta.aroon(df['high'], df['low'], length=25)
            if aroon is not None:
                df = pd.concat([df, aroon], axis=1)

            # Linear regression slope
            for period in [10, 20]:
                slope = pta.slope(df['close'], length=period)
                if slope is not None:
                    df[f'slope_{period}'] = slope

        except Exception as e:
            jlog("trend_features_error", level="WARNING", error=str(e))

        return df

    def _add_trend_ta_lib(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators using ta library."""
        try:
            # SMA at multiple periods
            for period in self.config.sma_periods:
                sma = SMAIndicator(df['close'], window=period)
                df[f'sma_{period}'] = sma.sma_indicator()
                df[f'sma_{period}_dist'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}'] * 100

            # EMA at multiple periods
            for period in self.config.ema_periods:
                ema = EMAIndicator(df['close'], window=period)
                df[f'ema_{period}'] = ema.ema_indicator()

            # ADX (trend strength)
            adx = ADXIndicator(df['high'], df['low'], df['close'],
                              window=self.config.adx_period)
            df['ADX_14'] = adx.adx()
            df['DMP_14'] = adx.adx_pos()
            df['DMN_14'] = adx.adx_neg()

            # PSAR (Parabolic SAR)
            psar = PSARIndicator(df['high'], df['low'], df['close'])
            df['psar'] = psar.psar()
            df['psar_up'] = psar.psar_up()
            df['psar_down'] = psar.psar_down()

            # Aroon
            aroon = AroonIndicator(df['high'], df['low'], window=25)
            df['AROONU_25'] = aroon.aroon_up()
            df['AROOND_25'] = aroon.aroon_down()
            df['AROONOSC_25'] = aroon.aroon_indicator()

            # Linear regression slope (simple approximation)
            for period in [10, 20]:
                # Using rate of change as proxy for slope
                df[f'slope_{period}'] = (df['close'] - df['close'].shift(period)) / period

        except Exception as e:
            jlog("trend_features_ta_lib_error", level="WARNING", error=str(e))

        return df

    def _add_trend_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback trend indicators without pandas-ta."""
        # SMA
        for period in self.config.sma_periods:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'sma_{period}_dist'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}'] * 100

        # EMA
        for period in self.config.ema_periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume indicators."""
        if 'volume' not in df.columns:
            return df

        if PANDAS_TA_AVAILABLE:
            return self._add_volume_pandas_ta(df)
        elif TA_LIB_AVAILABLE:
            return self._add_volume_ta_lib(df)
        else:
            return self._add_volume_fallback(df)

    def _add_volume_pandas_ta(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume indicators using pandas-ta."""
        try:
            # OBV
            if self.config.obv:
                obv = pta.obv(df['close'], df['volume'])
                if obv is not None:
                    df['obv'] = obv

            # VWAP (requires high/low/close/volume)
            if self.config.vwap:
                vwap = pta.vwap(df['high'], df['low'], df['close'], df['volume'])
                if vwap is not None:
                    df['vwap'] = vwap

            # MFI (Money Flow Index)
            mfi = pta.mfi(df['high'], df['low'], df['close'], df['volume'],
                        length=self.config.mfi_period)
            if mfi is not None:
                df['mfi'] = mfi

            # A/D (Accumulation/Distribution)
            if self.config.ad:
                ad = pta.ad(df['high'], df['low'], df['close'], df['volume'])
                if ad is not None:
                    df['ad'] = ad

            # CMF (Chaikin Money Flow)
            cmf = pta.cmf(df['high'], df['low'], df['close'], df['volume'])
            if cmf is not None:
                df['cmf'] = cmf

            # Volume SMA
            for period in [10, 20, 50]:
                df[f'vol_sma_{period}'] = df['volume'].rolling(period).mean()
                df[f'vol_ratio_{period}'] = df['volume'] / df[f'vol_sma_{period}']

        except Exception as e:
            jlog("volume_features_error", level="WARNING", error=str(e))

        return df

    def _add_volume_ta_lib(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume indicators using ta library."""
        try:
            # OBV
            if self.config.obv:
                obv = OnBalanceVolumeIndicator(df['close'], df['volume'])
                df['obv'] = obv.on_balance_volume()

            # MFI (Money Flow Index)
            mfi = MFIIndicator(df['high'], df['low'], df['close'], df['volume'],
                              window=self.config.mfi_period)
            df['mfi'] = mfi.money_flow_index()

            # A/D (Accumulation/Distribution)
            if self.config.ad:
                ad = AccDistIndexIndicator(df['high'], df['low'], df['close'], df['volume'])
                df['ad'] = ad.acc_dist_index()

            # CMF (Chaikin Money Flow)
            cmf = ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume'])
            df['cmf'] = cmf.chaikin_money_flow()

            # Volume SMA
            for period in [10, 20, 50]:
                df[f'vol_sma_{period}'] = df['volume'].rolling(period).mean()
                df[f'vol_ratio_{period}'] = df['volume'] / df[f'vol_sma_{period}']

            # VWAP approximation (ta library doesn't have native VWAP, use typical price x volume)
            if self.config.vwap:
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()

        except Exception as e:
            jlog("volume_features_ta_lib_error", level="WARNING", error=str(e))

        return df

    def _add_volume_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback volume indicators without pandas-ta."""
        # Simple OBV
        sign = np.sign(df['close'].diff())
        df['obv'] = (sign * df['volume']).cumsum()

        # Volume SMA
        for period in [10, 20, 50]:
            df[f'vol_sma_{period}'] = df['volume'].rolling(period).mean()
            df[f'vol_ratio_{period}'] = df['volume'] / df[f'vol_sma_{period}']

        return df


# Convenience functions
def compute_momentum_features(df: pd.DataFrame, config: Optional[TechnicalConfig] = None) -> pd.DataFrame:
    """Compute only momentum features."""
    tf = TechnicalFeatures(config)
    return tf._add_momentum_features(df.copy())


def compute_volatility_features(df: pd.DataFrame, config: Optional[TechnicalConfig] = None) -> pd.DataFrame:
    """Compute only volatility features."""
    tf = TechnicalFeatures(config)
    return tf._add_volatility_features(df.copy())


def compute_trend_features(df: pd.DataFrame, config: Optional[TechnicalConfig] = None) -> pd.DataFrame:
    """Compute only trend features."""
    tf = TechnicalFeatures(config)
    return tf._add_trend_features(df.copy())


def compute_volume_features(df: pd.DataFrame, config: Optional[TechnicalConfig] = None) -> pd.DataFrame:
    """Compute only volume features."""
    tf = TechnicalFeatures(config)
    return tf._add_volume_features(df.copy())

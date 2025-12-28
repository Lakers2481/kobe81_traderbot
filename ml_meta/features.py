from __future__ import annotations

import math
import pandas as pd


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    o, h, l, c = df['open'], df['high'], df['low'], df['close']
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def _realized_vol(c: pd.Series, n: int = 20) -> pd.Series:
    r = c.pct_change()
    return r.rolling(n, min_periods=n).std()


def _donchian_hi(lo: pd.Series, hi: pd.Series, n: int) -> pd.Series:
    return hi.rolling(n, min_periods=n).max()


def _donchian_lo(lo: pd.Series, hi: pd.Series, n: int) -> pd.Series:
    return lo.rolling(n, min_periods=n).min()


def compute_features_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling features per (symbol,timestamp).

    Input df must contain: ['timestamp','symbol','open','high','low','close','volume']
    It can be multiple symbols concatenated. Returns a DataFrame with the same
    index as df and feature columns; caller can merge on ['symbol','timestamp'].
    """
    if df.empty:
        return pd.DataFrame(columns=['timestamp','symbol'])

    out = df[['timestamp','symbol']].copy()
    out['atr14'] = 0.0
    out['sma20_over_200'] = 0.0
    out['rv20'] = 0.0
    out['don20_width'] = 0.0
    out['pos_in_don20'] = 0.0
    out['ret5'] = 0.0
    out['log_vol'] = 0.0

    def _compute(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values('timestamp').copy()
        atr14 = _atr(g, 14)
        sma20 = _sma(g['close'], 20)
        sma200 = _sma(g['close'], 200)
        rv20 = _realized_vol(g['close'], 20)
        d_hi20 = _donchian_hi(g['low'], g['high'], 20)
        d_lo20 = _donchian_lo(g['low'], g['high'], 20)
        width = (d_hi20 - d_lo20).replace(0, float('nan'))
        pos = (g['close'] - d_lo20) / width
        ret5 = g['close'].pct_change(5)
        log_vol = g['volume'].apply(lambda x: math.log(x) if pd.notna(x) and x > 0 else float('nan'))
        return pd.DataFrame({
            'timestamp': g['timestamp'],
            'symbol': g['symbol'],
            'atr14': atr14,
            'sma20_over_200': (sma20 / sma200).fillna(0.0),
            'rv20': rv20,
            'don20_width': width,
            'pos_in_don20': pos,
            'ret5': ret5,
            'log_vol': log_vol,
        })

    parts = []
    for sym, g in df.groupby('symbol'):
        parts.append(_compute(g))
    f = pd.concat(parts, ignore_index=True)
    # Fill remaining NaNs conservatively with zeros for modeling
    for col in ['atr14','sma20_over_200','rv20','don20_width','pos_in_don20','ret5','log_vol']:
        if col in f:
            f[col] = f[col].astype(float).fillna(0.0)
    return f


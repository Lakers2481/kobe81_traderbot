from __future__ import annotations

"""
Anomaly detection helpers (price/volume) for self-monitoring.

Provides lightweight z-score based detectors and a simple volume spike flag.
Optionally emits Telegram alerts via core.alerts when thresholds are exceeded.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd


@dataclass
class AnomalyConfig:
    z_window: int = 20
    z_threshold: float = 4.0    # 4-sigma return spike
    vol_window: int = 20
    vol_multiplier: float = 3.0 # 3x average volume
    telegram: bool = False


def _zscore(s: pd.Series, win: int) -> pd.Series:
    m = s.rolling(win, min_periods=win).mean()
    sd = s.rolling(win, min_periods=win).std()
    return (s - m) / sd.replace(0, pd.NA)


def detect_anomalies(df: pd.DataFrame, cfg: Optional[AnomalyConfig] = None) -> Dict[str, Any]:
    cfg = cfg or AnomalyConfig()
    out: Dict[str, Any] = {"price_z": None, "vol_spike": None, "symbol": None, "timestamp": None}
    if df.empty:
        return out
    df = df.sort_values("timestamp")
    if 'symbol' in df:
        # analyze last symbol chunk for simplicity
        last_sym = df['symbol'].iloc[-1]
        g = df[df['symbol'] == last_sym].copy()
    else:
        g = df.copy()
        last_sym = None
    c = g['close'].astype(float)
    v = g['volume'].astype(float)
    r = c.pct_change().fillna(0.0)
    z = _zscore(r, cfg.z_window)
    vol_avg = v.rolling(cfg.vol_window, min_periods=cfg.vol_window).mean()
    vol_spike = (v > (cfg.vol_multiplier * vol_avg))
    # last row
    if len(g) > 0:
        ts = g['timestamp'].iloc[-1]
        out.update({
            "price_z": float(z.iloc[-1]) if pd.notna(z.iloc[-1]) else None,
            "vol_spike": bool(vol_spike.iloc[-1]) if pd.notna(vol_spike.iloc[-1]) else False,
            "symbol": str(last_sym) if last_sym is not None else None,
            "timestamp": str(ts),
        })
        # Optional alert
        if cfg.telegram and ((out["price_z"] is not None and abs(out["price_z"]) >= cfg.z_threshold) or out["vol_spike"]):
            try:
                from core.alerts import send_telegram
                try:
                    from core.clock.tz_utils import fmt_ct, now_et
                    now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                except Exception:
                    stamp = None
                msg = f"[ANOMALY] {out['symbol'] or ''} ts={out['timestamp']} z={out['price_z']} vol_spike={out['vol_spike']}" + (f" [{stamp}]" if stamp else '')
                send_telegram(msg)
            except Exception:
                pass
    return out

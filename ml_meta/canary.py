from __future__ import annotations

"""
Canary scoring: annotate signals with both deployed and candidate model probabilities.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def annotate_with_canary(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return signals
    try:
        from ml_meta.model import model_paths, FEATURE_COLS
        import joblib
        df = signals.copy()
        for strat in ('ibs_rsi','turtle_soup'):
            if strat == 'turtle_soup':
                mask = df['strategy'].astype(str).str.lower().isin(['turtle_soup','ict'])
            else:
                mask = df['strategy'].astype(str).str.lower().isin(['ibs_rsi','ibs','rsi2'])
            if not mask.any():
                continue
            _, cand_meta = model_paths(strat, kind='candidate')
            cand_pkl, _ = model_paths(strat, kind='candidate')
            dep_pkl, _ = model_paths(strat, kind='deployed')
            if not cand_pkl.exists() or not dep_pkl.exists():
                continue
            cand = joblib.load(cand_pkl)
            dep = joblib.load(dep_pkl)
            X = df.loc[mask, FEATURE_COLS].astype(float)
            try:
                df.loc[mask, f'conf_canary_{strat}'] = cand.predict_proba(X)[:,1]
                df.loc[mask, f'conf_deployed_{strat}'] = dep.predict_proba(X)[:,1]
            except Exception:
                pass
        return df
    except Exception:
        return signals

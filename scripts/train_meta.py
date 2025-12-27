#!/usr/bin/env python3
from __future__ import annotations

"""
Train per-strategy ML meta-models to score signals (success probability).

Inputs: data/ml/signal_dataset.parquet (from build_signal_dataset.py)
Outputs: state/models/meta_donchian.pkl/.json, meta_turtle_soup.pkl/.json
"""

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml_meta.model import default_model, FEATURE_COLS, save_model, CANDIDATE_DIR
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss


def time_split3(df: pd.DataFrame, frac_train: float = 0.6, frac_calib: float = 0.2):
    df = df.sort_values('timestamp')
    n = len(df)
    t_cut = int(n * frac_train)
    c_cut = int(n * (frac_train + frac_calib))
    return df.iloc[:t_cut], df.iloc[t_cut:c_cut], df.iloc[c_cut:]


def train_for_strategy(df: pd.DataFrame, strategy_name: str) -> Dict:
    # Filter rows
    key = strategy_name.upper()
    sdf = df[df['strategy'].astype(str).str.upper() == key].copy()
    if sdf.empty:
        return {"status": "EMPTY"}

    # Train/calibrate/test split
    train_df, calib_df, test_df = time_split3(sdf, 0.6, 0.2)
    X_tr = train_df[FEATURE_COLS].astype(float).values
    y_tr = train_df['label'].astype(int).values
    X_c = calib_df[FEATURE_COLS].astype(float).values if not calib_df.empty else None
    y_c = calib_df['label'].astype(int).values if not calib_df.empty else None
    X_te = test_df[FEATURE_COLS].astype(float).values
    y_te = test_df['label'].astype(int).values

    base = default_model()
    if X_c is not None and len(X_c) > 0:
        base.fit(X_tr, y_tr)
        model = CalibratedClassifierCV(base, method='isotonic', cv='prefit')
        model.fit(X_c, y_c)
    else:
        model = CalibratedClassifierCV(base, method='sigmoid', cv=3)
        model.fit(X_tr, y_tr)

    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc = float((pred == y_te).mean()) if len(y_te) else 0.0
    baseline = float(max((y_te.mean(), 1 - y_te.mean()))) if len(y_te) else 0.0
    brier = float(brier_score_loss(y_te, proba)) if len(y_te) else 1.0

    # Additional trading metrics from dataset
    wr = float(test_df['label'].mean()) if len(test_df) else 0.0
    pf = 0.0; sharpe = 0.0
    if 'ret' in test_df.columns and len(test_df) > 0:
        rets = test_df['ret'].astype(float).fillna(0.0)
        pos = rets[rets > 0].sum()
        neg = rets[rets < 0].sum()
        pf = float(pos / abs(neg)) if neg < 0 else float('inf') if pos > 0 else 0.0
        if rets.std(ddof=0) > 0:
            sharpe = float((rets.mean() / (rets.std(ddof=0) + 1e-9)) * np.sqrt(252))

    meta = {
        "status": "OK",
        "rows": len(sdf),
        "test_rows": int(len(y_te)),
        "accuracy": acc,
        "baseline": baseline,
        "brier": brier,
        "wr": wr,
        "pf": pf,
        "sharpe": sharpe,
    }
    save_model(strategy_name, model, meta, kind='candidate')
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description='Train ML meta-models for Donchian/ICT')
    ap.add_argument('--dataset', type=str, default='data/ml/signal_dataset.parquet')
    ap.add_argument('--outdir', type=str, default=str(CANDIDATE_DIR))
    args = ap.parse_args()

    p = Path(args.dataset)
    if not p.exists():
        print('Dataset not found:', p)
        return
    df = pd.read_parquet(p)
    # Ensure feature columns exist
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0

    results = {
        'donchian': train_for_strategy(df, 'donchian'),
        'turtle_soup': train_for_strategy(df, 'turtle_soup'),
    }
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'meta_train_summary.json').write_text(pd.Series(results).to_json(indent=2))
    print('Training complete. Summary written to', outdir / 'meta_train_summary.json')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
from __future__ import annotations

"""
Morning Report

Summarizes overnight insights:
- Sentiment leaders/laggards (change vs prior day if available)
- Model changes (promotions, accuracy)
- SPY regime & anomaly highlights (matrix profile discords)

Outputs: reports/morning_report_YYYYMMDD.html
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from altdata.sentiment import load_daily_cache, normalize_sentiment_to_conf
from data.providers.polygon_eod import fetch_daily_bars_polygon
from ml_meta.model import DEPLOYED_DIR, load_model, FEATURE_COLS
from core.journal import append_journal


def load_json(p: Path) -> dict:
    if not p.exists():
        return {}
    try:
        return pd.read_json(p).to_dict()
    except Exception:
        try:
            import json
            return json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            return {}


def sentiment_section(day: str, prev_day: Optional[str]) -> str:
    cur = load_daily_cache(day)
    if cur.empty:
        return '<h2>Sentiment</h2><p>No news sentiment available.</p>'
    cur['conf'] = cur['sent_mean'].apply(normalize_sentiment_to_conf)
    top = cur.sort_values('conf', ascending=False).head(10)[['symbol','sent_mean','sent_count','conf']]
    low = cur.sort_values('conf', ascending=True).head(10)[['symbol','sent_mean','sent_count','conf']]
    html = ['<h2>Sentiment</h2>', '<h3>Leaders</h3>', top.to_html(index=False), '<h3>Laggards</h3>', low.to_html(index=False)]
    if prev_day:
        prev = load_daily_cache(prev_day)
        if not prev.empty:
            prev = prev[['symbol','sent_mean']].rename(columns={'sent_mean':'prev_sent'})
            chg = pd.merge(cur[['symbol','sent_mean']], prev, on='symbol', how='left')
            chg['delta'] = chg['sent_mean'] - chg['prev_sent']
            chg = chg.dropna(subset=['delta']).sort_values('delta', ascending=False)
            html += ['<h3>Top Sentiment Changes</h3>', chg.head(10).to_html(index=False)]
    return '\n'.join(html)


def spy_regime_and_anomalies(start: str, end: str) -> str:
    try:
        df = fetch_daily_bars_polygon('SPY', start, end, cache_dir=ROOT / 'data' / 'cache')
        if df.empty:
            return '<h2>SPY Regime</h2><p>No SPY data</p>'
        df = df.sort_values('timestamp')
        df['sma200'] = df['close'].rolling(200, min_periods=200).mean()
        regime = 'BULL' if df['close'].iloc[-1] >= df['sma200'].iloc[-1] else 'BEAR/NEUTRAL'
        # Anomalies via matrix profile (last 180 bars; discord top-3)
        try:
            import stumpy
            closes = df['close'].tail(180).to_numpy(dtype=np.float64)
            m = 14
            if len(closes) > m * 2:
                mp = stumpy.stump(closes, m)
                prof = mp[:, 0]
                idxs = np.argsort(prof)[-3:][::-1]
                discords = ', '.join([str(int(i)) for i in idxs])
                an_text = f"Discord indices (m={m}): {discords}"
            else:
                an_text = "Insufficient length for matrix profile."
        except Exception as e:
            an_text = f"Matrix profile unavailable: {e}"
        return f"<h2>SPY Regime</h2><p>Regime: {regime}</p><p>{an_text}</p>"
    except Exception:
        return '<h2>SPY Regime</h2><p>Error computing regime</p>'


def model_section() -> str:
    dep = DEPLOYED_DIR / 'meta_current.json'
    if not dep.exists():
        return '<h2>Models</h2><p>No deployed summary available.</p>'
    try:
        import json
        data = json.loads(dep.read_text(encoding='utf-8'))
        rows = []
        for k, v in data.items():
            rows.append({'strategy': k, 'accuracy': v.get('accuracy', 0.0), 'test_rows': v.get('test_rows', 0), 'status': v.get('status', '')})
        df = pd.DataFrame(rows).sort_values('strategy')
        return '<h2>Models</h2>' + df.to_html(index=False)
    except Exception:
        return '<h2>Models</h2><p>Could not load deployed summary.</p>'


def calibration_section() -> str:
    """Show simple reliability table over recent dataset if available."""
    ds = ROOT / 'data' / 'ml' / 'signal_dataset.parquet'
    if not ds.exists():
        return '<h2>Calibration</h2><p>Dataset not found.</p>'
    try:
        df = pd.read_parquet(ds)
        if df.empty:
            return '<h2>Calibration</h2><p>No data.</p>'
        # last 180 days
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=180)
        df = df[df['timestamp'] >= cutoff]
        if df.empty:
            return '<h2>Calibration</h2><p>No recent data (last 180d).</p>'
        # Score both strategies separately and aggregate
        rows = []
        for strat in ('DONCHIAN','TURTLE_SOUP'):
            sdf = df[df['strategy'].astype(str).str.upper() == strat]
            if sdf.empty:
                continue
            m = load_model('donchian' if strat == 'DONCHIAN' else 'turtle_soup')
            if m is None:
                continue
            X = sdf[FEATURE_COLS].astype(float)
            try:
                p = m.predict_proba(X)[:, 1]
            except Exception:
                continue
            lab = sdf['label'].astype(int).values if 'label' in sdf.columns else None
            bins = pd.cut(p, bins=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], include_lowest=True)
            dfb = pd.DataFrame({'proba': p, 'label': lab, 'bin': bins})
            grp = dfb.groupby('bin').agg(count=('label','size'), pred_mean=('proba','mean'), obs_rate=('label','mean'))
            grp['strategy'] = strat
            rows.append(grp.reset_index())
        if not rows:
            return '<h2>Calibration</h2><p>No model scores available.</p>'
        out = pd.concat(rows, ignore_index=True)
        out['pred_mean'] = (out['pred_mean']*100).round(1)
        out['obs_rate'] = (out['obs_rate']*100).round(1)
        show = out[['strategy','bin','count','pred_mean','obs_rate']]
        return '<h2>Calibration (last 180d)</h2>' + show.to_html(index=False)
    except Exception as e:
        return f'<h2>Calibration</h2><p>Error: {e}</p>'


def feature_impact_section() -> str:
    """Show feature impact using SHAP if available; else coefficients."""
    # Try Donchian model first
    m = load_model('donchian')
    if m is None:
        return '<h2>Feature Impact</h2><p>No deployed model.</p>'
    # Attempt SHAP
    try:
        import shap  # optional dependency
        # Build a small synthetic or cached dataset for background
        ds = ROOT / 'data' / 'ml' / 'signal_dataset.parquet'
        X = None
        if ds.exists():
            df = pd.read_parquet(ds)
            if not df.empty:
                X = df[FEATURE_COLS].astype(float).head(512)
        if X is None or X.empty:
            # fallback random-like background to shape the explainer
            X = pd.DataFrame({c: np.zeros(64, dtype=float) for c in FEATURE_COLS})
        explainer = shap.Explainer(m.predict_proba, X, feature_names=FEATURE_COLS)
        sv = explainer(X)
        # Mean absolute SHAP values per feature
        imp = np.abs(sv.values[..., 1]).mean(axis=0)
        df_imp = pd.DataFrame({'feature': FEATURE_COLS, 'mean_abs_shap': imp}).sort_values('mean_abs_shap', ascending=False).head(10)
        df_imp['mean_abs_shap'] = df_imp['mean_abs_shap'].round(5)
        return '<h2>Feature Impact (SHAP)</h2>' + df_imp.to_html(index=False)
    except Exception:
        pass
    # Fallback to coefficients if linear model accessible
    try:
        from sklearn.pipeline import Pipeline
        import numpy as np
        est = None
        if hasattr(m, 'calibrated_classifiers_') and m.calibrated_classifiers_:
            cc = m.calibrated_classifiers_[0]
            est = getattr(cc, 'estimator', None) or getattr(cc, 'base_estimator', None)
        elif hasattr(m, 'base_estimator'):
            est = m.base_estimator
        if isinstance(est, Pipeline) and 'clf' in est.named_steps:
            clf = est.named_steps['clf']
            coef = getattr(clf, 'coef_', None)
            if coef is not None:
                coef = coef.reshape(-1)
                df = pd.DataFrame({'feature': FEATURE_COLS, 'coef': coef, 'abs_coef': np.abs(coef)})
                df = df.sort_values('abs_coef', ascending=False).head(10)
                df['coef'] = df['coef'].round(4)
                df['abs_coef'] = df['abs_coef'].round(4)
                return '<h2>Feature Impact (coefficients)</h2>' + df[['feature','coef','abs_coef']].to_html(index=False)
    except Exception as e:
        return f'<h2>Feature Impact</h2><p>Error: {e}</p>'
    return '<h2>Feature Impact</h2><p>Unavailable (no SHAP or accessible coefficients).</p>'


def main() -> None:
    ap = argparse.ArgumentParser(description='Generate morning report (overnight insights)')
    ap.add_argument('--date', type=str, default=None)
    args = ap.parse_args()

    today = datetime.now().date() if not args.date else datetime.fromisoformat(args.date).date()
    prev = today - timedelta(days=1)
    start = (today - timedelta(days=540)).isoformat()
    end = today.isoformat()

    html = ['<html><head><meta charset="utf-8"><title>Kobe Morning Report</title>',
            '<style>body{font-family:Arial;margin:20px} table{border-collapse:collapse} th,td{border:1px solid #ddd;padding:6px} th{background:#f3f3f3}</style>',
            '</head><body>', f'<h1>Morning Report - {today.isoformat()}</h1>']

    html.append(sentiment_section(today.isoformat(), prev.isoformat()))
    html.append(spy_regime_and_anomalies(start, end))
    html.append(model_section())
    html.append(calibration_section())
    html.append(feature_impact_section())
    html.append('</body></html>')

    outdir = ROOT / 'reports'
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f'morning_report_{today.strftime("%Y%m%d")}.html'
    out.write_text('\n'.join(html), encoding='utf-8')
    print('Morning report written:', out)
    # Journal insights
    try:
        payload = {
            'report': str(out),
            'day': today.isoformat(),
        }
        append_journal('morning_report', payload)
    except Exception:
        pass


if __name__ == '__main__':
    main()


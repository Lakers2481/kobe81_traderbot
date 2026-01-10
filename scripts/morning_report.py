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
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from core.clock.tz_utils import fmt_ct, fmt_et, now_et

from altdata.sentiment import load_daily_cache, normalize_sentiment_to_conf
from data.providers.polygon_eod import fetch_daily_bars_polygon
from ml_meta.model import DEPLOYED_DIR, load_model, FEATURE_COLS, model_paths
from ml_meta.features import compute_features_frame
from explainability.trade_explainer import explain_trade, ExplainConfig
try:
    from explainability.narrative_gen import generate as gen_narrative
except Exception:
    def gen_narrative(_sig: dict) -> dict:
        return {"technical": "", "casual": "", "executive": ""}
import io
import base64
import joblib


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
    try:
        scored = score_recent_dataset(lookback_days=180)
        if not scored:
            return '<h2>Calibration</h2><p>No model scores available.</p>'
        rows = []
        for strat, sdf in scored.items():
            if sdf.empty:
                continue
            bins = pd.cut(sdf['proba'].astype(float), bins=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], include_lowest=True)
            dfb = pd.DataFrame({'proba': sdf['proba'].astype(float), 'label': sdf['label'].astype(int), 'bin': bins})
            grp = dfb.groupby('bin').agg(count=('label','size'), pred_mean=('proba','mean'), obs_rate=('label','mean'))
            grp['strategy'] = strat
            rows.append(grp.reset_index())
        out = pd.concat(rows, ignore_index=True)
        out['pred_mean'] = (out['pred_mean']*100).round(1)
        out['obs_rate'] = (out['obs_rate']*100).round(1)
        show = out[['strategy','bin','count','pred_mean','obs_rate']]
        return '<h2>Calibration (last 180d)</h2>' + show.to_html(index=False)
    except Exception as e:
        return f'<h2>Calibration</h2><p>Error: {e}</p>'


def score_recent_dataset(lookback_days: int = 180) -> dict:
    """Return per-strategy DataFrames with columns proba,label,ret for recent window."""
    ds = ROOT / 'data' / 'ml' / 'signal_dataset.parquet'
    if not ds.exists():
        return {}
    df = pd.read_parquet(ds)
    if df.empty:
        return {}
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=int(lookback_days))
    df = df[df['timestamp'] >= cutoff]
    if df.empty:
        return {}
    out = {}
    for strat_key, model_key in (('IBS_RSI','ibs_rsi'), ('TURTLE_SOUP','turtle_soup')):
        sdf = df[df['strategy'].astype(str).str.upper() == strat_key]
        if sdf.empty:
            continue
        m = load_model(model_key)
        if m is None:
            continue
        X = sdf[FEATURE_COLS].astype(float)
        try:
            proba = m.predict_proba(X)[:, 1]
            out[strat_key] = pd.DataFrame({
                'proba': proba,
                'label': sdf['label'].astype(int).values if 'label' in sdf.columns else 0,
                'ret': sdf['ret'].astype(float).values if 'ret' in sdf.columns else 0.0,
            })
        except Exception:
            pass
    return out


def threshold_sweep_section() -> str:
    """Evaluate thresholds and show coverage/accuracy/wr/pf for each strategy."""
    try:
        scored = score_recent_dataset(lookback_days=180)
        if not scored:
            return ''
        thrs = [0.4, 0.5, 0.6, 0.7, 0.8]
        rows = []
        for strat, sdf in scored.items():
            n = len(sdf)
            if n == 0:
                continue
            for t in thrs:
                sel = sdf[sdf['proba'] >= t]
                cov = len(sel) / n if n else 0.0
                if len(sel) > 0:
                    yhat = (sel['proba'] >= t).astype(int)
                    acc = float((yhat.values == sel['label'].values).mean())
                    wr = float(sel['label'].mean())
                    pos = float(sel[sel['ret'] > 0]['ret'].sum())
                    neg = float(sel[sel['ret'] < 0]['ret'].sum())
                    pf = (pos / abs(neg)) if neg < 0 else (float('inf') if pos > 0 else 0.0)
                else:
                    acc = 0.0; wr = 0.0; pf = 0.0
                rows.append({'strategy': strat, 'threshold': t, 'coverage': round(cov,3), 'accuracy': round(acc,3), 'wr': round(wr,3), 'pf': (round(pf,2) if pf != float('inf') else 'inf')})
        if not rows:
            return ''
        df = pd.DataFrame(rows)
        return '<h2>Threshold Sweep (last 180d)</h2>' + df.to_html(index=False)
    except Exception as e:
        return f'<h2>Threshold Sweep</h2><p>Error: {e}</p>'


def score_recent_dataset_kind(lookback_days: int, kind: str) -> dict:
    """Score recent dataset with 'deployed' or 'candidate' models."""
    try:
        ds = ROOT / 'data' / 'ml' / 'signal_dataset.parquet'
        if not ds.exists():
            return {}
        df = pd.read_parquet(ds)
        if df.empty:
            return {}
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=int(lookback_days))
        df = df[df['timestamp'] >= cutoff]
        if df.empty:
            return {}
        out = {}
        for strat_key, model_key in (('IBS_RSI','ibs_rsi'), ('TURTLE_SOUP','turtle_soup')):
            sdf = df[df['strategy'].astype(str).str.upper() == strat_key]
            if sdf.empty:
                continue
            pkl, _ = model_paths(model_key, kind=('candidate' if kind=='candidate' else 'deployed'))
            if not pkl.exists():
                continue
            m = joblib.load(pkl)
            X = sdf[FEATURE_COLS].astype(float)
            proba = m.predict_proba(X)[:, 1]
            out[strat_key] = pd.DataFrame({
                'proba': proba,
                'label': sdf['label'].astype(int).values if 'label' in sdf.columns else 0,
                'ret': sdf['ret'].astype(float).values if 'ret' in sdf.columns else 0.0,
            })
        return out
    except Exception:
        return {}


def canary_section() -> str:
    """Compare deployed vs candidate over last 180d; show delta accuracy and PF."""
    try:
        dep = score_recent_dataset_kind(180, 'deployed')
        can = score_recent_dataset_kind(180, 'candidate')
        if not dep or not can:
            return ''
        rows = []
        for strat in ('IBS_RSI','TURTLE_SOUP'):
            if strat not in dep or strat not in can:
                continue
            dl = dep[strat]
            cl = can[strat]
            n = min(len(dl), len(cl))
            if n == 0:
                continue
            dl = dl.head(n); cl = cl.head(n)
            y = dl['label'].astype(int)
            yhat_d = (dl['proba'] >= 0.5).astype(int)
            yhat_c = (cl['proba'] >= 0.5).astype(int)
            acc_d = float((yhat_d.values == y.values).mean())
            acc_c = float((yhat_c.values == y.values).mean())
            pos_d = float(dl[dl['ret'] > 0]['ret'].sum()); neg_d = float(dl[dl['ret'] < 0]['ret'].sum())
            pf_d = (pos_d / abs(neg_d)) if neg_d < 0 else (float('inf') if pos_d > 0 else 0.0)
            pos_c = float(cl[cl['ret'] > 0]['ret'].sum()); neg_c = float(cl[cl['ret'] < 0]['ret'].sum())
            pf_c = (pos_c / abs(neg_c)) if neg_c < 0 else (float('inf') if pos_c > 0 else 0.0)
            rows.append({
                'strategy': strat,
                'acc_deployed': round(acc_d,3),
                'acc_canary': round(acc_c,3),
                'pf_deployed': (round(pf_d,2) if pf_d != float('inf') else 'inf'),
                'pf_canary': (round(pf_c,2) if pf_c != float('inf') else 'inf'),
                'delta_acc': round(acc_c - acc_d,3),
            })
        if not rows:
            return ''
        df = pd.DataFrame(rows)
        return '<h2>Canary vs Deployed (last 180d)</h2>' + df.to_html(index=False)
    except Exception:
        return ''


def _png_data_url(fig) -> str:
    import matplotlib.pyplot as plt  # ensured inside function scope
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return f"data:image/png;base64,{b64}"


def reliability_plot_section() -> str:
    """Calibration curve plot per strategy; returns an embedded PNG if matplotlib exists."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        scored = score_recent_dataset(lookback_days=180)
        if not scored:
            return ''
        fig, ax = plt.subplots(figsize=(6, 4))
        np.linspace(0.05, 0.95, 10)
        for strat, sdf in scored.items():
            p = sdf['proba'].astype(float).clip(0, 1)
            y = sdf['label'].astype(int)
            bins = np.linspace(0.0, 1.0, 11)
            idx = np.digitize(p, bins) - 1
            pred = []
            obs = []
            for b in range(10):
                mask = (idx == b)
                if mask.any():
                    pred.append(float(p[mask].mean()))
                    obs.append(float(y[mask].mean()))
            if pred and obs:
                ax.plot(pred, obs, marker='o', label=strat)
        ax.plot([0,1],[0,1],'k--',alpha=0.5)
        ax.set_title('Calibration (last 180d)')
        ax.set_xlabel('Predicted probability')
        ax.set_ylabel('Observed frequency')
        ax.legend()
        data_url = _png_data_url(fig)
        return f"<h2>Calibration Plot</h2><img src='{data_url}' alt='Calibration plot' />"
    except Exception:
        return ''


def top3_explainability_section(today_iso: str) -> str:
    """Show explainability and narratives for today's Top-3 and TOTD if present."""
    import pandas as pd
    picks_path = ROOT / 'logs' / 'daily_picks.csv'
    totd_path = ROOT / 'logs' / 'trade_of_day.csv'
    if not picks_path.exists():
        return ''
    try:
        picks = pd.read_csv(picks_path)
        if picks.empty:
            return ''
        # Limit to 3 rows
        picks = picks.head(3)
        # Collect needed symbols and dates
        syms = list({str(s) for s in picks['symbol'].astype(str).tolist()})
        # Fetch last 260d per symbol for features
        frames = []
        start = (pd.Timestamp(today_iso) - pd.Timedelta(days=260)).date().isoformat()
        for s in syms:
            df = fetch_daily_bars_polygon(s, start, today_iso, cache_dir=ROOT / 'data' / 'cache')
            if not df.empty:
                if 'symbol' not in df:
                    df = df.copy(); df['symbol'] = s
                frames.append(df)
        feats = compute_features_frame(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame()

        rows = ["<h2>Top-3 Explainability</h2>"]
        for _, r in picks.iterrows():
            sig = {k: (None if (isinstance(v,float) and pd.isna(v)) else v) for k,v in r.items()}
            strat_key = 'ibs_rsi' if str(r.get('strategy','')).lower().startswith(('ibs','rsi')) else 'turtle_soup'
            model = load_model(strat_key)
            # Match features row by date/symbol
            frow = None
            try:
                if not feats.empty and 'timestamp' in feats and 'symbol' in feats:
                    ts = pd.to_datetime(r.get('timestamp')).normalize()
                    frow = feats[(pd.to_datetime(feats['timestamp']).dt.normalize()==ts) & (feats['symbol'].astype(str)==str(r['symbol']))]
                    if not frow.empty:
                        frow = frow.tail(1).reindex(columns=FEATURE_COLS).iloc[0]
                    else:
                        frow = None
            except Exception:
                frow = None
            exp = explain_trade(sig, features_row=frow, model=model, cfg=ExplainConfig(top_k=5))
            nar = gen_narrative(sig)
            # Build HTML card
            rows.append('<div style="border:1px solid #ddd; padding:8px; margin:6px 0">')
            rows.append(f"<b>{exp['summary']}</b><br/>")
            if exp.get('narrative'):
                rows.append(f"<div>{exp['narrative']}</div>")
            if nar:
                tech = nar.get('technical',''); nar.get('casual',''); exe = nar.get('executive','')
                rows.append(f"<div><i>Technical:</i> {tech}</div>")
                rows.append(f"<div><i>Executive:</i> {exe}</div>")
            contrib = exp.get('contributions') or []
            if contrib:
                rows.append('<table style="border-collapse:collapse"><tr><th>Feature</th><th>Contribution</th></tr>')
                for c in contrib:
                    rows.append(f"<tr><td style='border:1px solid #ddd;padding:4px'>{c['feature']}</td><td style='border:1px solid #ddd;padding:4px'>{c['contribution']:+.6f}</td></tr>")
                rows.append('</table>')
            rows.append('</div>')

        # TOTD if available
        if totd_path.exists():
            try:
                t = pd.read_csv(totd_path).head(1)
                if not t.empty:
                    r = t.iloc[0]
                    sig = r.to_dict()
                    rows.append('<h2>Trade of the Day Explainability</h2>')
                    sname = str(r.get('strategy','')).lower()
                    strat_key = 'ibs_rsi' if sname.startswith('ibs') or sname.startswith('rsi') else 'turtle_soup'
                    model = load_model(strat_key)
                    frow = None
                    try:
                        if not feats.empty and 'timestamp' in feats and 'symbol' in feats:
                            ts = pd.to_datetime(r.get('timestamp')).normalize()
                            frow = feats[(pd.to_datetime(feats['timestamp']).dt.normalize()==ts) & (feats['symbol'].astype(str)==str(r['symbol']))]
                            if not frow.empty:
                                frow = frow.tail(1).reindex(columns=FEATURE_COLS).iloc[0]
                            else:
                                frow = None
                    except Exception:
                        frow = None
                    exp = explain_trade(sig, features_row=frow, model=model, cfg=ExplainConfig(top_k=5))
                    nar = gen_narrative(sig)
                    rows.append(f"<b>{exp['summary']}</b><br/>")
                    if exp.get('narrative'):
                        rows.append(f"<div>{exp['narrative']}</div>")
                    if nar:
                        tech = nar.get('technical',''); exe = nar.get('executive','')
                        rows.append(f"<div><i>Technical:</i> {tech}</div>")
                        rows.append(f"<div><i>Executive:</i> {exe}</div>")
            except Exception:
                pass

        return "\n".join(rows)
    except Exception as e:
        return f"<h2>Explainability</h2><p>Error: {e}</p>"


def feature_impact_section() -> str:
    """Show feature impact using SHAP if available; else coefficients."""
    # Try IBS+RSI model first (fallback to Turtle Soup)
    m = load_model('ibs_rsi')
    if m is None:
        m = load_model('turtle_soup')
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


# NOTE: Legacy main() removed. The report is generated by the unified main() below with CT display.


def _header_html() -> str:
    now = now_et()
    return (
        f"<h1>Kobe Morning Report - {now.strftime('%Y%m%d')}</h1>"
        f"<p><em>Display: {fmt_ct(now)} | {fmt_et(now)} (CT and ET, 12-hour). Internal scheduling/trading operates in ET.</em></p>"
    )


def build_html(day: str, prev_day: Optional[str]) -> str:
    parts = [
        '<html><head><meta charset="utf-8"><title>Kobe Morning Report</title>',
        '<style>body{font-family:Arial;margin:20px} table{border-collapse:collapse} th,td{border:1px solid #ddd;padding:6px} th{background:#f3f3f3}</style>',
        '</head><body>',
        _header_html(),
        sentiment_section(day, prev_day),
        spy_regime_and_anomalies((pd.Timestamp(day) - pd.Timedelta(days=540)).strftime('%Y-%m-%d'), day),
        model_section(),
        calibration_section(),
        reliability_plot_section(),
        threshold_sweep_section(),
        top3_explainability_section(day),
        canary_section(),
        '</body></html>'
    ]
    return '\n'.join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description='Generate Morning HTML report')
    ap.add_argument('--date', type=str, default=None, help='YYYY-MM-DD (defaults to today ET)')
    ap.add_argument('--outdir', type=str, default='reports')
    args = ap.parse_args()

    now = now_et()
    day = args.date or now.strftime('%Y-%m-%d')
    prev_day = (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"morning_report_{now.strftime('%Y%m%d')}.html"

    html = build_html(day, prev_day)
    out.write_text(html, encoding='utf-8')
    print('Morning report written:', out)


if __name__ == '__main__':
    main()


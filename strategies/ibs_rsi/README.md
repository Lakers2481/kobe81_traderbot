IBS+RSI Strategy — Replication Guide

Scope
- Strategy: IBS + RSI(2) Mean Reversion (long-only, daily)
- File: strategies/ibs_rsi/strategy.py (class IbsRsiStrategy, IbsRsiParams)

Rules (no lookahead)
- Compute IBS and RSI2 on the prior bar: indicators use .shift(1)
- Generate signal at close(t); fills occur at open(t+1)
- Trend filter: Close >= SMA(200) when enabled

Default parameters (IbsRsiParams) — v2.2
- ibs_max=0.08, rsi_max=5.0, atr_mult=2.0, r_multiple=2.0, time_stop_bars=7, min_price=15.0

Quick backtest (walk-forward, small slice; ensure train-days >= 200 for SMA200)
python scripts/run_wf_polygon.py --universe data/universe/optionable_liquid_900.csv --start 2024-01-02 --end 2024-12-26 --train-days 252 --test-days 63 --cap 30 --outdir wf_outputs_verify_2024_252x63 --fallback-free --dotenv ./.env

Metrics from WF
python scripts/metrics.py --wfdir wf_outputs_verify_quick --strategy IBS_RSI

Parameter tuning (compact grid)
python scripts/optimize.py --strategy ibs_rsi --universe data/universe/optionable_liquid_900.csv --start 2025-01-02 --end 2025-12-26 --cap 150 --outdir optimize_outputs --ibs-max 0.10,0.15,0.20 --rsi-max 5,10,15 --atr-mults 0.8,1.0,1.2 --r-mults 1.5,2.0,2.5 --time-stops 5,7 --dotenv ./.env

Notes
- Use STATUS.md “Replication Checklist (KEY)” for canonical steps.
- Always verify environment with: python scripts/status.py --json --dotenv ./.env

Turtle Soup Strategy (ICT Liquidity Sweep) — Replication Guide

Scope
- Strategy: Turtle Soup (failed breakout / liquidity sweep, long bias)
- File: strategies/ict/turtle_soup.py (class TurtleSoupStrategy, TurtleSoupParams)

Rules (no lookahead)
- Indicators and channel references use prior bars (.shift(1))
- Signal at close(t); fills at open(t+1)
- Trend filter via SMA(200) and min price

Default parameters (TurtleSoupParams) — v2.2
- lookback=20, min_bars_since_extreme=3, stop_buffer_mult=0.2, r_multiple=0.5, time_stop_bars=3, min_price=15.0

Quick backtest (walk-forward, small slice; use IBS SMA200 note for train-days)
python scripts/run_wf_polygon.py --universe data/universe/optionable_liquid_900.csv --start 2024-01-02 --end 2024-12-26 --train-days 252 --test-days 63 --cap 30 --outdir wf_outputs_verify_2024_252x63 --fallback-free --dotenv ./.env

Metrics from WF
python scripts/metrics.py --wfdir wf_outputs_verify_quick --strategy TURTLE_SOUP

Parameter tuning (compact grid)
python scripts/optimize.py --strategy turtle_soup --universe data/universe/optionable_liquid_900.csv --start 2025-01-02 --end 2025-12-26 --cap 150 --outdir optimize_outputs --ict-lookbacks 20,30 --ict-min-bars 3,5 --ict-stop-bufs 0.5,1.0 --ict-time-stops 5,7 --ict-r-mults 2.0,3.0 --dotenv ./.env

Notes
- Turtle Soup is rare; prefer longer windows / larger caps for KPI robustness.
- Use STATUS.md “Replication Checklist (KEY)” as the source of truth.

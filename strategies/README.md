Strategies — v2.2 Quick Guide

Scope
- Active set: IBS+RSI Mean Reversion and Turtle Soup (Liquidity Sweep), long-only, daily.
- Source of truth for performance and replication: docs/STATUS.md.

v2.2 Defaults (Quant Interview Ready)
- IBS+RSI (IbsRsiParams)
  - ibs_max=0.08, rsi_max=5.0, atr_mult=2.0, r_multiple=2.0, time_stop_bars=7, min_price=15.0
  - File: strategies/ibs_rsi/strategy.py
- Turtle Soup (TurtleSoupParams)
  - lookback=20, min_bars_since_extreme=3, stop_buffer_mult=0.2, r_multiple=0.5, time_stop_bars=3, min_price=15.0
  - WF filter: enforce min sweep strength ≥ 0.3 ATR via scripts/run_wf_polygon.py (arg: --turtle-soup-min-sweep)
  - File: strategies/ict/turtle_soup.py

Evidence (Backtest)
- File: reports/backtest_dual_latest.txt
  - IBS+RSI: 59.9% WR, 1.46 PF (867 trades)
  - Turtle Soup: 61.0% WR, 1.37 PF (305 trades)
  - Combined: 60.2% WR, 1.44 PF (1,172 trades)

Replicate Exactly
- Dual backtest (cap=200, ~5–6 min):
  - python scripts/backtest_dual_strategy.py --universe data/universe/optionable_liquid_900.csv --start 2015-01-01 --end 2024-12-31 --cap 200
- Walk-forward (ensure SMA200 history; train-days ≥ 200):
  - set KOBE_CONFIG_PATH=%CD%\config\base_backtest.yaml
  - python scripts/run_wf_polygon.py --universe data/universe/optionable_liquid_900.csv --start 2023-01-02 --end 2024-12-26 --train-days 252 --test-days 63 --cap 60 --outdir wf_outputs_verify_2023_2024 --fallback-free --dotenv ./.env
  - set KOBE_CONFIG_PATH=

Lookahead Discipline (both strategies)
- Indicators computed on prior bars via .shift(1).
- Signal at close(t); entry at open(t+1).

Where to Read More
- IBS+RSI details and quick WF: strategies/ibs_rsi/README.md
- Turtle Soup details and quick WF: strategies/ict/README.md
- Optimization rationale and grids: docs/V2.2_OPTIMIZATION_GUIDE.md
- System‑wide replication and artifacts: docs/STATUS.md


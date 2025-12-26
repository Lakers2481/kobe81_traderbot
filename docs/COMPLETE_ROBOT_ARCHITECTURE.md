# Kobe81 Traderbot — Complete Architecture Blueprint

This document maps the end‑to‑end trading blueprint to the Kobe codebase. It mirrors Layers 0–10, cross‑cutting concerns, and deployment pipeline, and points to concrete modules and scripts.

## Layer 0: External Data
- Vendors: Polygon (EOD OHLCV), Alpaca (broker + quotes)
- Config: `config/env_loader.py` (loads `.env`), `config/settings.json`

## Layer 1: Ingestion
- Data provider: `data/providers/polygon_eod.py`
  - `fetch_daily_bars_polygon()` — EOD 1D bars, cached to `data/cache`
  - Rate limiting (sleep), error handling

## Layer 2: Processing
- Cleaning/joins are simple for EOD bars; strategy indicators compute in strategy modules with lookahead prevention via `shift(1)`.

## Layer 3: Strategy Engine
- Canonical strategies:
  - RSI‑2 — `strategies/connors_rsi2/strategy.py`
  - IBS — `strategies/ibs/strategy.py`
- Combined filter:
  - AND logic in WF/backtest scripts by merging RSI‑2 and IBS signals
- Parameters: RSI‑2 (Wilder, long≤10, exit≥70, SMA200), IBS ((C−L)/(H−L), long<0.2, SMA200)

## Layer 4: Signal Generation
- `scan_signals_over_time()` (both strategies): generates all historical entries using shifted indicators (no lookahead), fills next bar open.

## Layer 5: Risk Management
- Policy Gate: `risk/policy_gate.py` — per‑order and daily budgets, price bounds, shorts toggle
- Kill switch: `state/KILL_SWITCH` file checked in `scripts/run_paper_trade.py`

## Layer 6: Portfolio Construction
- Sizing: fixed notional per trade (~$75 order budget) in backtester/live runners
- Rebalancing not required for single‑signal entries (v1)

## Layer 7: Order Management
- State: `oms/order_state.py`
- Idempotency: `oms/idempotency_store.py` (SQLite)
- Time in force: IOC limit enforced in broker adapter

## Layer 8: Execution
- Broker adapter: `execution/broker_alpaca.py`
  - Best ask fetch, IOC limit submission, idempotency, notes
  - Paper/live URLs via env

## Layer 9: Post‑Trade
- Reconciliation: `scripts/reconcile_alpaca.py` (snapshots orders/positions)
- Hash‑chain audit: `core/hash_chain.py`, `scripts/verify_hash_chain.py` — appends on submission
- Metrics: backtester writes `trade_list.csv`, `equity_curve.csv`, `summary.json`

## Layer 10: Learning & Monitoring
- Walk‑forward: `backtest/walk_forward.py`, `scripts/run_wf_polygon.py`
- KPI report: `scripts/aggregate_wf_report.py` (HTML)
- Health endpoints: `monitor/health_endpoints.py`, `scripts/start_health.py`

## Cross‑Cutting Concerns
- Config pinning: `core/config_pin.py`, `scripts/show_config_pin.py` (SHA256 of settings)
- Structured logs: `core/structured_log.py` (JSON lines under `logs/events.jsonl`)
- Kill switch: `state/KILL_SWITCH`
- Secrets: `.env` external to repo; loaded via `config/env_loader.py`
- Governance: PolicyGate budgets + veto, CLI scripts require human invocation

## Config-Gated Enhancements (config/base.yaml)
All features below are **disabled by default** to preserve existing behavior:

### 1. Commission/Fees Model (Backtest)
- Config: `backtest.commissions.enabled: true`
- Module: `backtest/engine.py`
- Reports gross_pnl, net_pnl, total_fees in summary.json

### 2. LULD/Volatility Clamp (Execution)
- Config: `execution.clamp.enabled: true`
- Module: `execution/broker_alpaca.py`
- Fixed % or ATR-based clamping to prevent price overshoot

### 3. Rate Limiter + Retry (Execution)
- Config: `execution.rate_limiter.enabled: true`
- Module: `core/rate_limiter.py`
- Token bucket (120/min), exponential backoff on 429

### 4. Earnings Proximity Filter (Signals)
- Config: `filters.earnings.enabled: true`
- Module: `core/earnings_filter.py`
- Skips signals 2 days before / 1 day after earnings

### 5. Metrics Endpoint (Monitoring)
- Config: `health.metrics.enabled: true` (default: true)
- Module: `monitor/health_endpoints.py`
- GET /metrics returns request counters and performance stats

### 6. Regime Filter (Signals)
- Config: `regime_filter.enabled: true`
- Module: `core/regime_filter.py`
- SPY-based trend gate: close > SMA(200), SMA(20) > SMA(200)
- Volatility gate: realized vol ≤ max_ann_vol threshold
- Filters signals to only trade in favorable market regimes

### 7. Signal Selection (Portfolio)
- Config: `selection.enabled: true`
- Module: Uses `config/settings_loader.py`
- Ranks signals by composite score (RSI-2, IBS, liquidity, vol penalty)
- Picks top_n signals per day (default: 10)

### 8. Volatility-Targeted Sizing (Portfolio)
- Config: `sizing.enabled: true`
- Module: `backtest/engine.py`
- Formula: qty = (risk_pct × equity) / (entry - stop)
- Default: 0.5% risk per trade (0.005)
- Falls back to fixed notional if no stop_loss provided

## Robustness Tools

### Parameter Optimization
- Script: `scripts/optimize.py`
- Grid search over RSI-2 and IBS parameters
- Outputs: heatmap CSV, HTML report, best_params.json
- Finds robust parameter plateaus (not single spikes)

### Monte Carlo Testing
- Script: `scripts/monte_carlo.py`
- Trade reordering and block bootstrap (1000 iterations)
- Computes robustness score (0-100) based on:
  - Sharpe stability
  - Drawdown stability
  - Win rate stability
  - Positive expectancy percentage
- Outputs: distributions CSV, robustness_score.json, HTML report

## Crypto Backtesting (Research-Only)
Crypto strategies mirror equities (RSI-2, IBS, AND) using hourly bars.

- Data Provider: `data/providers/polygon_crypto.py`
  - `fetch_crypto_bars()` for X:BTCUSD, X:ETHUSD, etc.
  - Hourly bars with caching under `data/cache/crypto/`
- Universe: `data/universe/crypto_top3.csv`, `crypto_top10.csv`
- Scripts: `scripts/run_wf_crypto.py`, `scripts/run_showdown_crypto.py`
- Outputs: Same structure as equities (wf_outputs_crypto/, showdown_outputs_crypto/)
- IBS guard: Flat bars (high==low) excluded to prevent division by zero
- No live crypto execution

## Deployment Pipeline
1. Shadow/Backtest: `scripts/smoke_test.py`, `scripts/run_backtest.py`
2. Paper Validation: `scripts/run_paper_trade.py` (micro)
3. Live Dry‑Run/Micro: `scripts/run_live_trade_micro.py` (micro)
4. Live Normal: extend budgets after acceptance

## Evidence & Commands
- Build 950 universe with proof:
  - `scripts/build_universe_polygon.py` → `data/universe/optionable_liquid_final.csv` and `.full.csv`
  - `scripts/check_polygon_earliest_universe.py` → `data/universe/earliest_latest_universe.csv`
  - `scripts/validate_universe_coverage.py` (assert ≥950 symbols and ≥10 years each)
- Prefetch data (faster WF): `scripts/prefetch_polygon_universe.py`
- Walk‑forward/Report: `scripts/run_wf_polygon.py`, `scripts/aggregate_wf_report.py`
- Preflight: `scripts/preflight.py` (env keys, config pin, broker probe)
- Health: `scripts/start_health.py`
- Audit verify: `scripts/verify_hash_chain.py`

## Notes
- Strategies are canonical; indicators shifted one bar; fills at next open; ATR(14)×2 stop and 5‑bar time stop.
- Default concurrency is conservative; adjust based on your Polygon plan.

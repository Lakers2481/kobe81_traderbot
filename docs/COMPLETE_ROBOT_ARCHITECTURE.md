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
- Secrets: `.env` external to repo; loaded via `configs/env_loader.py`
- Governance: PolicyGate budgets + veto, CLI scripts require human invocation

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

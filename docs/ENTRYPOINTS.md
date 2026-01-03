# ENTRYPOINTS.md - All Runnable Entry Points

> **Last Updated:** 2026-01-03
> **Total Entrypoints:** 180+

---

## Quick Reference

| Category | Primary Script | Command |
|----------|---------------|---------|
| **Daily Scan** | `scan.py` | `python scripts/scan.py --cap 900 --deterministic --top3` |
| **Paper Trade** | `run_paper_trade.py` | `python scripts/run_paper_trade.py --universe data/universe/optionable_liquid_900.csv --cap 50` |
| **Live Trade** | `run_live_trade_micro.py` | `python scripts/run_live_trade_micro.py --cap 10` |
| **24/7 Runner** | `runner.py` | `python scripts/runner.py --mode paper --scan-times 09:35,10:30,15:55` |
| **Backtest** | `backtest_dual_strategy.py` | `python scripts/backtest_dual_strategy.py --start 2023-01-01 --end 2024-12-31` |
| **Preflight** | `preflight.py` | `python scripts/preflight.py --dotenv ./.env` |

---

## 1. CORE EXECUTION

### Daily Scanner (CANONICAL)
```bash
python scripts/scan.py --cap 900 --deterministic --top3
```
- **File:** `scripts/scan.py`
- **What it does:** Scans 900-stock universe for IBS+RSI and Turtle Soup signals
- **Env vars:** `POLYGON_API_KEY`
- **Outputs:**
  - `logs/daily_picks.csv` - Top 3 picks
  - `logs/trade_of_day.csv` - Single TOTD
  - `logs/signals.jsonl` - All signals
- **Key args:** `--cap`, `--deterministic`, `--top3`, `--no-quality-gate`, `--preview`, `--min-conf`

### Paper Trading
```bash
python scripts/run_paper_trade.py --universe data/universe/optionable_liquid_900.csv --cap 50
```
- **File:** `scripts/run_paper_trade.py`
- **What it does:** Paper trades with dual position caps (2% risk, 20% notional)
- **Env vars:** `POLYGON_API_KEY`, `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY`
- **Outputs:** Orders to paper-api.alpaca.markets
- **Key args:** `--universe`, `--cap`, `--watchlist-only`, `--fallback-enabled`

### Live Trading (REAL MONEY)
```bash
python scripts/run_live_trade_micro.py --universe data/universe/optionable_liquid_900.csv --cap 10
```
- **File:** `scripts/run_live_trade_micro.py`
- **What it does:** Live execution via Alpaca (IOC LIMIT)
- **Env vars:** `ALPACA_BASE_URL` (production), API keys
- **DANGER:** Uses REAL MONEY - requires live endpoint

---

## 2. SCHEDULERS & RUNNERS

### 24/7 Runner
```bash
python scripts/runner.py --mode paper --universe data/universe/optionable_liquid_900.csv --cap 50 --scan-times 09:35,10:30,15:55
```
- **File:** `scripts/runner.py`
- **What it does:** Runs paper/live trading at configurable times
- **Features:** Single-instance enforcement, heartbeat, kill switch integration
- **Kill switch:** `state/KILL_SWITCH`
- **Key args:** `--mode` (paper/live), `--scan-times`, `--approve-live`

### Autonomous Brain
```bash
python scripts/run_autonomous.py --cycle 60
```
- **File:** `scripts/run_autonomous.py`
- **What it does:** Continuous learning, research, backtesting
- **Key args:** `--cycle`, `--status`, `--once`, `--daemon`

### Master Scheduler
```bash
python scripts/scheduler_kobe.py
```
- **File:** `scripts/scheduler_kobe.py`
- **Schedule (ET):**
  - 5:30 - DB_BACKUP
  - 6:00 - DATA_UPDATE
  - 8:00 - PRE_GAME
  - 8:15 - PREGAME_BLUEPRINT
  - 9:45 - FIRST_SCAN
  - 10:00-16:00 - POSITION_MANAGER (every 15 min)

### Job Runner (Windows Task Scheduler)
```bash
python scripts/run_job.py --tag DB_BACKUP --dotenv ./.env
```
- **File:** `scripts/run_job.py`
- **Tags:** DB_BACKUP, DATA_UPDATE, MORNING_REPORT, PRE_GAME, FIRST_SCAN, HALF_TIME, etc.

---

## 3. PROFESSIONAL EXECUTION FLOW

### Overnight Watchlist (3:30 PM)
```bash
python scripts/overnight_watchlist.py --cap 900 --prefetch
```
- **Output:** `state/watchlist/next_day.json`

### Premarket Validator (8:00 AM)
```bash
python scripts/premarket_validator.py --gap-threshold 0.03
```
- **Output:** `state/watchlist/today_validated.json`
- **Flags:** VALID, GAP_INVALIDATED, NEWS_RISK

### Opening Range Observer (9:30-10:00)
```bash
python scripts/opening_range_observer.py
```
- **Output:** `state/watchlist/opening_range.json`
- **Note:** NO TRADES - observation only

### Pre-Game Blueprint (8:15 AM)
```bash
python scripts/generate_pregame_blueprint.py --cap 900 --top 5 --execute 2
```
- **Outputs:**
  - `reports/pregame_YYYYMMDD.json`
  - `reports/pregame_YYYYMMDD.md`

---

## 4. BACKTESTING & VALIDATION

### Canonical Backtest
```bash
python scripts/backtest_dual_strategy.py --universe data/universe/optionable_liquid_900.csv --start 2023-01-01 --end 2024-12-31 --cap 150
```
- **Expected:** ~64% WR, ~1.60 PF
- **File:** `scripts/backtest_dual_strategy.py`

### Walk-Forward Backtest
```bash
python scripts/run_wf_polygon.py --universe data/universe/optionable_liquid_900.csv --start 2015-01-01 --end 2024-12-31 --train-days 252 --test-days 63
```
- **Output:** `wf_outputs/<strategy>/split_NN/`

### WF Report Aggregation
```bash
python scripts/aggregate_wf_report.py --wfdir wf_outputs
```

### Other Backtests
```bash
python scripts/backtest_ibs_rsi.py --start 2023-01-01 --end 2024-12-31
python scripts/backtest_momentum_dip.py --start 2023-01-01 --end 2024-12-31
python scripts/backtest_totd.py --start 2023-01-01 --end 2024-12-31
```

---

## 5. STARTUP & SHUTDOWN

### Start System
```bash
python scripts/start.py --mode paper
```
- **Checks:** Kill switch, duplicate processes, preflight

### Stop System
```bash
python scripts/stop.py
```

### Restart System
```bash
python scripts/restart.py --mode paper
```

### Preflight Checks
```bash
python scripts/preflight.py --dotenv ./.env
```
- **Validates:** Env vars, config pin, Alpaca/Polygon connectivity

---

## 6. EMERGENCY CONTROLS

### Kill Switch (HALT ALL TRADING)
```bash
python scripts/kill.py --reason "Manual intervention required"
```
- **Creates:** `state/KILL_SWITCH`
- **Effect:** Immediate order submission halt

### Resume from Kill Switch
```bash
python scripts/resume.py --confirm
```

---

## 7. WEB DASHBOARDS

### Trading Dashboard
```bash
python scripts/dashboard.py --start --port 8080
```
- **Endpoints:** `/`, `/api/status`, `/api/positions`, `/api/pnl`, `/api/health`

### Quant Dashboard
```bash
python scripts/quant_dashboard.py
```

### Health Endpoints
- **File:** `monitor/health_endpoints.py`
- **Port:** 5000

---

## 8. TELEGRAM

### Telegram Test
```bash
python scripts/send_telegram_test.py
```
- **Env vars:** `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`

---

## 9. DATA MANAGEMENT

### Build Universe
```bash
python scripts/build_universe_polygon.py --cidates data/universe/optionable_liquid_cidates.csv --start 2015-01-01 --end 2024-12-31 --min-years 10 --cap 900 --concurrency 3
```
- **Output:** `data/universe/optionable_liquid_900.csv`

### Prefetch EOD Data
```bash
python scripts/prefetch_polygon_universe.py --universe data/universe/optionable_liquid_900.csv --start 2015-01-01 --end 2024-12-31
```

### Freeze Data Lake
```bash
python scripts/freeze_equities_eod.py --universe data/universe/optionable_liquid_900.csv --start 2015-01-01 --end 2025-12-31 --provider stooq
python scripts/freeze_crypto_ohlcv.py --symbols BTCUSDT,ETHUSDT --start 2020-01-01 --end 2025-12-31
```

### Validate Lake
```bash
python scripts/validate_lake.py --dataset-id YOUR_DATASET_ID
```

---

## 10. TRAINING & ML

### Train LSTM
```bash
python scripts/train_lstm_confidence.py
```

### Train HMM
```bash
python scripts/train_hmm_regime.py
```

### Train Ensemble
```bash
python scripts/train_ensemble.py
```

### Train RL Agent
```bash
python scripts/train_rl_agent.py
```

### Generate Training Data
```bash
python scripts/generate_training_data.py
```

---

## 11. DIAGNOSTICS

### Integrity Check
```bash
python scripts/integrity_check.py
```

### Verify Hash Chain
```bash
python scripts/verify_hash_chain.py
```

### Debug Signals
```bash
python scripts/debug_signals.py
```

### Smoke Test
```bash
python scripts/smoke_test.py
```

### Quality Check
```bash
python scripts/quality_check.py
```

---

## 12. STATUS & REPORTING

### System Status
```bash
python scripts/status.py
```

### Positions
```bash
python scripts/positions.py
```

### P&L Summary
```bash
python scripts/pnl.py
```

### Orders
```bash
python scripts/orders.py
```

### Signals
```bash
python scripts/signals.py
```

### Logs
```bash
python scripts/logs.py
```

---

## 13. MAINTENANCE

### Backup State
```bash
python scripts/backup_state.py
```

### Snapshot
```bash
python scripts/snapshot.py
```

### Cleanup
```bash
python scripts/cleanup.py
```

### Reconcile Alpaca
```bash
python scripts/reconcile_alpaca.py
```

---

## Environment Variables Required

| Variable | Required For | Source |
|----------|-------------|--------|
| `POLYGON_API_KEY` | Data fetching | Polygon.io |
| `ALPACA_API_KEY_ID` | Broker | Alpaca |
| `ALPACA_API_SECRET_KEY` | Broker | Alpaca |
| `ALPACA_BASE_URL` | Paper/Live switch | Alpaca |
| `TELEGRAM_BOT_TOKEN` | Alerts | Telegram |
| `TELEGRAM_CHAT_ID` | Alerts | Telegram |

---

## Related Documentation

- [REPO_MAP.md](REPO_MAP.md) - Directory structure
- [ARCHITECTURE.md](ARCHITECTURE.md) - Pipeline wiring
- [JOBS_AND_SCHEDULER.md](JOBS_AND_SCHEDULER.md) - Job schedules

# Kobe Trading System - Entrypoints Audit

**Generated:** 2026-01-05
**Total Entrypoints:** 193
**Files with `__main__` blocks:** 286

---

## Summary Statistics

| Category | Count |
|----------|-------|
| Core Trading | 12 |
| Backtesting | 8 |
| Scanning | 6 |
| Web Apps | 4 |
| Schedulers/Daemons | 8 |
| Autonomous Brain | 7 |
| ML Training | 10 |
| Data Management | 12 |
| Risk Management | 6 |
| Monitoring | 8 |
| Utilities | 45 |
| Testing | 15 |
| Tools | 12 |
| Guardian | 6 |
| Portfolio | 4 |
| Analytics | 6 |
| Cognitive | 4 |
| Docker Services | 4 |
| Other | 16 |

---

## Critical Entrypoints (MUST KNOW)

These are the most important scripts for operating the trading system:

| Script | Purpose | Mode |
|--------|---------|------|
| `scripts/runner.py` | 24/7 multi-asset scheduled trading | paper/live |
| `scripts/run_paper_trade.py` | Paper trading with IOC LIMIT orders | paper |
| `scripts/run_live_trade_micro.py` | Live trading (REAL MONEY) | live |
| `scripts/scan.py` | Daily stock scanner (DualStrategyScanner) | paper/live |
| `scripts/kill.py` | Emergency halt - creates KILL_SWITCH | paper/live |
| `scripts/preflight.py` | 10 critical checks before trading | paper/live |
| `scripts/backtest_dual_strategy.py` | Canonical backtest for strategies | backtest |
| `scripts/scheduler_kobe.py` | Master scheduler with 80+ tasks | paper/live |

---

## Core Trading (12 scripts)

| Script | CLI | Description |
|--------|-----|-------------|
| `scripts/runner.py` | argparse | 24/7 runner with heartbeat, kill zones, drift detection |
| `scripts/run_paper_trade.py` | argparse | Paper trading with cognitive brain |
| `scripts/run_live_trade_micro.py` | argparse | Live micro trading (real money) |
| `scripts/start.py` | argparse | System startup with preflight checks |
| `scripts/stop.py` | argparse | Graceful shutdown |
| `scripts/kill.py` | argparse | Emergency halt - creates KILL_SWITCH |
| `scripts/resume.py` | argparse | Deactivate kill switch |
| `scripts/restart.py` | argparse | Clean restart |
| `scripts/positions.py` | argparse | Show open positions with P&L |
| `scripts/orders.py` | argparse | Order history and fills |
| `scripts/pnl.py` | argparse | P&L summary |
| `scripts/exit_manager.py` | argparse | Manage position exits |

---

## Scanning (6 scripts)

| Script | CLI | Description |
|--------|-----|-------------|
| `scripts/scan.py` | argparse | Daily scanner (IBS+RSI + Turtle Soup) |
| `scripts/overnight_watchlist.py` | argparse | Build Top 5 watchlist (3:30 PM) |
| `scripts/premarket_validator.py` | argparse | Validate gaps/news (8:00 AM) |
| `scripts/opening_range_observer.py` | argparse | Observe 9:30-10:00 (NO TRADES) |
| `scripts/watchlist.py` | argparse | Manage custom watchlists |
| `scripts/signals.py` | argparse | View raw generated signals |

---

## Backtesting (8 scripts)

| Script | CLI | Description |
|--------|-----|-------------|
| `scripts/backtest_dual_strategy.py` | argparse | Canonical backtest for strategies |
| `scripts/run_wf_polygon.py` | argparse | Walk-forward backtest |
| `scripts/run_backtest.py` | argparse | Generic backtest runner |
| `scripts/run_backtest_polygon.py` | argparse | Backtest with Polygon data |
| `scripts/run_backtest_options_synth.py` | argparse | Synthetic options backtest |
| `scripts/run_showdown_polygon.py` | argparse | Strategy comparison |
| `scripts/run_showdown_crypto.py` | argparse | Crypto strategy comparison |
| `scripts/mini_backtest.py` | argparse | Quick mini backtest |

---

## Web Applications (4 apps)

### FastAPI Applications
| File | Run Command | Description |
|------|-------------|-------------|
| `web/main.py` | `uvicorn web.main:app --port 8000` | Main dashboard with API |
| `web/dashboard_pro.py` | `uvicorn web.dashboard_pro:app --port 8080` | Bloomberg-style dashboard |

### Streamlit Applications
| File | Run Command | Description |
|------|-------------|-------------|
| `web/dashboard.py` | `streamlit run web/dashboard.py` | Trading status dashboard |
| `scripts/quant_dashboard.py` | `streamlit run scripts/quant_dashboard.py` | Quant analysis dashboard |

---

## Schedulers & Daemons (8 scripts)

| Script | Loop Type | Description |
|--------|-----------|-------------|
| `scripts/scheduler_kobe.py` | schedule-based | Master scheduler v2.0 (80+ tasks) |
| `scripts/run_autonomous.py` | while loop | 24/7 autonomous brain |
| `scripts/overnight_runner.py` | once | Overnight task runner |
| `scripts/supervisor.py` | daemon | Process supervisor |
| `scripts/watchdog.py` | daemon | System watchdog |
| `scripts/heartbeat.py` | daemon | Heartbeat tracking |
| `scripts/scheduler_ctl.py` | CLI | Scheduler control |
| `scripts/run_daily_pipeline.py` | once | Daily pipeline |

---

## Autonomous Brain (7 modules)

| File | Description |
|------|-------------|
| `autonomous/brain.py` | Core orchestrator (24/7 self-aware) |
| `autonomous/run.py` | Alternative runner |
| `autonomous/master_brain.py` | Master brain |
| `autonomous/master_brain_full.py` | Full capabilities |
| `autonomous/comprehensive_brain.py` | Comprehensive implementation |
| `autonomous/scheduler.py` | Task scheduler |
| `autonomous/__main__.py` | Module entry (python -m autonomous) |

---

## ML Training (10 scripts)

| Script | Description |
|--------|-------------|
| `scripts/train_hmm_regime.py` | Train HMM regime detector |
| `scripts/train_lstm_confidence.py` | Train LSTM confidence model |
| `scripts/train_ensemble_models.py` | Train XGBoost, LightGBM |
| `scripts/train_ensemble.py` | Train ensemble predictor |
| `scripts/train_rl_agent.py` | Train RL agent (PPO/DQN/A2C) |
| `scripts/train_meta.py` | Train meta model |
| `scripts/run_weekly_training.py` | Weekly ML retraining |
| `scripts/generate_training_data.py` | Generate training data |
| `scripts/promote_models.py` | Promote models to production |
| `scripts/feature_experiment.py` | Feature engineering experiments |

---

## Data Management (12 scripts)

| Script | Description |
|--------|-------------|
| `scripts/build_universe_polygon.py` | Build 900-stock universe |
| `scripts/build_universe_900.py` | Build optimized universe |
| `scripts/prefetch_polygon_universe.py` | Prefetch EOD bars |
| `scripts/freeze_equities_eod.py` | Freeze equities from Stooq |
| `scripts/freeze_crypto_ohlcv.py` | Freeze crypto from Binance |
| `scripts/backfill_yfinance.py` | Backfill from Yahoo Finance |
| `scripts/validate_lake.py` | Validate data lake |
| `scripts/validate_universe_coverage.py` | Validate universe |
| `scripts/check_data_quality.py` | Check data quality |
| `scripts/universe.py` | Manage universe |
| `scripts/data.py` | Data fetch status |
| `scripts/polygon.py` | Validate Polygon data |

---

## Risk Management (6 scripts)

| Script | Description |
|--------|-------------|
| `scripts/risk_cli.py` | Risk management CLI |
| `scripts/exposure.py` | Sector/factor exposure |
| `scripts/correlation.py` | Position correlation matrix |
| `scripts/correlation_check.py` | Check correlation limits |
| `scripts/drawdown.py` | Drawdown analysis |
| `scripts/monte_carlo.py` | Monte Carlo simulation |

---

## Monitoring (8 scripts)

| Script | Description |
|--------|-------------|
| `scripts/preflight.py` | 10 critical checks |
| `scripts/status.py` | System health dashboard |
| `scripts/health_monitor.py` | Health monitoring |
| `scripts/start_health.py` | Start health server |
| `scripts/logs.py` | View recent events |
| `scripts/alerts.py` | Manage alerts |
| `scripts/telegram.py` | Telegram notifications |
| `scripts/performance.py` | Performance monitoring |

---

## Docker Services

### docker-compose.yml Services

| Service | Command | Description |
|---------|---------|-------------|
| `kobe-paper` | `python scripts/runner.py --mode paper` | 24/7 paper trading |
| `kobe-scanner` | `python scripts/scan.py --top3` | Daily scanner (profile: scanner) |
| `kobe-preflight` | `python scripts/preflight.py` | Preflight check (profile: utils) |
| `kobe-verify` | `python scripts/verify_system.py` | System verification (profile: utils) |

### Dockerfile Default Command
```bash
python scripts/runner.py --mode paper --cap 50
```

---

## Guardian System (6 scripts)

| Script | Description |
|--------|-------------|
| `guardian/guardian.py` | System guardian (watchdog) |
| `guardian/system_monitor.py` | System monitoring |
| `guardian/alert_manager.py` | Alert management |
| `guardian/decision_engine.py` | Automated decisions |
| `guardian/self_learner.py` | Self-learning module |
| `guardian/daily_digest.py` | Daily digest |

---

## Testing (15 scripts)

| Script | Description |
|--------|-------------|
| `scripts/test.py` | Run unit/integration tests |
| `scripts/validate.py` | Tests + type checks |
| `scripts/smoke_test.py` | Smoke tests |
| `scripts/integrity_check.py` | Detect lookahead, bias, bugs |
| `scripts/quality_check.py` | Quality checks with scoring |
| `scripts/verify_system.py` | Full system verification |
| `scripts/verify_architecture.py` | Verify architecture |
| `scripts/verify_scan_consistency.py` | Verify scanner determinism |
| `scripts/validate_data_pipeline.py` | Validate data pipeline |
| `scripts/readiness_check.py` | Production readiness |
| `scripts/interview_quick_test.py` | Quick test for interviews |
| `testing/stress_test.py` | Stress testing |
| `testing/monte_carlo.py` | Monte Carlo testing |

---

## Tools (12 scripts)

| Script | Description |
|--------|-------------|
| `tools/verify_repo.py` | Repository verification |
| `tools/verify_robot.py` | Robot functionality |
| `tools/verify_alive.py` | Liveness check |
| `tools/verify_100_components.py` | Component verification |
| `tools/cleanup_cache.py` | Clean up cache |
| `tools/bounce_profile.py` | Build bounce profiles |
| `tools/build_bounce_db.py` | Build bounce database |
| `tools/build_bounce_profiles_all.py` | Build all profiles |
| `tools/today_bounce_watchlist.py` | Today's bounce watchlist |
| `tools/super_audit_verifier.py` | Super audit |

---

## CLI Framework Usage

| Framework | Count |
|-----------|-------|
| argparse | 173 |
| FastAPI | 2 |
| Streamlit | 2 |
| typer | 0 |
| click | 0 |

---

## Quick Reference - Common Commands

### Start Trading
```bash
# Paper trading (single run)
python scripts/run_paper_trade.py --universe data/universe/optionable_liquid_800.csv --start 2024-01-01 --end 2024-12-31 --cap 50

# 24/7 runner (paper)
python scripts/runner.py --mode paper --universe data/universe/optionable_liquid_800.csv --cap 50 --scan-times 09:35,10:30,15:55

# Start system (with preflight)
python scripts/start.py --mode paper
```

### Scanning
```bash
# Daily scan with top 3 picks
python scripts/scan.py --cap 900 --deterministic --top3

# Build overnight watchlist
python scripts/overnight_watchlist.py
```

### Backtesting
```bash
# Canonical backtest
python scripts/backtest_dual_strategy.py --universe data/universe/optionable_liquid_800.csv --start 2023-01-01 --end 2024-12-31 --cap 150

# Walk-forward
python scripts/run_wf_polygon.py --universe data/universe/optionable_liquid_800.csv --start 2015-01-01 --end 2024-12-31 --train-days 252 --test-days 63
```

### Emergency Controls
```bash
# Emergency halt
python scripts/kill.py

# Resume after halt
python scripts/resume.py --confirm
```

### Monitoring
```bash
# System status
python scripts/status.py

# Preflight checks
python scripts/preflight.py --dotenv ./.env

# View positions
python scripts/positions.py
```

### Docker
```bash
# Start paper trading
docker-compose up kobe-paper

# Run scanner
docker-compose --profile scanner up kobe-scanner

# Run preflight
docker-compose --profile utils run kobe-preflight
```

---

## Notes

1. **286 files have `if __name__ == "__main__"` blocks** - many are tests or modules with demo code
2. **193 unique runnable entrypoints** documented here are the primary scripts
3. **All scripts use argparse** for CLI parsing (no click/typer)
4. **FastAPI** powers the web dashboard and API
5. **Streamlit** provides additional dashboard visualizations
6. **Docker** support via Dockerfile and docker-compose.yml
7. **No systemd service files** found (Windows platform)

---

*See `01_ENTRYPOINTS.json` for full structured data.*

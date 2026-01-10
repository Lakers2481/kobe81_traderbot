# Kobe Trading System - Operations Runbook

> **Version:** 2.4
> **Last Updated:** 2026-01-05
> **Status:** Production Ready

---

## Table of Contents

1. [Pre-Market Checklist](#pre-market-checklist)
2. [Daily Operations](#daily-operations)
3. [Emergency Procedures](#emergency-procedures)
4. [Configuration Toggles](#configuration-toggles)
5. [Monitoring & Metrics](#monitoring--metrics)
6. [Troubleshooting](#troubleshooting)
7. [Maintenance Procedures](#maintenance-procedures)

---

## Pre-Market Checklist

Run these checks **before market open** (8:30 AM ET):

### 0. Live Trading Preflight (REQUIRED for LIVE) - NEW 2026-01-05

```bash
python scripts/preflight_live.py --mode live
```

**ALL checks must PASS.** If ANY blocking check fails, trading is halted.

**15+ Checks Include:**
- Settings schema validation (Pydantic)
- Broker API keys present
- Broker connectivity
- Market calendar
- Kill switch inactive
- LLM budget available
- Position reconciliation
- Data freshness
- Config pin match
- And more...

### 1. Cognitive Preflight (Required)

```bash
python scripts/preflight.py --cognitive
```

**Expected Output:**
- Episodic Memory: OK (< max episodes, no corruption)
- Self Model: OK (calibration within bounds)
- Reflection Engine: OK (learnings not overfitting)
- Ensemble Health: OK (all models loaded)
- Kill Switch: OK (file does not exist)
- API Connectivity: OK (Polygon, Alpaca responding)

### 2. Kill Switch Check

```bash
# Windows
if exist state\KILL_SWITCH (echo BLOCKED - KILL SWITCH ACTIVE) else (echo OK - No kill switch)

# Linux/Mac
ls state/KILL_SWITCH 2>/dev/null && echo "BLOCKED" || echo "OK"
```

**If KILL_SWITCH exists:** Do NOT proceed. See [Emergency Procedures](#emergency-procedures).

### 3. API Key Verification

```bash
python scripts/preflight.py --apis
```

Verifies:
- `POLYGON_API_KEY` - Data feed
- `ALPACA_API_KEY_ID` - Broker auth
- `ALPACA_API_SECRET_KEY` - Broker auth
- `ANTHROPIC_API_KEY` - LLM (optional)

### 4. VIX Level Check

```bash
python -c "from data.providers.polygon_eod import get_vix_level; print(f'VIX: {get_vix_level()}')"
```

| VIX Level | Action |
|-----------|--------|
| < 20 | Normal operations |
| 20-25 | Monitor closely |
| 25-35 | Reduce position sizes by 50% |
| > 35 | Consider halting new entries |

### 5. Data Freshness Check

```bash
python scripts/preflight.py --data
```

Verifies:
- Universe file exists and is recent
- Cache files not stale (< 24 hours)
- Provider endpoints responding

### 6. Position Reconciliation

```bash
python scripts/reconcile_alpaca.py
```

Compares local `state/positions.json` with Alpaca broker positions.

---

## Daily Operations

### Scan Schedule

| Time (ET) | Scan Type | Command |
|-----------|-----------|---------|
| 09:35 | Morning Scan | `python scripts/scan.py --cap 900 --deterministic --top3` |
| 10:30 | Mid-Morning | `python scripts/scan.py --cap 900 --deterministic --top3` |
| 15:55 | EOD Scan | `python scripts/scan.py --cap 900 --deterministic --top3` |

### Running the 24/7 Scheduler

```bash
# Paper trading mode
python scripts/runner.py --mode paper --universe data/universe/optionable_liquid_800.csv --cap 50 --scan-times 09:35,10:30,15:55

# Live trading mode (REAL MONEY)
python scripts/runner.py --mode live --universe data/universe/optionable_liquid_800.csv --cap 10 --scan-times 09:35,10:30,15:55
```

### Monitoring Endpoints

| Endpoint | URL | Purpose |
|----------|-----|---------|
| Health | `http://localhost:8080/health` | System liveness |
| Metrics | `http://localhost:8080/metrics` | All metrics dashboard |
| Prometheus | `http://localhost:8080/metrics/prometheus` | Prometheus format metrics |
| Ready | `http://localhost:8080/ready` | Readiness check |

**Rate Limiting (NEW 2026-01-05):**
- `/metrics` and `/metrics/prometheus` are rate-limited in live mode
- Limit: 60 requests/minute
- Exceeding limit returns HTTP 429 with `Retry-After` header
- Rate limiting disabled in paper mode for easier development

### Log Locations

| Log | Path | Purpose |
|-----|------|---------|
| Events | `logs/events.jsonl` | Structured event log |
| Cognitive | `logs/cognitive_decisions.jsonl` | Decision chain of thought |
| Trades | `logs/trades.jsonl` | Trade execution log |
| Errors | `logs/errors.jsonl` | Error log |
| Daily Picks | `logs/daily_picks.csv` | Scanner output |

---

## Emergency Procedures

### IMMEDIATE HALT - Create Kill Switch

```bash
# Windows
echo 1 > state\KILL_SWITCH

# Linux/Mac
echo 1 > state/KILL_SWITCH
```

**Effect:** All order submissions blocked immediately. Existing positions remain.

### Cancel All Open Orders

```bash
python scripts/cancel_all.py
```

### Close All Positions (Emergency Liquidation)

```bash
python scripts/close_all_positions.py --confirm
```

**WARNING:** This sells all positions at market. Use only in emergencies.

### Reconcile After Emergency

```bash
# 1. Verify positions match broker
python scripts/reconcile_alpaca.py

# 2. Verify hash chain integrity
python scripts/verify_hash_chain.py

# 3. Check for orphaned orders
python scripts/check_orphaned_orders.py
```

### Resume After Emergency

```bash
# 1. Verify all checks pass
python scripts/preflight.py --all

# 2. Remove kill switch (only if safe)
# Windows
del state\KILL_SWITCH

# Linux/Mac
rm state/KILL_SWITCH

# 3. Verify removal
python scripts/preflight.py --cognitive
```

---

## Configuration Toggles

All toggles in `config/base.yaml`:

### LLM Configuration

| Toggle | Path | Default | Effect |
|--------|------|---------|--------|
| LLM Enabled | `cognitive.llm_analyzer.enabled` | `true` | Master switch for Claude calls |
| LLM Mode | `cognitive.llm_analyzer.mode` | `briefings_only` | `off` / `briefings_only` / `full` |
| Cost Tracking | `cognitive.llm_analyzer.cost_tracking.enabled` | `true` | Track USD costs |
| Max Daily USD | `cognitive.llm_analyzer.cost_tracking.max_daily_usd` | `50.0` | Hard limit on LLM spend |
| Alert Threshold | `cognitive.llm_analyzer.cost_tracking.alert_threshold_usd` | `25.0` | Warn at this level |
| Cache Enabled | `cognitive.llm_analyzer.cache.enabled` | `true` | Enable response caching |
| Cache TTL | `cognitive.llm_analyzer.cache.ttl_hours` | `24` | Cache expiry |

### Selection Mode

| Toggle | Path | Default | Effect |
|--------|------|---------|--------|
| Selection Mode | `selection.mode` | `totd` | `totd` (single) / `top_n` (multiple) |
| Top-N Enabled | `selection.top_n.enabled` | `false` | Enable multi-position mode |
| Max Positions | `selection.top_n.n` | `3` | Max positions per day |
| Max Correlation | `selection.top_n.max_correlation` | `0.70` | Correlation limit |
| Max Sector % | `selection.top_n.max_sector_pct` | `0.40` | Sector concentration limit |
| Max Single Name | `selection.top_n.max_single_name_pct` | `0.25` | Single position limit |

### Risk Limits

| Toggle | Path | Default | Effect |
|--------|------|---------|--------|
| Per-Order Max | `risk.policy_gate.per_order_max_usd` | `75` | Max $ per order |
| Daily Budget | `risk.policy_gate.daily_budget_usd` | `1000` | Max $ per day |
| Position Limit | `risk.position_limit` | `10` | Max concurrent positions |
| VaR Limit | `risk.advanced.var_limit_pct` | `0.02` | 2% portfolio VaR limit |

### Quality Gate

| Toggle | Path | Default | Effect |
|--------|------|---------|--------|
| Min Score | `quality_gate.min_score` | `55` | Minimum signal score (raise to 70 when ML trained) |
| Max Signals | `quality_gate.max_signals_per_day` | `3` | Max signals to accept |

### Cognitive Safety

| Toggle | Path | Default | Effect |
|--------|------|---------|--------|
| Preflight Required | `cognitive.preflight.required` | `true` | Require preflight before trading |
| Circuit Breakers | `cognitive.circuit_breakers.enabled` | `true` | Enable failure mode detection |
| Calibration Bounds | `cognitive.circuit_breakers.calibration_bounds` | `[0.05, 0.95]` | Self-model drift limits |
| Max Loss Streak | `cognitive.circuit_breakers.max_loss_streak` | `5` | Halt after N consecutive losses |

---

## Monitoring & Metrics

### Key Metrics to Watch

Access via `http://localhost:8080/metrics`:

```json
{
  "llm": {
    "cost_usd_today": 12.50,
    "cost_budget_usd": 50.00,
    "calls_today": 25,
    "avg_latency_ms": 450
  },
  "execution": {
    "ioc_fill_rate_avg": 0.85,
    "slippage_bps_avg": 3.2,
    "clamp_usage_count": 12
  },
  "data": {
    "provider_success_rates": {
      "polygon": 0.99,
      "yfinance": 0.95,
      "stooq": 0.92
    },
    "fallback_count_today": 5
  },
  "cognitive": {
    "decisions_today": 15,
    "approval_rate": 0.73,
    "avg_confidence": 0.68
  }
}
```

### Alert Thresholds

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| LLM Cost USD | > $25/day | > $45/day | Reduce LLM mode |
| IOC Fill Rate | < 80% | < 60% | Increase clamp offset |
| Provider Success | < 95% | < 80% | Check API status |
| Avg Slippage | > 5 bps | > 10 bps | Widen limit offset |
| Cognitive Approval | < 60% | < 40% | Review signal quality |

### Health Check Script

```bash
# Quick health check
curl http://localhost:8080/health

# Full metrics dump
curl http://localhost:8080/metrics | python -m json.tool
```

---

## Troubleshooting

### Common Issues

#### 1. "KILL_SWITCH active" on startup

```bash
# Check why it was created
type state\KILL_SWITCH  # Windows
cat state/KILL_SWITCH   # Linux/Mac

# If safe to remove
del state\KILL_SWITCH   # Windows
rm state/KILL_SWITCH    # Linux/Mac

# Re-run preflight
python scripts/preflight.py --cognitive
```

#### 2. "ML model returning 0.5"

Models not trained. Either:
- Run training: `python scripts/train_ensemble.py`
- Or lower quality gate: Set `quality_gate.min_score: 55` in config

#### 3. "Data provider timeout"

```bash
# Check provider health
python -c "from data.providers.multi_source import get_provider_stats; print(get_provider_stats())"

# Force cache clear
del cache\polygon\*.csv  # Windows
rm cache/polygon/*.csv   # Linux/Mac
```

#### 4. "Position mismatch with broker"

```bash
# Reconcile
python scripts/reconcile_alpaca.py

# If still mismatched, force sync
python scripts/reconcile_alpaca.py --force-sync
```

#### 5. "LLM budget exceeded"

```bash
# Check current usage (NEW 2026-01-05 - uses token_budget module)
python -c "from llm.token_budget import get_token_budget; print(get_token_budget().get_status())"

# Check remaining budget percentage
python -c "from llm.token_budget import get_token_budget; print(f'{get_token_budget().get_remaining_percent():.1f}% remaining')"

# Budget auto-resets at midnight
# To manually reset (testing only):
python -c "from llm.token_budget import get_token_budget; b=get_token_budget(); b.used_today=0; b.cost_usd_today=0; b._save_state()"

# Disable LLM temporarily if needed
# Edit config/base.yaml: cognitive.llm_analyzer.enabled: false
```

**LLM Budget Enforcement (NEW 2026-01-05):**
- LLM calls now check budget BEFORE making API requests
- If budget exceeded, LLM features are skipped (not blocked)
- State file: `state/llm_token_usage.json`
- Default limits: 100K tokens/day, $50/day
- Alert threshold: $25 (logs warning)

#### 6. "Circuit breaker triggered"

```bash
# Check which breaker
python -c "from cognitive.circuit_breakers import get_breaker_status; print(get_breaker_status())"

# Common causes:
# - calibration_drift: Self-model stuck at extremes
# - loss_concentration: Too many recent losses
# - confidence_collapse: Model returning constant values
# - regime_stuck: Regime detector not updating
```

---

## Maintenance Procedures

### Daily

1. Review `logs/daily_picks.csv` for signal quality
2. Check `/metrics` endpoint for anomalies
3. Verify no orphaned orders

### Weekly

1. Run full reconciliation: `python scripts/reconcile_alpaca.py --full`
2. Review cognitive decisions: `tail -100 logs/cognitive_decisions.jsonl`
3. Check hash chain: `python scripts/verify_hash_chain.py`

### Monthly

1. Backup state: `python scripts/backup.py --full`
2. Review and rotate logs: `python scripts/cleanup.py --logs --older-than 30`
3. Update universe if needed: `python scripts/build_universe_polygon.py`
4. Regenerate requirements.lock.txt: `pip freeze > requirements.lock.txt`

### Quarterly

1. Retrain ML models: `python scripts/train_ensemble.py`
2. Review walk-forward performance: `python scripts/run_wf_polygon.py`
3. Update strategy parameters if needed
4. Full system audit

---

## Verification Commands

```bash
# 1. After ML training
dir models\
python -c "from ml_advanced.ensemble.ensemble_predictor import EnsemblePredictor; p = EnsemblePredictor(); print('Ensemble loaded')"

# 2. Cognitive preflight
python scripts/preflight.py --cognitive

# 3. Full test suite
pytest tests/ -q

# 4. Backtest sanity check
python scripts/backtest_dual_strategy.py --start 2023-01-01 --end 2024-12-31 --cap 150

# 5. LLM disabled smoke test
python scripts/generate_briefing.py --phase pregame  # With llm.enabled=false

# 6. Top-N selection test (with selection.top_n.enabled=true)
python scripts/scan.py --cap 100 --top3
```

---

## Contact & Escalation

| Issue Type | First Response | Escalation |
|------------|---------------|------------|
| Data outage | Check provider status, use fallback | Contact Polygon support |
| Broker API down | Create kill switch, wait | Contact Alpaca support |
| Unexpected losses | Review cognitive logs, halt if needed | Manual review required |
| System crash | Check logs, restart | Review crash dump |

---

## Appendix: File Locations

| Category | Path | Purpose |
|----------|------|---------|
| Config | `config/base.yaml` | Main configuration |
| Config Schema | `config/settings_schema.py` | Pydantic validation (NEW 2026-01-05) |
| Frozen Params | `config/frozen_strategy_params_v2.2.json` | Strategy parameters |
| State | `state/` | Runtime state files |
| State Manager | `portfolio/state_manager.py` | Central state management (NEW 2026-01-05) |
| LLM Budget | `state/llm_token_usage.json` | Token/cost tracking (NEW 2026-01-05) |
| Logs | `logs/` | All log files |
| Cache | `cache/` | Data cache |
| Models | `models/` | Trained ML models |
| Universe | `data/universe/` | Stock universes |
| Data Quality | `data/quality/` | Canary scripts (NEW 2026-01-05) |
| Backtest Output | `wf_outputs/` | Walk-forward results |

---

## Appendix: New Features (2026-01-05)

### State Manager

All state files are now managed through a central `StateManager` with:
- File locking (cross-platform via `filelock`)
- Atomic writes (temp file + rename pattern)
- Thread-safe operations

```python
from portfolio.state_manager import get_state_manager
sm = get_state_manager()

# Read state
positions = sm.get_positions()
budget = sm.get_weekly_budget()

# Write state (atomic with locking)
sm.set_positions(new_positions)
```

### Data Quality Canaries

Run canary checks to validate data sources:

```bash
# Earnings data source check
python -c "from core.earnings_filter import run_earnings_canary; print(run_earnings_canary())"

# Price discontinuity check (splits/dividends)
python -c "from data.quality import check_recent_data; print(check_recent_data('AAPL'))"
```

### Secrets Masking

Prevent API keys from appearing in logs:

```python
from core.secrets import mask_secrets, SecretsMaskingFilter

# Mask in strings
safe = mask_secrets("POLYGON_API_KEY=abc123")  # Returns masked version

# Add to logging
import logging
handler = logging.StreamHandler()
handler.addFilter(SecretsMaskingFilter())
```

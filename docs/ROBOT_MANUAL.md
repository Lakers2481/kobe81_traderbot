# ROBOT_MANUAL.md - Complete Kobe Trading Robot Guide

> **Last Updated:** 2026-01-03
> **Version:** 2.3
> **System Status:** Production Ready (Micro-Cap)

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Data Pipeline](#3-data-pipeline)
4. [Signal Generation](#4-signal-generation)
5. [ML Confidence Scoring](#5-ml-confidence-scoring)
6. [Risk Management](#6-risk-management)
7. [Execution](#7-execution)
8. [State Management](#8-state-management)
9. [Professional Execution Flow](#9-professional-execution-flow)
10. [Monitoring & Alerts](#10-monitoring--alerts)
11. [Operations Guide](#11-operations-guide)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. System Overview

### What is Kobe?

Kobe is a Python quantitative trading system for mean-reversion strategies. It combines:

- **IBS+RSI Strategy**: Internal Bar Strength < 0.08 + RSI(2) < 5
- **ICT Turtle Soup**: Liquidity sweep detection with 0.3 ATR filter

### Key Features

| Feature | Description |
|---------|-------------|
| Dual Strategy | Combines mean-reversion + liquidity sweep |
| ML Confidence | GradientBoost scoring per strategy |
| Risk Gates | PolicyGate, KillZoneGate, ExposureGate |
| IOC LIMIT Orders | No market orders, controlled fills |
| Kill Switch | Emergency halt mechanism |
| Idempotency | Duplicate prevention via SQLite |

### Performance (Backtest)

| Metric | Value |
|--------|-------|
| Win Rate | 64% |
| Profit Factor | 1.60 |
| Max Drawdown | <15% |
| Sharpe Ratio | >1.0 |

---

## 2. Architecture

### Layer Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER/SCHEDULER                           │
│                    scripts/*.py, runner.py                       │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                               │
│  data/providers/polygon_eod.py → multi_source.py (fallback)    │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SIGNAL GENERATION                           │
│  strategies/dual_strategy/combined.py → DualStrategyScanner    │
│  ├── IBS+RSI (strategies/ibs_rsi/strategy.py)                   │
│  └── Turtle Soup (strategies/ict/turtle_soup.py)                │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ML CONFIDENCE                               │
│  ml_meta/model.py → GradientBoost per strategy                  │
│  scripts/scan.py:1116 → blends ML + sentiment                   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RISK GATES                                  │
│  risk/policy_gate.py → $75/order, $1k/daily                     │
│  risk/kill_zone_gate.py → 9:30-10:00 blocked                    │
│  risk/weekly_exposure_gate.py → 40% weekly cap                  │
│  risk/signal_quality_gate.py → Score thresholds                 │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      POSITION SIZING                             │
│  risk/equity_sizer.py → 2% risk per trade                       │
│  Dual cap: min(shares_by_risk, shares_by_notional)              │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      EXECUTION                                   │
│  execution/broker_alpaca.py → place_ioc_limit()                 │
│  oms/idempotency_store.py → duplicate prevention                │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      STATE & LOGGING                             │
│  state/*.json → positions, orders, watchlist                    │
│  logs/*.jsonl → structured events                               │
│  core/hash_chain.py → audit trail                               │
└─────────────────────────────────────────────────────────────────┘
```

### Key Files

| Layer | Primary File | Purpose |
|-------|--------------|---------|
| Data | `data/providers/polygon_eod.py` | EOD OHLCV fetch |
| Strategy | `strategies/dual_strategy/combined.py` | DualStrategyScanner |
| ML | `ml_meta/model.py` | Confidence scoring |
| Risk | `risk/policy_gate.py` | Budget enforcement |
| Execution | `execution/broker_alpaca.py` | IOC LIMIT orders |
| State | `oms/idempotency_store.py` | Duplicate prevention |

---

## 3. Data Pipeline

### Data Flow

```
Polygon.io API
      │
      ▼
polygon_eod.py → fetch_daily_bars_polygon()
      │
      ▼
CSV Cache (cache/polygon/)
      │
      ▼
multi_source.py → fetch_daily_bars_multi() [fallback chain]
      │
      ▼
DataFrame with OHLCV
```

### Providers

| Provider | Priority | API Key Required |
|----------|----------|------------------|
| Polygon.io | Primary | Yes |
| Stooq | Fallback 1 | No |
| Yahoo Finance | Fallback 2 | No |

### Universe

```
data/universe/optionable_liquid_900.csv
- 900 stocks
- Optionable
- Liquid (high volume)
- 10+ years history
```

---

## 4. Signal Generation

### DualStrategyScanner

```python
from strategies.dual_strategy import DualStrategyScanner, DualStrategyParams

scanner = DualStrategyScanner(DualStrategyParams())
signals = scanner.scan_signals_over_time(df)
```

### IBS+RSI Strategy

| Parameter | Value |
|-----------|-------|
| IBS Threshold | < 0.08 |
| RSI Period | 2 |
| RSI Threshold | < 5 |
| SMA Filter | 200-day |

**Entry:** IBS < 0.08 AND RSI(2) < 5 AND Price > SMA(200)
**Exit:** ATR(14) x 2 stop OR 7-bar time stop

### Turtle Soup Strategy

| Parameter | Value |
|-----------|-------|
| Lookback | 20 bars |
| Sweep Strength | >= 0.3 ATR |
| Reclaim Required | Yes |

**Entry:** Price sweeps below 20-day low by 0.3+ ATR, then reclaims
**Exit:** ATR-based stop

### Output Columns

| Column | Description |
|--------|-------------|
| timestamp | Signal time |
| symbol | Stock ticker |
| side | LONG/SHORT |
| entry_price | Entry level |
| stop_loss | Stop level |
| take_profit | Target level |
| reason | IBS_RSI or TURTLE_SOUP |

---

## 5. ML Confidence Scoring

### Model Location

```
ml_meta/model.py → MLConfidenceModel
models/*.pkl → Trained models
```

### Scoring Flow

```
Features (150+) → GradientBoost → Probability Score
                                        │
                                        ▼
                              Blend with Sentiment
                                        │
                                        ▼
                              Final Confidence (0-100)
```

### Features

| Category | Count | Examples |
|----------|-------|----------|
| Technical | 50+ | RSI, ATR, MACD, Bollinger |
| Price Action | 30+ | Candle patterns, momentum |
| Volume | 20+ | Volume ratio, accumulation |
| Time/Calendar | 15+ | Day of week, month |
| Lag Features | 20+ | t-1 to t-20 values |

---

## 6. Risk Management

### Risk Gates (Sequential)

```
Signal → PolicyGate → KillZoneGate → ExposureGate → QualityGate → PASS/FAIL
```

### PolicyGate

| Limit | Value |
|-------|-------|
| Max per order | $75 |
| Max daily | $1,000 |

### KillZoneGate

| Time (ET) | Status |
|-----------|--------|
| Before 9:30 | BLOCKED |
| 9:30-10:00 | BLOCKED (amateur hour) |
| 10:00-11:30 | ALLOWED (primary window) |
| 11:30-14:30 | BLOCKED (lunch chop) |
| 14:30-15:30 | ALLOWED (power hour) |
| 15:30-16:00 | BLOCKED (close) |

### ExposureGate

| Limit | Value |
|-------|-------|
| Per position | 10% |
| Daily | 20% |
| Weekly | 40% |

### Position Sizing

```python
shares_by_risk = (equity * 0.02) / (entry_price - stop_loss)
shares_by_notional = (equity * 0.20) / entry_price
final_shares = min(shares_by_risk, shares_by_notional)
```

---

## 7. Execution

### Broker Adapter

```python
from execution.broker_alpaca import AlpacaBroker

broker = AlpacaBroker()
broker.place_ioc_limit(symbol, qty, side, limit_price)
```

### Order Types

| Type | Used | Reason |
|------|------|--------|
| IOC LIMIT | Yes | Controlled fills, no chasing |
| Market | Never | Uncontrolled slippage |
| Day LIMIT | No | Could fill at wrong time |

### Limit Price Formula

```python
limit_price = best_ask * 1.001  # 0.1% buffer
```

### Idempotency

```python
from oms.idempotency_store import IdempotencyStore

store = IdempotencyStore()
if store.check(order_id):
    print("Duplicate detected, skipping")
else:
    broker.submit(order)
    store.record(order_id)
```

---

## 8. State Management

### State Files

| File | Purpose |
|------|---------|
| `state/positions.json` | Current positions |
| `state/order_state.json` | Order records |
| `state/watchlist/next_day.json` | Tomorrow's watchlist |
| `state/watchlist/today_validated.json` | Today's validated |
| `state/KILL_SWITCH` | Emergency halt flag |
| `state/idempotency_store.sqlite` | Duplicate prevention |
| `state/hash_chain.jsonl` | Audit trail |

### Kill Switch

```bash
# Activate
python scripts/kill.py --reason "Emergency"
# Creates state/KILL_SWITCH

# Deactivate
python scripts/resume.py --confirm
# Removes state/KILL_SWITCH
```

---

## 9. Professional Execution Flow

### Daily Schedule

| Time (ET) | Action | Script |
|-----------|--------|--------|
| 3:30 PM (prev) | Build watchlist | `overnight_watchlist.py` |
| 8:00 AM | Validate gaps/news | `premarket_validator.py` |
| 8:15 AM | Pre-game blueprint | `generate_pregame_blueprint.py` |
| 9:30-10:00 | Observe (NO TRADE) | `opening_range_observer.py` |
| 10:00 AM | First scan | `run_paper_trade.py` |
| 10:30 AM | Fallback scan | `run_paper_trade.py --fallback-enabled` |
| 14:30 PM | Power hour | `run_paper_trade.py` |

### Watchlist Flow

```
Overnight Scan → Premarket Validation → Opening Observation → Trade
       │                │                      │                │
       ▼                ▼                      ▼                ▼
   Top 5 picks      VALID/INVALID         Log only          Execute
```

---

## 10. Monitoring & Alerts

### Health Endpoints

```
http://localhost:5000/health     # Health check
http://localhost:5000/metrics    # Prometheus metrics
```

### Telegram Alerts

```python
from alerts.telegram_alerter import TelegramAlerter

alerter = TelegramAlerter()
alerter.send_alert("Order filled: AAPL 100 @ 150.00")
```

### Logging

```python
from core.structured_log import log_event

log_event("order_submitted", {
    "symbol": "AAPL",
    "qty": 100,
    "price": 150.00
})
# Writes to logs/events.jsonl
```

---

## 11. Operations Guide

### Daily Checklist

**Pre-Market (8:00 AM):**
```bash
python scripts/preflight.py --dotenv ./.env
python scripts/premarket_validator.py
```

**Market Open (10:00 AM):**
```bash
python scripts/run_paper_trade.py --universe data/universe/optionable_liquid_900.csv --cap 50 --watchlist-only
```

**End of Day:**
```bash
python scripts/overnight_watchlist.py --cap 900 --prefetch
python scripts/positions.py
python scripts/pnl.py
```

### Weekly Checklist

```bash
python scripts/reconcile_alpaca.py
python scripts/backtest_dual_strategy.py --start 2023-01-01 --end 2024-12-31 --cap 150
```

---

## 12. Troubleshooting

### Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| No signals | Kill zone blocked | Wait for valid window |
| Order rejected | Kill switch active | `python scripts/resume.py --confirm` |
| Data missing | Cache expired | `python scripts/prefetch_polygon_universe.py` |
| Wrong win rate | Standalone strategy | Use `DualStrategyScanner` |

### Debug Commands

```bash
# Check kill switch status
ls state/KILL_SWITCH

# View recent logs
python scripts/logs.py

# Debug signals
python scripts/debug_signals.py

# Verify hash chain
python scripts/verify_hash_chain.py
```

### Emergency Procedures

1. **HALT ALL TRADING:**
   ```bash
   python scripts/kill.py --reason "Emergency"
   ```

2. **VIEW POSITIONS:**
   ```bash
   python scripts/positions.py
   ```

3. **MANUAL CLOSE:** Use Alpaca dashboard

4. **RESUME:**
   ```bash
   python scripts/resume.py --confirm
   ```

---

## Related Documentation

- [CLAUDE.md](../CLAUDE.md) - Rules and commands
- [STATUS.md](STATUS.md) - Single Source of Truth
- [ARCHITECTURE.md](ARCHITECTURE.md) - Detailed wiring
- [READINESS.md](READINESS.md) - Production status
- [PROFESSIONAL_EXECUTION_FLOW.md](PROFESSIONAL_EXECUTION_FLOW.md) - Trading flow

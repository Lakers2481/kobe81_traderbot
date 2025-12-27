# Kobe Trading Robot - Complete System Blueprint

> **Purpose:** Complete reference for understanding the Kobe trading system.
> Any AI reading this document should fully understand the architecture, data flow, and all components.

## Executive Summary

**Kobe** is a production-grade algorithmic trading system for US equities with:
- **Strategies:** Donchian Breakout (trend-following) + ICT Turtle Soup (mean-reversion)
- **Universe:** 900 optionable, liquid US stocks with 10+ years of history
- **Data:** Polygon.io (EOD), Alpaca (execution, quotes)
- **Risk:** Multi-layer safety (PolicyGate, LiquidityGate, Kill Switch)
- **Mode:** Paper trading ready, live trading capable (micro budget)

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           KOBE TRADING SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │  DATA LAYER  │───▶│  STRATEGIES  │───▶│   RISK GATE  │───▶│  EXECUTION │ │
│  │              │    │              │    │              │    │            │ │
│  │ • Polygon    │    │ • Donchian   │    │ • PolicyGate │    │ • Alpaca   │ │
│  │ • Universe   │    │ • TurtleSoup │    │ • Liquidity  │    │ • IOC Limit│ │
│  │ • Cache      │    │ • ML Meta    │    │ • Kill Switch│    │ • Logging  │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └────────────┘ │
│         │                   │                   │                   │       │
│         ▼                   ▼                   ▼                   ▼       │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                           STATE & AUDIT                                  ││
│  │  • OMS (order state, idempotency)  • Hash Chain (tamper detection)      ││
│  │  • Heartbeat (process health)       • Structured Logs (events.jsonl)    ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                           MONITORING                                     ││
│  │  • Health Endpoints (:8000)  • Drift Detection  • Calibration Tracking  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
kobe81_traderbot/
├── backtest/           # Backtesting engine
│   ├── engine.py       # Core backtester with equity curve, stops
│   ├── walk_forward.py # Walk-forward validation splits
│   ├── vectorized.py   # Fast vectorized backtester
│   └── monte_carlo.py  # Monte Carlo simulation
│
├── cognitive/          # AI decision-making layer
│   ├── brain.py        # Central cognitive coordinator
│   ├── global_workspace.py  # Publish-subscribe messaging
│   ├── episodic_memory.py   # Trade episode tracking
│   ├── semantic_memory.py   # Trading rules/knowledge
│   ├── reflection.py        # Self-introspection
│   └── curiosity.py         # Hypothesis generation
│
├── core/               # Core utilities
│   ├── hash_chain.py   # Tamper-evident audit chain
│   ├── structured_log.py    # JSON logging
│   ├── kill_switch.py  # Emergency halt mechanism
│   └── config.py       # Configuration management
│
├── data/               # Data management
│   ├── providers/
│   │   └── polygon_eod.py   # Polygon.io EOD data with caching
│   ├── universe/
│   │   ├── loader.py        # Symbol list loading
│   │   └── optionable_liquid_900.csv  # 900-stock universe
│   └── lake/
│       └── dataset.py       # Data lake manifests
│
├── execution/          # Order execution
│   └── broker_alpaca.py     # Alpaca broker integration
│       # Key functions:
│       # - place_ioc_limit()     IOC limit order
│       # - get_best_ask()        Quote fetching
│       # - resolve_ioc_status()  Fill confirmation
│       # - execute_signal()      High-level entry point
│
├── ml_meta/            # ML meta-model
│   ├── features.py     # Technical feature extraction
│   ├── trainer.py      # Model training pipeline
│   └── ensemble.py     # Multi-model ensemble
│
├── monitor/            # System monitoring
│   ├── health_endpoints.py  # HTTP health checks (/health, /metrics)
│   ├── drift_detector.py    # Performance drift detection
│   ├── calibration.py       # Probability calibration
│   └── heartbeat.py         # Process heartbeat tracking
│
├── oms/                # Order Management System
│   ├── order_state.py       # OrderRecord dataclass
│   └── idempotency_store.py # Duplicate order prevention
│
├── ops/                # Operations utilities
│   ├── locks.py        # File-based locking
│   └── windows/        # Windows Task Scheduler XML
│
├── options/            # Options pricing (synthetic)
│   ├── black_scholes.py     # BS pricing with Greeks
│   ├── volatility.py        # Realized vol estimation
│   └── selection.py         # Strike selection
│
├── preflight/          # Pre-trading validation
│   ├── evidence_gate.py     # Strategy promotion gates
│   └── data_quality.py      # Data validation
│
├── research/           # Alpha research
│   ├── features.py     # Research features (25+)
│   ├── alphas.py       # Alpha signals (18+)
│   └── screener.py     # Walk-forward screening
│
├── risk/               # Risk management
│   ├── policy_gate.py       # Budget enforcement
│   │   # Per-order: $75 max
│   │   # Daily: $1,000 max
│   ├── liquidity_gate.py    # ADV/spread checks
│   └── advanced/            # Advanced risk
│       ├── monte_carlo_var.py
│       ├── kelly_position_sizer.py
│       └── correlation_limits.py
│
├── scripts/            # Runnable scripts
│   ├── runner.py            # 24/7 scheduler daemon
│   ├── scan.py              # Daily stock scanner
│   ├── preflight.py         # Pre-trade validation
│   ├── run_paper_trade.py   # Paper trading
│   ├── run_live_trade_micro.py  # Live trading (micro)
│   └── reconcile_alpaca.py  # Position reconciliation
│
├── strategies/         # Trading strategies
│   ├── donchian/
│   │   └── strategy.py      # Donchian Breakout
│   └── ict/
│       └── turtle_soup.py   # ICT Turtle Soup
│
├── state/              # Runtime state files
│   ├── KILL_SWITCH          # Emergency halt marker (if exists)
│   ├── heartbeat.json       # Process heartbeat
│   ├── kobe.lock            # Single-instance lock
│   └── positions.json       # Current positions
│
├── logs/               # Log files
│   ├── events.jsonl         # Structured event log
│   ├── trades.jsonl         # Trade execution log
│   └── daily_picks.csv      # Scanner output
│
├── tests/              # Test suite
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   └── test_*.py            # Module tests
│
├── docs/               # Documentation
│   ├── STATUS.md            # Current status & work log
│   ├── ARCHITECTURE.md      # ASCII architecture diagram
│   └── ROBOT_BLUEPRINT.md   # This file
│
├── CLAUDE.md           # Claude Code guidance
├── requirements.txt    # Python dependencies
└── pytest.ini          # Test configuration
```

---

## Data Flow: Signal to Order

```
1. SCAN (scripts/scan.py)
   │
   ├── Load Universe (900 stocks from data/universe/)
   │
   ├── Fetch EOD Data (Polygon.io with CSV caching)
   │
   ├── Generate Signals (strategies/)
   │   ├── Donchian Breakout: Entry on price > upper band
   │   └── Turtle Soup: Entry on IBS < 0.2 after setup
   │
   └── Output: logs/daily_picks.csv

2. RISK CHECK (risk/)
   │
   ├── PolicyGate.check()
   │   ├── Per-order budget ($75 max)
   │   └── Daily budget ($1,000 max)
   │
   ├── LiquidityGate.check()
   │   ├── ADV threshold ($100k min)
   │   ├── Spread threshold (0.5% max)
   │   └── Order impact (% of ADV)
   │
   └── Kill Switch check (core/kill_switch.py)

3. EXECUTION (execution/broker_alpaca.py)
   │
   ├── Get Best Ask (quote API)
   │
   ├── Place IOC Limit Order
   │   └── Limit = best_ask × 1.001
   │
   ├── Resolve Status (poll for FILLED/CANCELLED)
   │
   ├── Log Trade Event (logs/trades.jsonl)
   │
   └── Update Metrics (monitor/health_endpoints.py)

4. STATE UPDATE (oms/, core/)
   │
   ├── Update OrderRecord
   │
   ├── Append Hash Chain (tamper detection)
   │
   └── Update Positions
```

---

## Key Components Deep Dive

### 1. Strategies

**Donchian Breakout** (`strategies/donchian/strategy.py`)
- Entry: Close crosses above upper Donchian channel
- Exit: ATR(14)×2 stop loss OR 5-bar time stop
- Filter: Above SMA(200)
- Indicator shift: All signals delayed 1 bar (no lookahead)

**ICT Turtle Soup** (`strategies/ict/turtle_soup.py`)
- Entry: IBS < 0.2 after false breakout of prior low
- Exit: ATR(14)×2 stop OR 5-bar time stop
- Filter: Above SMA(200)
- Indicator shift: Signals shifted 1 bar

### 2. Risk Management

**PolicyGate** (`risk/policy_gate.py`)
```python
class PolicyGate:
    per_order_budget: float = 75.0   # Max per trade
    daily_budget: float = 1000.0     # Max per day

    def check(self, amount: float) -> Tuple[bool, str]:
        # Returns (allowed, reason)
```

**LiquidityGate** (`risk/liquidity_gate.py`)
```python
class LiquidityGate:
    min_adv: float = 100_000        # Minimum avg daily volume ($)
    max_spread_pct: float = 0.50    # Maximum bid-ask spread (%)
    max_order_pct_adv: float = 1.0  # Max order size as % of ADV
```

**Kill Switch** (`core/kill_switch.py`)
- Create `state/KILL_SWITCH` file to halt all order submissions
- All broker functions decorated with `@require_no_kill_switch`

### 3. Execution

**IOC Limit Orders** (`execution/broker_alpaca.py`)
```python
def place_ioc_limit(
    symbol: str,
    side: str,          # "buy" or "sell"
    qty: int,
    limit_price: float,
    client_order_id: str
) -> OrderRecord:
    # Places Immediate-Or-Cancel limit order
    # Resolves status (polls for fill)
    # Logs trade event
    # Returns OrderRecord with fill details
```

### 4. Monitoring

**Health Endpoints** (`monitor/health_endpoints.py`)
- `GET /health` - Basic health check
- `GET /readiness` - Ready for trading
- `GET /liveness` - Process alive
- `GET /metrics` - Performance metrics (JSON)

**Metrics Tracked:**
- `ioc_submitted`, `ioc_filled`, `ioc_cancelled`, `liquidity_blocked`
- `pnl_realized_usd`, `pnl_unrealized_usd`
- `win_count`, `loss_count`, `win_rate`
- Timestamps for last trade events

**Heartbeat** (`monitor/heartbeat.py`)
- Writes `state/heartbeat.json` every 60 seconds
- Contains: timestamp, pid, mode, last_action
- Use `is_heartbeat_stale(max_age_s=300)` to detect hung processes

### 5. State Files

| File | Purpose | Format |
|------|---------|--------|
| `state/KILL_SWITCH` | Emergency halt marker | Empty file (presence = halt) |
| `state/heartbeat.json` | Process health | JSON: timestamp, pid, mode |
| `state/kobe.lock` | Single-instance lock | Lock file |
| `state/positions.json` | Current holdings | JSON: {symbol: qty} |
| `logs/events.jsonl` | Structured events | JSON lines |
| `logs/trades.jsonl` | Trade execution log | JSON lines |
| `logs/hash_chain.jsonl` | Tamper-evident audit | JSON lines with hashes |

---

## Configuration

### Environment Variables (`.env`)
```bash
POLYGON_API_KEY=...          # Polygon.io API key
ALPACA_API_KEY_ID=...        # Alpaca API key
ALPACA_API_SECRET_KEY=...    # Alpaca secret
ALPACA_BASE_URL=...          # https://paper-api.alpaca.markets or https://api.alpaca.markets
TELEGRAM_TOKEN=...           # Optional: Telegram alerts
TELEGRAM_CHAT_ID=...         # Optional: Telegram chat ID
```

### Core Config (`core/config.py`)
- Config pinning with SHA256 hash
- Hot-reload disabled for safety
- All config changes logged

---

## Running the System

### Preflight Check
```bash
python scripts/preflight.py --dotenv .env
# Checks: env, config, broker, quotes API, Polygon freshness
```

### Daily Scan
```bash
python scripts/scan.py --universe data/universe/optionable_liquid_900.csv
# Output: logs/daily_picks.csv
```

### Paper Trading
```bash
python scripts/run_paper_trade.py --universe data/universe/optionable_liquid_900.csv --cap 50
```

### 24/7 Runner
```bash
python scripts/runner.py --mode paper --universe data/universe/optionable_liquid_900.csv --cap 50 --scan-times 09:35,10:30,15:55
```

### Live Trading (Micro Budget)
```bash
python scripts/run_live_trade_micro.py --universe data/universe/optionable_liquid_900.csv --cap 10
```

---

## Safety Rails

1. **Kill Switch**: Create `state/KILL_SWITCH` to halt all orders instantly
2. **PolicyGate**: Hard limits on per-order ($75) and daily ($1,000) spending
3. **LiquidityGate**: Blocks orders on illiquid or wide-spread stocks
4. **Idempotency Store**: Prevents duplicate order submissions
5. **Hash Chain**: Tamper-evident audit trail
6. **File Lock**: Prevents multiple runner instances
7. **Heartbeat**: Detects hung processes
8. **Signal Handlers**: Graceful shutdown on SIGTERM/SIGINT

---

## Test Suite

```bash
# Run all tests
python -m pytest tests/ -v

# Quick smoke test
python -m pytest tests/ -v -k "test_import"

# Current status: 365 tests passing
```

**Test Categories:**
- `tests/unit/` - Unit tests for individual modules
- `tests/integration/` - End-to-end workflow tests
- `tests/test_*.py` - Module-specific tests

---

## Common Tasks for AI Agents

### Add a New Strategy
1. Create `strategies/new_name/strategy.py`
2. Implement `Strategy` class with `generate_signals()` and `scan_signals_over_time()`
3. Add tests in `tests/unit/test_strategies.py`
4. Register in universe loader if needed

### Debug Order Issues
1. Check `logs/trades.jsonl` for execution details
2. Check `logs/events.jsonl` for system events
3. Check `/metrics` endpoint for counters
4. Verify kill switch: `ls state/KILL_SWITCH`

### Add New Metrics
1. Edit `monitor/health_endpoints.py`
2. Add counter to `_metrics["requests"]`
3. Call `update_trade_event(kind)` where needed

### Modify Risk Limits
1. Edit `risk/policy_gate.py` for budget limits
2. Edit `risk/liquidity_gate.py` for liquidity thresholds
3. Run tests: `python -m pytest tests/unit/test_risk.py -v`

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2025-12-27 | 1.0 | Initial blueprint, all 365 tests passing |

---

## Quick Reference

| What | Where |
|------|-------|
| Start trading | `python scripts/runner.py --mode paper` |
| Stop trading | Create `state/KILL_SWITCH` |
| Check health | `curl http://localhost:8000/health` |
| View metrics | `curl http://localhost:8000/metrics` |
| View logs | `tail logs/events.jsonl` |
| View trades | `tail logs/trades.jsonl` |
| Run tests | `python -m pytest tests/ -v` |
| Daily picks | `cat logs/daily_picks.csv` |
| Positions | `cat state/positions.json` |

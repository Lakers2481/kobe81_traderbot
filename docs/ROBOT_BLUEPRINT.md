# Kobe Trading Robot - Complete System Blueprint

> **Purpose:** Complete reference for understanding the Kobe trading system.
> Any AI reading this document should fully understand the architecture, data flow, and all components.

## Executive Summary

**Kobe** is a production-grade algorithmic trading system for US equities with:
- **Strategies:** IBS+RSI (trend-following) + ICT Turtle Soup (mean-reversion)
- **Universe:** 900 optionable, liquid US stocks with 10+ years of history
- **Data:** Polygon.io (EOD), Alpaca (execution, quotes)
- **Risk:** Multi-layer safety (PolicyGate, LiquidityGate, Kill Switch)
- **Mode:** Paper trading ready, live trading capable (micro budget)

---

## System Architecture Overview

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                           KOBE TRADING SYSTEM                                â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚                                                                              â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”‚
â”‚  â”‚  DATA LAYER  â”‚â”€â”€â”€â–¶â”‚  STRATEGIES  â”‚â”€â”€â”€â–¶â”‚   RISK GATE  â”‚â”€â”€â”€â–¶â”‚  EXECUTION â”‚ â”‚
â”‚  â”‚              â”‚    â”‚              â”‚    â”‚              â”‚    â”‚            â”‚ â”‚
â”‚  â”‚ â€¢ Polygon    â”‚    â”‚ â€¢ IBS+RSI   â”‚    â”‚ â€¢ PolicyGate â”‚    â”‚ â€¢ Alpaca   â”‚ â”‚
â”‚  â”‚ â€¢ Universe   â”‚    â”‚ â€¢ TurtleSoup â”‚    â”‚ â€¢ Liquidity  â”‚    â”‚ â€¢ IOC Limitâ”‚ â”‚
â”‚  â”‚ â€¢ Cache      â”‚    â”‚ â€¢ ML Meta    â”‚    â”‚ â€¢ Kill Switchâ”‚    â”‚ â€¢ Logging  â”‚ â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜ â”‚
â”‚         â”‚                   â”‚                   â”‚                   â”‚       â”‚
â”‚         â–¼                   â–¼                   â–¼                   â–¼       â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”â”‚
â”‚  â”‚                           STATE & AUDIT                                  â”‚â”‚
â”‚  â”‚  â€¢ OMS (order state, idempotency)  â€¢ Hash Chain (tamper detection)      â”‚â”‚
â”‚  â”‚  â€¢ Heartbeat (process health)       â€¢ Structured Logs (events.jsonl)    â”‚â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜â”‚
â”‚                                                                              â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”â”‚
â”‚  â”‚                           MONITORING                                     â”‚â”‚
â”‚  â”‚  â€¢ Health Endpoints (:8000)  â€¢ Drift Detection  â€¢ Calibration Tracking  â”‚â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

---

## Directory Structure

```
kobe81_traderbot/
â”œâ”€â”€ backtest/           # Backtesting engine
â”‚   â”œâ”€â”€ engine.py       # Core backtester with equity curve, stops
â”‚   â”œâ”€â”€ walk_forward.py # Walk-forward validation splits
â”‚   â”œâ”€â”€ vectorized.py   # Fast vectorized backtester
â”‚   â””â”€â”€ monte_carlo.py  # Monte Carlo simulation
â”‚
â”œâ”€â”€ cognitive/          # AI decision-making layer
â”‚   â”œâ”€â”€ brain.py        # Central cognitive coordinator
â”‚   â”œâ”€â”€ global_workspace.py  # Publish-subscribe messaging
â”‚   â”œâ”€â”€ episodic_memory.py   # Trade episode tracking
â”‚   â”œâ”€â”€ semantic_memory.py   # Trading rules/knowledge
â”‚   â”œâ”€â”€ reflection.py        # Self-introspection
â”‚   â””â”€â”€ curiosity.py         # Hypothesis generation
â”‚
â”œâ”€â”€ core/               # Core utilities
â”‚   â”œâ”€â”€ hash_chain.py   # Tamper-evident audit chain
â”‚   â”œâ”€â”€ structured_log.py    # JSON logging
â”‚   â”œâ”€â”€ kill_switch.py  # Emergency halt mechanism
â”‚   â””â”€â”€ config.py       # Configuration management
â”‚
â”œâ”€â”€ data/               # Data management
â”‚   â”œâ”€â”€ providers/
â”‚   â”‚   â””â”€â”€ polygon_eod.py   # Polygon.io EOD data with caching
â”‚   â”œâ”€â”€ universe/
â”‚   â”‚   â”œâ”€â”€ loader.py        # Symbol list loading
â”‚   â”‚   â””â”€â”€ optionable_liquid_900.csv  # 900-stock universe
â”‚   â””â”€â”€ lake/
â”‚       â””â”€â”€ dataset.py       # Data lake manifests
â”‚
â”œâ”€â”€ execution/          # Order execution
â”‚   â””â”€â”€ broker_alpaca.py     # Alpaca broker integration
â”‚       # Key functions:
â”‚       # - place_ioc_limit()     IOC limit order
â”‚       # - get_best_ask()        Quote fetching
â”‚       # - resolve_ioc_status()  Fill confirmation
â”‚       # - execute_signal()      High-level entry point
â”‚
â”œâ”€â”€ ml_meta/            # ML meta-model
â”‚   â”œâ”€â”€ features.py     # Technical feature extraction
â”‚   â”œâ”€â”€ trainer.py      # Model training pipeline
â”‚   â””â”€â”€ ensemble.py     # Multi-model ensemble
â”‚
â”œâ”€â”€ monitor/            # System monitoring
â”‚   â”œâ”€â”€ health_endpoints.py  # HTTP health checks (/health, /metrics)
â”‚   â”œâ”€â”€ drift_detector.py    # Performance drift detection
â”‚   â”œâ”€â”€ calibration.py       # Probability calibration
â”‚   â””â”€â”€ heartbeat.py         # Process heartbeat tracking
â”‚
â”œâ”€â”€ oms/                # Order Management System
â”‚   â”œâ”€â”€ order_state.py       # OrderRecord dataclass
â”‚   â””â”€â”€ idempotency_store.py # Duplicate order prevention
â”‚
â”œâ”€â”€ ops/                # Operations utilities
â”‚   â”œâ”€â”€ locks.py        # File-based locking
â”‚   â””â”€â”€ windows/        # Windows Task Scheduler XML
â”‚
â”œâ”€â”€ options/            # Options pricing (synthetic)
â”‚   â”œâ”€â”€ black_scholes.py     # BS pricing with Greeks
â”‚   â”œâ”€â”€ volatility.py        # Realized vol estimation
â”‚   â””â”€â”€ selection.py         # Strike selection
â”‚
â”œâ”€â”€ preflight/          # Pre-trading validation
â”‚   â”œâ”€â”€ evidence_gate.py     # Strategy promotion gates
â”‚   â””â”€â”€ data_quality.py      # Data validation
â”‚
â”œâ”€â”€ research/           # Alpha research
â”‚   â”œâ”€â”€ features.py     # Research features (25+)
â”‚   â”œâ”€â”€ alphas.py       # Alpha signals (18+)
â”‚   â””â”€â”€ screener.py     # Walk-forward screening
â”‚
â”œâ”€â”€ risk/               # Risk management
â”‚   â”œâ”€â”€ policy_gate.py       # Budget enforcement
â”‚   â”‚   # Per-order: $75 max
â”‚   â”‚   # Daily: $1,000 max
â”‚   â”œâ”€â”€ liquidity_gate.py    # ADV/spread checks
â”‚   â””â”€â”€ advanced/            # Advanced risk
â”‚       â”œâ”€â”€ monte_carlo_var.py
â”‚       â”œâ”€â”€ kelly_position_sizer.py
â”‚       â””â”€â”€ correlation_limits.py
â”‚
â”œâ”€â”€ scripts/            # Runnable scripts
â”‚   â”œâ”€â”€ runner.py            # 24/7 scheduler daemon
â”‚   â”œâ”€â”€ scan.py              # Daily stock scanner
â”‚   â”œâ”€â”€ preflight.py         # Pre-trade validation
â”‚   â”œâ”€â”€ run_paper_trade.py   # Paper trading
â”‚   â”œâ”€â”€ run_live_trade_micro.py  # Live trading (micro)
â”‚   â””â”€â”€ reconcile_alpaca.py  # Position reconciliation
â”‚
â”œâ”€â”€ strategies/         # Trading strategies
â”‚   â”œâ”€â”€ IBS+RSI/
â”‚   â”‚   â””â”€â”€ strategy.py      # IBS+RSI
â”‚   â””â”€â”€ ict/
â”‚       â””â”€â”€ turtle_soup.py   # ICT Turtle Soup
â”‚
â”œâ”€â”€ state/              # Runtime state files
â”‚   â”œâ”€â”€ KILL_SWITCH          # Emergency halt marker (if exists)
â”‚   â”œâ”€â”€ heartbeat.json       # Process heartbeat
â”‚   â”œâ”€â”€ kobe.lock            # Single-instance lock
â”‚   â””â”€â”€ positions.json       # Current positions
â”‚
â”œâ”€â”€ logs/               # Log files
â”‚   â”œâ”€â”€ events.jsonl         # Structured event log
â”‚   â”œâ”€â”€ trades.jsonl         # Trade execution log
â”‚   â””â”€â”€ daily_picks.csv      # Scanner output
â”‚
â”œâ”€â”€ tests/              # Test suite
â”‚   â”œâ”€â”€ unit/                # Unit tests
â”‚   â”œâ”€â”€ integration/         # Integration tests
â”‚   â””â”€â”€ test_*.py            # Module tests
â”‚
â”œâ”€â”€ docs/               # Documentation
â”‚   â”œâ”€â”€ STATUS.md            # Current status & work log
â”‚   â”œâ”€â”€ ARCHITECTURE.md      # ASCII architecture diagram
â”‚   â””â”€â”€ ROBOT_BLUEPRINT.md   # This file
â”‚
â”œâ”€â”€ CLAUDE.md           # Claude Code guidance
â”œâ”€â”€ requirements.txt    # Python dependencies
â””â”€â”€ pytest.ini          # Test configuration
```

---

## Data Flow: Signal to Order

```
1. SCAN (scripts/scan.py)
   â”‚
   â”œâ”€â”€ Load Universe (900 stocks from data/universe/)
   â”‚
   â”œâ”€â”€ Fetch EOD Data (Polygon.io with CSV caching)
   â”‚
   â”œâ”€â”€ Generate Signals (strategies/)
   â”‚   â”œâ”€â”€ IBS+RSI: Entry on price > upper band
   â”‚   â””â”€â”€ Turtle Soup: Entry on IBS < 0.2 after setup
   â”‚
   â””â”€â”€ Output: logs/daily_picks.csv

2. RISK CHECK (risk/)
   â”‚
   â”œâ”€â”€ PolicyGate.check()
   â”‚   â”œâ”€â”€ Per-order budget ($75 max)
   â”‚   â””â”€â”€ Daily budget ($1,000 max)
   â”‚
   â”œâ”€â”€ LiquidityGate.check()
   â”‚   â”œâ”€â”€ ADV threshold ($100k min)
   â”‚   â”œâ”€â”€ Spread threshold (0.5% max)
   â”‚   â””â”€â”€ Order impact (% of ADV)
   â”‚
   â””â”€â”€ Kill Switch check (core/kill_switch.py)

3. EXECUTION (execution/broker_alpaca.py)
   â”‚
   â”œâ”€â”€ Get Best Ask (quote API)
   â”‚
   â”œâ”€â”€ Place IOC Limit Order
   â”‚   â””â”€â”€ Limit = best_ask Ã— 1.001
   â”‚
   â”œâ”€â”€ Resolve Status (poll for FILLED/CANCELLED)
   â”‚
   â”œâ”€â”€ Log Trade Event (logs/trades.jsonl)
   â”‚
   â””â”€â”€ Update Metrics (monitor/health_endpoints.py)

4. STATE UPDATE (oms/, core/)
   â”‚
   â”œâ”€â”€ Update OrderRecord
   â”‚
   â”œâ”€â”€ Append Hash Chain (tamper detection)
   â”‚
   â””â”€â”€ Update Positions
```

---

## Key Components Deep Dive

### 1. Strategies

**IBS+RSI** (`strategies/IBS+RSI/strategy.py`)
- Entry: Close crosses above upper IBS+RSI channel
- Exit: ATR(14)Ã—2 stop loss OR 5-bar time stop
- Filter: Above SMA(200)
- Indicator shift: All signals delayed 1 bar (no lookahead)

**ICT Turtle Soup** (`strategies/ict/turtle_soup.py`)
- Entry: IBS < 0.2 after false breakout of prior low
- Exit: ATR(14)Ã—2 stop OR 5-bar time stop
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


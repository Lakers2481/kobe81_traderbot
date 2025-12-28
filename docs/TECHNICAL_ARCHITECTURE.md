# Kobe81 Traderbot - Technical Architecture Documentation

> Alignment Banner (v2.2): docs/STATUS.md is the canonical, single source of truth for active strategies, parameters, and performance metrics (QUANT INTERVIEW READY). Always consult docs/STATUS.md first.



> Note: Strategy set is standardized to IBS+RSI (mean reversion) + ICT Turtle Soup (mean reversion) and universe size is 900. Any mentions of IBS+RSI/950 symbols are legacy and will be updated. See `README.md` for the canonical setup.

**Version:** 1.0
**Last Updated:** 2025-12-26
**System Name:** Kobe81 Traderbot (Institutional-Grade Quantitative Trading System)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [10-Layer Architecture](#10-layer-architecture)
3. [Key Entry Points](#key-entry-points)
4. [Data Flow](#data-flow)
5. [Safety Mechanisms](#safety-mechanisms)
6. [Trading Strategies](#trading-strategies)
7. [System Components](#system-components)
8. [Operational Modes](#operational-modes)
9. [Evidence & Audit Trail](#evidence--audit-trail)
10. [Development & Deployment](#development--deployment)

---

## Executive Summary

Kobe81 is a production-grade quantitative trading system implementing canonical two complementary strategies: IBS+RSI (trend) and ICT Turtle Soup (mean reversion) with institutional-level risk management, compliance, and audit capabilities. The system supports backtesting, walk-forward validation, paper trading, and live execution with micro-budgets.

**Core Characteristics:**
- **Language:** Python 3.11+
- **Architecture:** 10-layer modular design with clear separation of concerns
- **Execution:** IOC LIMIT orders via Alpaca broker (paper/live)
- **Data:** Polygon.io EOD OHLCV with local CSV caching
- **Universe:** 900 optionable, liquid stocks with 10+ years of coverage
- **Risk Controls:** Kill switch, PolicyGate budgets, idempotency, tamper-proof audit chain
- **No Lookahead:** All indicators shifted 1 bar; signals at close(t), fills at open(t+1)

**Key Files:** 112 Python modules organized across 9 primary packages

---

## 10-Layer Architecture

The system follows a strict layered architecture with unidirectional dependencies:

### Layer 0: External Data
**Purpose:** Vendor integrations for market data and execution

**Modules:**
- `data/providers/polygon_eod.py` - Polygon.io daily bars API
- `execution/broker_alpaca.py` - Alpaca REST API for orders/quotes

**Configuration:**
- `config/env_loader.py` - Loads environment variables from `.env`
- `config/settings.json` - System parameters (initial cash, slippage, API limits)

**Environment Variables Required:**
```bash
POLYGON_API_KEY=<polygon_api_key>
ALPACA_API_KEY_ID=<alpaca_key>
ALPACA_API_SECRET_KEY=<alpaca_secret>
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # or live endpoint
```

---

### Layer 1: Data Ingestion
**Purpose:** Fetch, cache, and normalize market data

**Primary Module:** `data/providers/polygon_eod.py`

**Key Function:**
```python
fetch_daily_bars_polygon(
    symbol: str,
    start: str,        # YYYY-MM-DD
    end: str,          # YYYY-MM-DD
    cache_dir: Path,
    cfg: PolygonConfig
) -> pd.DataFrame
```

**Features:**
- **CSV Caching:** `data/cache/{SYMBOL}_{START}_{END}.csv`
- **Superset Detection:** Reuses cached ranges when available
- **Rate Limiting:** 0.30s sleep between requests (configurable)
- **Retry Logic:** 3 attempts with exponential backoff
- **Adjusted Prices:** Corporate actions automatically handled

**Output Schema:**
```
timestamp | symbol | open | high | low | close | volume
```

---

### Layer 2: Data Processing
**Purpose:** Clean and prepare data for strategy consumption

**Implementation:** Minimal processing layer - EOD bars are clean by design
- Strategies handle indicator computation with proper shifting
- No resampling or alignment needed (daily bars only)
- Lookback prevention enforced at strategy level via `.shift(1)`

---

### Layer 3: Universe Management
**Purpose:** Define and maintain the tradeable stock universe

**Module:** `data/universe/loader.py`

**Universe File:** `data/universe/optionable_liquid_900.csv`
- **Size:** 900 stocks
- **Criteria:**
  - Listed options contracts (verified via Polygon)
  - Average daily volume > threshold
  - 10+ years of historical data (2015-01-01 to 2024-12-31)
  - Price > $5 (minimum)

**Key Function:**
```python
load_universe(path: Path, cap: int = None) -> List[str]
```

**Builder Scripts:**
- `scripts/build_universe_polygon.py` - Constructs 900-stock universe with proofs
- `scripts/validate_universe_coverage.py` - Asserts coverage requirements
- `scripts/check_polygon_earliest_universe.py` - Verifies data availability

---

### Layer 4: Strategy Engine
**Purpose:** Generate trading signals based on technical indicators

**Implemented Strategies:**

#### Strategies (IBS+RSI + ICT)
**Parameters:**
```python
@dataclass
class     rsi_period: int = 2
    rsi_method: str = "wilder"          # Wilder smoothing
    sma_period: int = 200               # Trend filter
    atr_period: int = 14                # Stop calculation
    atr_stop_mult: float = 2.0          # ATR multiplier
    time_stop_bars: int = 5             # Max hold period
    long_entry_rsi_max: float = 10.0    # Oversold threshold
    short_entry_rsi_min: float = 90.0   # Overbought threshold
    long_exit_rsi_min: float = 70.0     # Exit signal
    short_exit_rsi_max: float = 30.0    # Exit signal
    min_price: float = 5.0
```

**Entry Logic:**
- **Long:** RSI(2) <= 10 AND Close > SMA(200)
- **Short:** RSI(2) >= 90 AND Close < SMA(200)

**Exit Logic:**
- **Stop Loss:** Entry Â± ATR(14) Ã— 2.0
- **Time Stop:** Close position after 5 bars
- **Signal Exit:** Long when RSI >= 70; Short when RSI <= 30

#### 2. ICT (Internal Bar Strength) (`strategies/ICT/strategy.py`)
**Parameters:**
```python
@dataclass
class IBSParams:
    sma_period: int = 200
    atr_period: int = 14
    atr_stop_mult: float = 2.0
    time_stop_bars: int = 5
    ibs_long_max: float = 0.2      # Weak close threshold
    ibs_short_min: float = 0.8     # Strong close threshold
    min_price: float = 5.0
```

**ICT Calculation:**
```python
ICT = (Close - Low) / (High - Low)
```

**Entry Logic:**
- **Long:** ICT < 0.2 AND Close > SMA(200)
- **Short:** ICT > 0.8 AND Close < SMA(200)

**Exit Logic:** Same as IBS+RSI/ICT (ATR stop + time stop)

#### 3. AND Filter (Combined Strategy)
**Implementation:** Merge IBS+RSI/ICT and ICT signals on same timestamp/symbol/side
```python
# In run_paper_trade.py and backtest scripts:
rsi2_signals = rsi2_strategy.scan_signals_over_time(data)
ibs_signals = ibs_strategy.scan_signals_over_time(data)
and_signals = pd.merge(rsi2_signals, ibs_signals,
                       on=['timestamp','symbol','side'])
```

**Strategy Interface:**
```python
class Strategy:
    def generate_signals(df: pd.DataFrame) -> pd.DataFrame
        # Returns signals for most recent bar only (live trading)

    def scan_signals_over_time(df: pd.DataFrame) -> pd.DataFrame
        # Returns ALL historical signals (backtesting)
```

**Lookahead Prevention:**
- All indicators computed with `.shift(1)`
- Signal computed at close(t) using indicators from close(t-1)
- Fill occurs at open(t+1)

---

### Layer 5: Risk Management
**Purpose:** Enforce position limits and safety bounds

**Module:** `risk/policy_gate.py`

**PolicyGate Configuration:**
```python
@dataclass
class RiskLimits:
    max_notional_per_order: float = 75.0    # Canary budget
    max_daily_notional: float = 1_000.0     # Daily cap
    min_price: float = 3.0                  # Penny stock filter
    max_price: float = 1000.0               # Upper bound
    allow_shorts: bool = False              # Shorts disabled by default
```

**Validation Logic:**
```python
gate = PolicyGate(limits)
ok, reason = gate.check(symbol, side, price, qty)
# Returns: (True, "ok") or (False, "exceeds_per_order_budget")
```

**Budget Tracking:**
- Per-order notional tracked immediately
- Daily budget accumulates across all orders
- Reset via `gate.reset_daily()` (called by scheduler)

**Veto Reasons:**
- `invalid_price_or_qty`
- `price_out_of_bounds`
- `shorts_disabled`
- `exceeds_per_order_budget`
- `exceeds_daily_budget`

---

### Layer 6: Backtest Engine
**Purpose:** Historical simulation with realistic fills and exit logic

**Module:** `backtest/engine.py`

**Key Class:**
```python
class Backtester:
    def __init__(
        cfg: BacktestConfig,
        get_signals: Callable,      # Strategy signal function
        fetch_bars: Callable        # Data provider
    )

    def run(symbols: List[str], outdir: str) -> Dict[str, Any]
```

**Simulation Logic:**
1. **Signal Generation:** Call strategy's `scan_signals_over_time()`
2. **Entry Fill:** next-bar open after signal timestamp
3. **Slippage:** 5 bps default (configurable)
4. **Position Sizing:** 0.7% of current cash per trade
5. **Exit Triggers:**
   - ATR stop hit (checked against bar low/high)
   - Time stop reached (5 bars)
   - Manual exit signal (future feature)
6. **P&L Calculation:** FIFO accounting with mark-to-market

**Output Artifacts:**
- `trade_list.csv` - Every entry/exit with timestamps
- `equity_curve.csv` - Daily portfolio values
- `summary.json` - Win rate, Sharpe, max DD, profit factor

**walk-forward Module:** `backtest/walk-forward.py`
- Splits data into rolling train/test windows
- Default: 252 trading days train, 63 days test
- Outputs per-split metrics for stability analysis
- HTML report generation via `scripts/aggregate_wf_report.py`

---

### Layer 7: Order Management System (OMS)
**Purpose:** Track order lifecycle and prevent duplicate submissions

**Modules:**
- `oms/order_state.py` - Order record data structures
- `oms/idempotency_store.py` - SQLite-backed duplicate prevention

**Order States:**
```python
class OrderStatus(Enum):
    PENDING = "PENDING"         # Initial state
    APPROVED = "APPROVED"       # Passed PolicyGate
    SUBMITTED = "SUBMITTED"     # Sent to broker
    FILLED = "FILLED"           # Execution confirmed
    CANCELLED = "CANCELLED"     # User/system cancel
    REJECTED = "REJECTED"       # Broker rejection
    VETOED = "VETOED"           # PolicyGate block
    CLOSED = "CLOSED"           # Terminal state
```

**Idempotency Mechanism:**
```python
store = IdempotencyStore("state/idempotency.sqlite")

# Before submission:
if store.exists(decision_id):
    return  # Skip duplicate

# After submission:
store.put(decision_id, idempotency_key)
```

**Decision ID Format:**
```
DEC_20251226_143022_AAPL_A3F91C
    ^     ^       ^     ^    ^
    |     |       |     |    â””â”€ Random hex (6 chars)
    |     |       |     â””â”€â”€â”€â”€â”€â”€ Symbol
    |     |       â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Time (HH:MM:SS)
    |     â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Date (YYYYMMDD)
    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Prefix
```

**Idempotency Key:** Same as decision_id (deterministic)

---

### Layer 8: Execution
**Purpose:** Submit orders to broker with IOC LIMIT semantics

**Module:** `execution/broker_alpaca.py`

**Key Functions:**

1. **Get Best Ask:**
```python
get_best_ask(symbol: str, timeout: int = 5) -> Optional[float]
# Fetches real-time quote from Alpaca market data
```

2. **Construct Order:**
```python
construct_decision(
    symbol: str,
    side: str,           # "long" or "short"
    qty: int,
    best_ask: float
) -> OrderRecord
# Creates OrderRecord with limit = best_ask Ã— 1.001
```

3. **Place IOC Limit:**
```python
place_ioc_limit(order: OrderRecord) -> OrderRecord
# POST /v2/orders with time_in_force="ioc"
# Returns updated order with broker_order_id and status
```

**IOC LIMIT Logic:**
- **Limit Price:** Best ask Ã— 1.001 (0.1% premium)
- **Time in Force:** Immediate-or-Cancel
- **Extended Hours:** Disabled
- **Client Order ID:** Passed as idempotency_key

**Error Handling:**
- HTTP 429: Rate limit (no retry in v1)
- HTTP 403: Insufficient buying power â†’ REJECTED
- HTTP 422: Invalid order â†’ REJECTED
- Network error: Logged, order marked REJECTED

---

### Layer 9: Core Infrastructure
**Purpose:** Logging, audit, configuration management

**Modules:**

#### 1. Structured Logging (`core/structured_log.py`)
```python
jlog(event: str, level: str = "INFO", **fields)
# Appends JSON line to logs/events.jsonl
# Format: {"ts": "2025-12-26T14:30:22", "level": "INFO",
#          "event": "order_submit", "symbol": "AAPL", ...}
```

**Log Events:**
- `runner_start`, `runner_execute`, `runner_done`
- `order_submit`, `order_reject`, `policy_veto`
- `kill_switch_active`, `skip_no_best_ask`

#### 2. Hash Chain Audit (`core/hash_chain.py`)
**Purpose:** Tamper-proof audit trail for all order submissions

```python
append_block(record: Dict[str, Any]) -> str
# Appends to state/hash_chain.jsonl with SHA256 linkage

verify_chain() -> bool
# Verifies entire chain integrity
```

**Block Structure:**
```json
{
  "prev_hash": "a3f91c...",
  "payload": {
    "decision_id": "DEC_...",
    "symbol": "AAPL",
    "qty": 10,
    "config_pin": "d4e5f6..."
  },
  "this_hash": "b7c8d9..."
}
```

**Verification Script:** `scripts/verify_hash_chain.py`

#### 3. Config Pinning (`core/config_pin.py`)
```python
sha256_file(path: Path) -> str
# Returns SHA256 hash of config/settings.json
# Stored in audit blocks to detect parameter drift
```

---

### Layer 10: Monitoring & Health
**Purpose:** System health checks and operational dashboards

**Module:** `monitor/health_endpoints.py`

**Health Server:**
```python
start_health_server(port: int = 8000) -> HTTPServer
# Starts HTTP server in background thread
```

**Endpoints:**
- `GET /readiness` â†’ `{"ready": true}`
- `GET /liveness` â†’ `{"alive": true}`

**Operational Scripts:**
- `scripts/start_health.py` - Launch health server
- `scripts/reconcile_alpaca.py` - Compare broker vs. local state
- `scripts/metrics.py` - Compute Sharpe, win rate, profit factor
- `scripts/benchmark.py` - Compare performance vs. SPY

---

## Key Entry Points

### 1. Preflight Check
**Script:** `scripts/preflight.py`

**Purpose:** Validate environment before trading

**Checks:**
1. Environment variables loaded
2. Required API keys present
3. Config file SHA256 computed
4. Alpaca `/v2/account` probe successful

**Usage:**
```bash
python scripts/preflight.py --dotenv /path/to/.env
```

**Exit Codes:**
- 0: All checks passed
- 2: Missing env keys
- 3: Config pin error
- 4: Alpaca probe failed

---

### 2. Build Universe
**Script:** `scripts/build_universe_polygon.py`

**Purpose:** Construct 900-stock universe with coverage proofs

**Process:**
1. Load candidate symbols from CSV
2. For each symbol:
   - Check options availability via Polygon
   - Fetch earliest/latest bars
   - Compute coverage years
3. Sort by coverage descending
4. Cap at 900 symbols
5. Write `optionable_liquid_900.csv` and `.full.csv`

**Usage:**
```bash
python scripts/build_universe_polygon.py \
  --candidates data/universe/optionable_liquid_candidates.csv \
  --start 2015-01-01 --end 2024-12-31 \
  --min-years 10 --cap 900 \
  --cache data/cache --concurrency 3
```

**Output:**
- `data/universe/optionable_liquid_900.csv` (900 symbols)
- `data/universe/optionable_liquid_final.full.csv` (coverage metadata)

---

### 3. Prefetch Data
**Script:** `scripts/prefetch_polygon_universe.py`

**Purpose:** Download all historical bars to cache (speeds up backtests)

**Usage:**
```bash
python scripts/prefetch_polygon_universe.py \
  --universe data/universe/optionable_liquid_900.csv \
  --start 2015-01-01 --end 2024-12-31 \
  --cache data/cache --concurrency 3
```

**Benefit:** Reduces walk-forward runtime from hours to minutes

---

### 4. walk-forward Backtest
**Script:** `scripts/run_wf_polygon.py`

**Purpose:** Rolling window validation of strategies

**Parameters:**
- `--train-days 252` (1 year training)
- `--test-days 63` (1 quarter testing)
- `--strategies rsi2,ICT,and` (comma-separated)

**Output Structure:**
```
wf_outputs/
â”œâ”€â”€ rsi2/
â”‚   â”œâ”€â”€ split_02/{trade_list.csv, equity_curve.csv, summary.json}
â”‚   â”œâ”€â”€ split_03/...
â”‚   â””â”€â”€ split_25/
â”œâ”€â”€ ICT/...
â”œâ”€â”€ and/...
â””â”€â”€ wf_summary_compare.csv
```

**Report Generation:**
```bash
python scripts/aggregate_wf_report.py \
  --wfdir wf_outputs --out wf_outputs/wf_report.html
```

---

### 5. Paper Trading
**Script:** `scripts/run_paper_trade.py`

**Purpose:** Real-time signal generation and order submission (paper account)

**Flow:**
1. Load universe (capped at 50 for paper)
2. Fetch latest bars (lookback: 540 days default)
3. Generate IBS+RSI/ICT and ICT signals
4. Filter to AND signals on most recent bar
5. Check kill switch (`state/KILL_SWITCH`)
6. For each signal:
   - Get best ask from Alpaca
   - Size position to fit $75 budget
   - Check PolicyGate
   - Submit IOC LIMIT order
   - Append to hash chain
   - Log to `logs/events.jsonl`

**Usage:**
```bash
python scripts/run_paper_trade.py \
  --universe data/universe/optionable_liquid_900.csv \
  --start 2024-06-01 --end 2025-12-26 --cap 50
```

---

### 6. Live Trading (Micro)
**Script:** `scripts/run_live_trade_micro.py`

**Purpose:** REAL MONEY trading with micro budgets

**Differences from Paper:**
- Universe capped at 10 stocks
- Requires `ALPACA_BASE_URL=https://api.alpaca.markets`
- Uses live API credentials
- Same safety mechanisms (kill switch, PolicyGate)

**Usage:**
```bash
# DANGER: REAL MONEY
python scripts/run_live_trade_micro.py \
  --universe data/universe/optionable_liquid_900.csv \
  --cap 10
```

---

### 7. 24/7 Runner
**Script:** `scripts/runner.py`

**Purpose:** Scheduled execution at market scan times

**Features:**
- Weekday filter (Mon-Fri)
- Configurable scan times (e.g., 09:35, 10:30, 15:55 ET)
- State persistence (`state/runner_last.json`)
- Prevents duplicate runs per day/time slot
- Graceful shutdown on SIGTERM

**Usage (Paper):**
```bash
python scripts/runner.py \
  --mode paper \
  --universe data/universe/optionable_liquid_900.csv \
  --cap 50 \
  --scan-times 09:35,10:30,15:55 \
  --lookback-days 540
```

**Usage (Live):**
```bash
python scripts/runner.py \
  --mode live \
  --universe data/universe/optionable_liquid_900.csv \
  --cap 10 \
  --scan-times 09:35,10:30,15:55
```

**Deployment:** See `docs/RUN_24x7.md` for Windows Task Scheduler setup

---

## Data Flow

### Signal to Execution Lifecycle

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ 1. DATA INGESTION                                               â”‚
â”‚    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                                             â”‚
â”‚    â”‚ Polygon API  â”‚ â”€â”€fetch_daily_bars_polygon()â”€â”€>             â”‚
â”‚    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                                             â”‚
â”‚           â”‚                                                      â”‚
â”‚           â–¼                                                      â”‚
â”‚    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                                             â”‚
â”‚    â”‚ CSV Cache    â”‚ data/cache/{symbol}_{start}_{end}.csv       â”‚
â”‚    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                                             â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                           â”‚
                           â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ 2. UNIVERSE FILTERING                                           â”‚
â”‚    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                                     â”‚
â”‚    â”‚ load_universe()      â”‚ â”€â”€ 900 symbols â”€â”€>                  â”‚
â”‚    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                                     â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                           â”‚
                           â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ 3. STRATEGY ENGINE                                              â”‚
â”‚    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                 â”‚
â”‚    â”‚ IBS+RSI/ICT Strategy                           â”‚                 â”‚
â”‚    â”‚  - Compute RSI(2), SMA(200), ATR(14)     â”‚                 â”‚
â”‚    â”‚  - Shift indicators by 1 bar             â”‚                 â”‚
â”‚    â”‚  - Generate signals where conditions met â”‚                 â”‚
â”‚    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                 â”‚
â”‚                           â”‚                                      â”‚
â”‚                           â–¼                                      â”‚
â”‚    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                 â”‚
â”‚    â”‚ ICT Strategy                             â”‚                 â”‚
â”‚    â”‚  - Compute ICT, SMA(200), ATR(14)        â”‚                 â”‚
â”‚    â”‚  - Shift indicators by 1 bar             â”‚                 â”‚
â”‚    â”‚  - Generate signals where conditions met â”‚                 â”‚
â”‚    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                 â”‚
â”‚                           â”‚                                      â”‚
â”‚                           â–¼                                      â”‚
â”‚    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                 â”‚
â”‚    â”‚ AND Filter (merge on timestamp/symbol)   â”‚                 â”‚
â”‚    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                 â”‚
â”‚                           â”‚                                      â”‚
â”‚                           â–¼                                      â”‚
â”‚              [List of signals for today]                        â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                           â”‚
                           â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ 4. KILL SWITCH CHECK                                            â”‚
â”‚    if state/KILL_SWITCH exists:                                 â”‚
â”‚        ABORT (log warning, exit)                                â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                           â”‚
                           â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ 5. RISK MANAGEMENT (per signal)                                 â”‚
â”‚    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                                     â”‚
â”‚    â”‚ get_best_ask()       â”‚ â”€â”€â”€ Alpaca quote API â”€â”€>            â”‚
â”‚    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                                     â”‚
â”‚              â”‚                                                   â”‚
â”‚              â–¼                                                   â”‚
â”‚    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                 â”‚
â”‚    â”‚ PolicyGate.check()                       â”‚                 â”‚
â”‚    â”‚  - Notional < $75 per order?             â”‚                 â”‚
â”‚    â”‚  - Daily total < $1,000?                 â”‚                 â”‚
â”‚    â”‚  - Price in [$3, $1000]?                 â”‚                 â”‚
â”‚    â”‚  - Shorts allowed?                       â”‚                 â”‚
â”‚    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                 â”‚
â”‚              â”‚                                                   â”‚
â”‚         â”Œâ”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”                                            â”‚
â”‚         â–¼          â–¼                                            â”‚
â”‚      [OK]      [VETO] â”€â”€> Log veto, skip                        â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                           â”‚
                           â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ 6. ORDER CONSTRUCTION                                           â”‚
â”‚    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”               â”‚
â”‚    â”‚ construct_decision()                       â”‚               â”‚
â”‚    â”‚  - decision_id = DEC_YYYYMMDD_HHMMSS_SYM_XXX â”‚            â”‚
â”‚    â”‚  - limit_price = best_ask Ã— 1.001          â”‚               â”‚
â”‚    â”‚  - qty = floor($75 / limit_price)          â”‚               â”‚
â”‚    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜               â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                           â”‚
                           â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ 7. IDEMPOTENCY CHECK                                            â”‚
â”‚    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                 â”‚
â”‚    â”‚ IdempotencyStore.exists(decision_id)?    â”‚                 â”‚
â”‚    â”‚  YES: Skip (duplicate)                   â”‚                 â”‚
â”‚    â”‚  NO: Proceed                             â”‚                 â”‚
â”‚    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                 â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                           â”‚
                           â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ 8. BROKER SUBMISSION                                            â”‚
â”‚    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                 â”‚
â”‚    â”‚ place_ioc_limit()                        â”‚                 â”‚
â”‚    â”‚  POST /v2/orders                         â”‚                 â”‚
â”‚    â”‚  {                                       â”‚                 â”‚
â”‚    â”‚    "symbol": "AAPL",                     â”‚                 â”‚
â”‚    â”‚    "qty": 10,                            â”‚                 â”‚
â”‚    â”‚    "side": "buy",                        â”‚                 â”‚
â”‚    â”‚    "type": "limit",                      â”‚                 â”‚
â”‚    â”‚    "time_in_force": "ioc",               â”‚                 â”‚
â”‚    â”‚    "limit_price": 150.15,                â”‚                 â”‚
â”‚    â”‚    "client_order_id": "DEC_..."          â”‚                 â”‚
â”‚    â”‚  }                                       â”‚                 â”‚
â”‚    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                 â”‚
â”‚              â”‚                                                   â”‚
â”‚         â”Œâ”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”                                            â”‚
â”‚         â–¼          â–¼                                            â”‚
â”‚   [SUBMITTED]  [REJECTED]                                       â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                           â”‚
                           â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ 9. AUDIT TRAIL                                                  â”‚
â”‚    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                 â”‚
â”‚    â”‚ append_block(hash_chain.jsonl)           â”‚                 â”‚
â”‚    â”‚  - decision_id, symbol, qty, price       â”‚                 â”‚
â”‚    â”‚  - config_pin (SHA256 of settings.json)  â”‚                 â”‚
â”‚    â”‚  - status, notes                         â”‚                 â”‚
â”‚    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                 â”‚
â”‚                           â”‚                                      â”‚
â”‚                           â–¼                                      â”‚
â”‚    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                 â”‚
â”‚    â”‚ jlog(events.jsonl)                       â”‚                 â”‚
â”‚    â”‚  - Structured JSON log entry             â”‚                 â”‚
â”‚    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                 â”‚
â”‚                           â”‚                                      â”‚
â”‚                           â–¼                                      â”‚
â”‚    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                 â”‚
â”‚    â”‚ IdempotencyStore.put(decision_id)        â”‚                 â”‚
â”‚    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                 â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                           â”‚
                           â–¼
                     [Order Complete]
```

---

## Safety Mechanisms

### 1. Kill Switch
**Location:** `state/KILL_SWITCH` (file-based)

**Mechanism:**
```python
# In run_paper_trade.py and run_live_trade_micro.py:
if Path('state/KILL_SWITCH').exists():
    jlog('kill_switch_active', level='WARN')
    print('KILL SWITCH active; aborting submissions.')
    sys.exit(0)
```

**Activation:**
```bash
# Emergency stop - create file
touch state/KILL_SWITCH

# Or via skill
/kill
```

**Deactivation:**
```bash
# Safe to resume
rm state/KILL_SWITCH

# Or via skill
/resume
```

**Use Cases:**
- Market crash / flash crash
- System malfunction detected
- Broker API issues
- Manual intervention needed

---

### 2. PolicyGate
**Location:** `risk/policy_gate.py`

**Budget Enforcement:**
```python
gate = PolicyGate(RiskLimits(
    max_notional_per_order=75.0,    # Canary budget
    max_daily_notional=1_000.0,     # Daily limit
    min_price=3.0,                  # Avoid penny stocks
    max_price=1000.0,               # Sanity check
    allow_shorts=False              # Shorts disabled
))

# Before every order:
ok, reason = gate.check(symbol, side, price, qty)
if not ok:
    # VETO: log and skip
    jlog('policy_veto', reason=reason, symbol=symbol)
```

**Veto Authority:** PolicyGate has final say - no human override in v1

**Budget Reset:** Manual via `gate.reset_daily()` or automatic at midnight (future feature)

---

### 3. Idempotency
**Location:** `oms/idempotency_store.py`

**Purpose:** Prevent duplicate order submissions (network retry, script re-run)

**Implementation:**
```python
store = IdempotencyStore("state/idempotency.sqlite")

# Before submission:
if store.exists(decision_id):
    order.status = OrderStatus.CLOSED
    order.notes = "duplicate_decision_id"
    return order  # Skip

# After successful submission:
store.put(decision_id, idempotency_key)
```

**Decision ID Uniqueness:**
- Timestamp: second-level precision
- Symbol: included in ID
- Random suffix: 6-char hex (16M combinations)

**Collision Probability:** Negligible for same-symbol, same-second submissions

**Cleanup:** Manual via `scripts/idempotency.py --clear-before 2025-01-01`

---

### 4. Hash Chain Audit
**Location:** `core/hash_chain.py`

**Purpose:** Tamper-proof audit trail (blockchain-style)

**Structure:**
```
Block 0: {prev_hash: null, payload: {...}, this_hash: "a3f91c..."}
Block 1: {prev_hash: "a3f91c...", payload: {...}, this_hash: "b7c8d9..."}
Block 2: {prev_hash: "b7c8d9...", payload: {...}, this_hash: "c1d2e3..."}
...
```

**Verification:**
```bash
python scripts/verify_hash_chain.py
# Output: "Hash chain valid: 1,234 blocks verified"
# Or: "Hash chain INVALID at block 567"
```

**Immutability:** Modifying any block breaks the chain (detected immediately)

**Use Case:** Regulatory audit, compliance evidence, dispute resolution

---

### 5. Lookahead Prevention
**Strategy Level:** All indicators shifted by 1 bar

**Example (IBS+RSI/ICT):**
```python
# WRONG (lookahead bias):
df['rsi2'] = rsi(df['close'], period=2)
entry_signal = df['rsi2'] <= 10

# CORRECT (no lookahead):
df['rsi2'] = rsi(df['close'], period=2)
df['rsi2_sig'] = df['rsi2'].shift(1)  # Use prior bar value
entry_signal = df['rsi2_sig'] <= 10
```

**Fill Timing:**
- Signal timestamp: Close of bar t
- Indicator values: From close of bar t-1
- Fill price: Open of bar t+1

**Enforcement:** Unit tests in `tests/unit/test_strategies.py`

---

### 6. Config Pinning
**Location:** `core/config_pin.py`

**Purpose:** Detect unauthorized parameter changes

**Mechanism:**
```python
# On each order submission:
config_pin = sha256_file('config/settings.json')
append_block({..., 'config_pin': config_pin})
```

**Detection:**
```bash
# Compare hash in audit blocks
grep config_pin state/hash_chain.jsonl | sort -u
# If > 1 unique hash: config was modified mid-session
```

**Governance:** Config changes require:
1. Update `config/settings.json`
2. Restart runner process
3. New config_pin in subsequent audit blocks

---

## Trading Strategies

- IBS+RSI (trend): channel breakout with ATR-based stop, time stop, optional R-multiple take profit. See strategies/IBS+RSI/strategy.py
- ICT Turtle Soup (mean reversion): failed breakout (liquidity sweep) with ATR/time stops and R-multiple. See strategies/ict/turtle_soup.py

## System Components

### Directory Structure

```
kobe81_traderbot/
â”œâ”€â”€ backtest/
â”‚   â”œâ”€â”€ engine.py              # Backtester core
â”‚   â””â”€â”€ walk-forward.py        # Rolling window validation
â”œâ”€â”€ config/
â”‚   â”œâ”€â”€ env_loader.py          # .env file parser
â”‚   â”œâ”€â”€ settings.json          # System parameters
â”‚   â””â”€â”€ strategies/            # Strategy configs (future)
â”œâ”€â”€ core/
â”‚   â”œâ”€â”€ config_pin.py          # SHA256 file hashing
â”‚   â”œâ”€â”€ hash_chain.py          # Audit blockchain
â”‚   â””â”€â”€ structured_log.py      # JSON logging
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ cache/                 # CSV cache for Polygon bars
â”‚   â”œâ”€â”€ providers/
â”‚   â”‚   â”œâ”€â”€ polygon_eod.py     # Polygon daily bars API
â”‚   â”‚   â””â”€â”€ multi_source.py    # Future: yfinance fallback
â”‚   â””â”€â”€ universe/
â”‚       â”œâ”€â”€ loader.py          # Symbol list management
â”‚       â””â”€â”€ optionable_liquid_900.csv  # 900 stocks
â”œâ”€â”€ docs/
â”‚   â”œâ”€â”€ README.md              # Quick start
â”‚   â”œâ”€â”€ RUN_24x7.md            # Scheduler setup
â”‚   â”œâ”€â”€ COMPLETE_ROBOT_ARCHITECTURE.md  # Layer mapping
â”‚   â””â”€â”€ TECHNICAL_ARCHITECTURE.md  # This file
â”œâ”€â”€ execution/
â”‚   â””â”€â”€ broker_alpaca.py       # Alpaca REST API
â”œâ”€â”€ logs/
â”‚   â””â”€â”€ events.jsonl           # Structured logs
â”œâ”€â”€ monitor/
â”‚   â””â”€â”€ health_endpoints.py    # HTTP health server
â”œâ”€â”€ oms/
â”‚   â”œâ”€â”€ order_state.py         # Order record enums
â”‚   â””â”€â”€ idempotency_store.py   # SQLite duplicate prevention
â”œâ”€â”€ risk/
â”‚   â””â”€â”€ policy_gate.py         # Budget enforcement
â”œâ”€â”€ scripts/                   # 80+ operational scripts
â”‚   â”œâ”€â”€ preflight.py           # Pre-trade checks
â”‚   â”œâ”€â”€ build_universe_polygon.py  # Universe builder
â”‚   â”œâ”€â”€ prefetch_polygon_universe.py  # Data prefetch
â”‚   â”œâ”€â”€ run_wf_polygon.py      # walk-forward backtest
â”‚   â”œâ”€â”€ run_backtest_polygon.py  # Single backtest
â”‚   â”œâ”€â”€ run_paper_trade.py     # Paper trading
â”‚   â”œâ”€â”€ run_live_trade_micro.py  # Live micro trading
â”‚   â”œâ”€â”€ runner.py              # 24/7 scheduler
â”‚   â”œâ”€â”€ reconcile_alpaca.py    # Broker reconciliation
â”‚   â”œâ”€â”€ verify_hash_chain.py   # Audit verification
â”‚   â””â”€â”€ [70+ more scripts...]
â”œâ”€â”€ state/
â”‚   â”œâ”€â”€ hash_chain.jsonl       # Audit blocks
â”‚   â”œâ”€â”€ idempotency.sqlite     # Duplicate tracking
â”‚   â”œâ”€â”€ runner_last.json       # Scheduler state
â”‚   â””â”€â”€ KILL_SWITCH            # Emergency stop (if exists)
â”œâ”€â”€ strategies/
â”‚   â”œâ”€â”€ IBS+RSI/
â”‚   â”‚   â”œâ”€â”€ strategy.py        # IBS+RSI/ICT implementation
â”‚   â”‚   â””â”€â”€ indicators.py      # RSI, SMA, ATR
â”‚   â””â”€â”€ ICT/
â”‚       â”œâ”€â”€ strategy.py        # ICT implementation
â”‚       â””â”€â”€ indicators.py      # ICT, SMA, ATR
â”œâ”€â”€ tests/
â”‚   â”œâ”€â”€ unit/
â”‚   â”‚   â”œâ”€â”€ test_strategies.py
â”‚   â”‚   â”œâ”€â”€ test_backtest.py
â”‚   â”‚   â”œâ”€â”€ test_core.py
â”‚   â”‚   â”œâ”€â”€ test_risk.py
â”‚   â”‚   â””â”€â”€ test_data.py
â”‚   â”œâ”€â”€ integration/
â”‚   â”‚   â””â”€â”€ test_workflow.py
â”‚   â””â”€â”€ conftest.py            # pytest fixtures
â”œâ”€â”€ wf_outputs/                # walk-forward results
â”‚   â”œâ”€â”€ rsi2/split_NN/
â”‚   â”œâ”€â”€ ICT/split_NN/
â”‚   â””â”€â”€ and/split_NN/
â”œâ”€â”€ .env                       # Environment variables (gitignored)
â”œâ”€â”€ .env.template              # Template for .env
â”œâ”€â”€ CLAUDE.md                  # Claude Code guidance
â”œâ”€â”€ README.md                  # Project overview
â””â”€â”€ requirements.txt           # Python dependencies
```

---

## Operational Modes

### Mode 1: Backtesting
**Purpose:** Historical simulation for strategy validation

**Script:** `scripts/run_backtest_polygon.py`

**Usage:**
```bash
python scripts/run_backtest_polygon.py \
  --strategy rsi2 \
  --universe data/universe/optionable_liquid_900.csv \
  --start 2020-01-01 --end 2024-12-31 \
  --cap 900 \
  --outdir backtest_outputs
```

**Outputs:**
- `backtest_outputs/trade_list.csv`
- `backtest_outputs/equity_curve.csv`
- `backtest_outputs/summary.json`

**Metrics Computed:**
- Total return, CAGR
- Win rate, profit factor
- Sharpe ratio, Sortino ratio
- Max drawdown, avg drawdown
- Trade count, avg hold time

---

### Mode 2: walk-forward Validation
**Purpose:** Rolling out-of-sample testing

**Script:** `scripts/run_wf_polygon.py`

**Parameters:**
- Training window: 252 days (1 year)
- Testing window: 63 days (1 quarter)
- Overlap: None (anchored forward)

**Process:**
1. Split data into chronological folds
2. For each fold:
   - Train on prior 252 days (parameter optimization optional)
   - Test on next 63 days (out-of-sample)
   - Record metrics
3. Aggregate statistics across all folds
4. Generate HTML report

**Usage:**
```bash
python scripts/run_wf_polygon.py \
  --universe data/universe/optionable_liquid_900.csv \
  --start 2015-01-01 --end 2024-12-31 \
  --train-days 252 --test-days 63 \
  --strategies rsi2,ICT,and \
  --outdir wf_outputs

python scripts/aggregate_wf_report.py \
  --wfdir wf_outputs --out wf_outputs/wf_report.html
```

**Stability Metrics:**
- Mean/Median/StdDev of Sharpe across splits
- Worst-case drawdown
- Consistency score (% of profitable splits)

---

### Mode 3: Paper Trading
**Purpose:** Live signal generation with simulated fills

**Script:** `scripts/run_paper_trade.py`

**Configuration:**
- Alpaca endpoint: `https://paper-api.alpaca.markets`
- Universe cap: 50 stocks (reduced for safety)
- Order budget: $75 per order
- Daily budget: $1,000

**Execution Flow:**
1. Fetch latest 540 days of data
2. Generate signals for all strategies
3. Filter to AND signals on most recent bar
4. Check kill switch
5. For each signal:
   - Get real-time best ask
   - Size position to fit budget
   - PolicyGate validation
   - Submit IOC LIMIT order
   - Log to audit chain

**Monitoring:**
```bash
# View recent logs
tail -f logs/events.jsonl

# Check order status
python scripts/orders.py --recent 10

# View positions
python scripts/positions.py
```

---

### Mode 4: Live Trading (Micro)
**Purpose:** Real-money execution with minimal risk

**Script:** `scripts/run_live_trade_micro.py`

**CRITICAL DIFFERENCES:**
- Alpaca endpoint: `https://api.alpaca.markets` (LIVE)
- Universe cap: 10 stocks (maximum)
- Order budget: $75 per order (strict)
- Daily budget: $1,000 (strict)
- Manual activation required

**Pre-Live Checklist:**
1. Paper trading successful for 30+ days
2. walk-forward validation passed
3. Reconciliation clean (no discrepancies)
4. Kill switch tested
5. Monitoring alerts configured
6. Backup/restore procedures tested

**Activation:**
```bash
# Update .env
ALPACA_BASE_URL=https://api.alpaca.markets

# Run live script
python scripts/run_live_trade_micro.py \
  --universe data/universe/optionable_liquid_900.csv \
  --cap 10
```

---

### Mode 5: 24/7 Scheduler
**Purpose:** Automated execution at market scan times

**Script:** `scripts/runner.py`

**Scheduling:**
- Scan times: 09:35, 10:30, 15:55 ET (configurable)
- Weekday filter: Monday-Friday only
- State persistence: `state/runner_last.json`
- Duplicate prevention: Won't re-run same time/day

**Loop Logic:**
```python
while True:
    now = datetime.now()
    if is_weekday(now):
        for scan_time in scan_times:
            if now >= scan_time and not already_ran(scan_time, today):
                run_submit()  # Call paper or live script
                mark_ran(scan_time, today)
    sleep(30 seconds)
```

**Deployment (Windows Task Scheduler):**
```
Action: Start program
Program: python.exe
Arguments: scripts/runner.py --mode paper --universe data/universe/optionable_liquid_900.csv
Start in: C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot
Trigger: At system startup
Conditions: Only if computer is on AC power
```

**Monitoring:**
- Health endpoint: `http://localhost:8000/liveness`
- Log tail: `tail -f logs/events.jsonl`
- Daily reconciliation: `python scripts/reconcile_alpaca.py`

---

## Evidence & Audit Trail

### Artifacts Generated

#### 1. walk-forward Outputs
**Location:** `wf_outputs/{strategy}/split_{NN}/`

**Files:**
- `trade_list.csv` - Every entry/exit with P&L
- `equity_curve.csv` - Daily portfolio values
- `summary.json` - Aggregate metrics

**Aggregated:**
- `wf_outputs/wf_summary_compare.csv` - side-by-side strategy comparison
- `wf_outputs/wf_report.html` - Interactive HTML report

#### 2. Structured Logs
**Location:** `logs/events.jsonl`

**Format:**
```json
{"ts": "2025-12-26T14:30:22.123Z", "level": "INFO",
 "event": "order_submit", "symbol": "AAPL", "qty": 10,
 "price": 150.15, "decision_id": "DEC_20251226_143022_AAPL_A3F91C"}
```

**Key Events:**
- `runner_start`, `runner_execute`, `runner_done`
- `order_submit`, `order_reject`, `policy_veto`
- `kill_switch_active`, `idempotency_duplicate`

**Query Examples:**
```bash
# All order submissions today
grep "order_submit" logs/events.jsonl | grep "2025-12-26"

# Policy vetoes
grep "policy_veto" logs/events.jsonl

# Errors
grep '"level":"ERROR"' logs/events.jsonl
```

#### 3. Hash Chain Audit
**Location:** `state/hash_chain.jsonl`

**Sample Block:**
```json
{
  "prev_hash": "a3f91c2d...",
  "payload": {
    "decision_id": "DEC_20251226_143022_AAPL_A3F91C",
    "symbol": "AAPL",
    "side": "BUY",
    "qty": 10,
    "limit_price": 150.15,
    "config_pin": "d4e5f6a7...",
    "status": "SUBMITTED",
    "notes": null
  },
  "this_hash": "b7c8d9e3..."
}
```

**Verification:**
```bash
python scripts/verify_hash_chain.py
# Output: Hash chain valid: 1,234 blocks verified
```

#### 4. Idempotency Store
**Location:** `state/idempotency.sqlite`

**Schema:**
```sql
CREATE TABLE idempotency (
    decision_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

**Query:**
```sql
SELECT * FROM idempotency
WHERE created_at >= '2025-12-26'
ORDER BY created_at DESC;
```

#### 5. Universe Proofs
**Location:** `data/universe/optionable_liquid_final.full.csv`

**Columns:**
- symbol
- has_options (boolean)
- earliest_date (YYYY-MM-DD)
- latest_date (YYYY-MM-DD)
- coverage_years (float)
- avg_daily_volume (if available)

**Validation:**
```bash
python scripts/validate_universe_coverage.py \
  --universe data/universe/optionable_liquid_900.csv \
  --min-symbols 900 --min-years 10
# Output: PASS or FAIL with details
```

---

## Development & Deployment

### Local Development

**Setup:**
```bash
# Clone repository
cd C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot

# Install dependencies
pip install -r requirements.txt

# Copy .env template
cp .env.template .env
# Edit .env with your API keys

# Run preflight checks
python scripts/preflight.py

# Run unit tests
pytest tests/unit/

# Run integration tests (slower)
pytest tests/integration/
```

**Testing Workflow:**
1. Write strategy in `strategies/{name}/`
2. Add unit tests in `tests/unit/test_strategies.py`
3. Run smoke test: `python scripts/smoke_test.py --strategy {name}`
4. Run backtest: `python scripts/run_backtest_polygon.py --strategy {name}`
5. Run walk-forward: `python scripts/run_wf_polygon.py --strategies {name}`
6. Review results before paper trading

---

### Production Deployment

**Shadow Mode (Backtest):**
```bash
# Full 10-year walk-forward
python scripts/run_wf_polygon.py \
  --universe data/universe/optionable_liquid_900.csv \
  --start 2015-01-01 --end 2024-12-31 \
  --strategies rsi2,ICT,and

# Review metrics
python scripts/aggregate_wf_report.py --wfdir wf_outputs
```

**Paper Validation (30 days minimum):**
```bash
# Daily paper trading
python scripts/runner.py --mode paper --cap 50 --once

# Weekly reconciliation
python scripts/reconcile_alpaca.py

# Monthly review
python scripts/metrics.py --since 2025-12-01
```

**Live Micro (After Paper Success):**
```bash
# Update .env to live endpoint
ALPACA_BASE_URL=https://api.alpaca.markets

# Start with 10-stock cap
python scripts/run_live_trade_micro.py --cap 10

# Monitor continuously
tail -f logs/events.jsonl
```

**Scale-Up (After 90 Days Clean):**
- Increase universe cap: 10 â†’ 25 â†’ 50
- Increase order budget: $75 â†’ $150 â†’ $300
- Increase daily budget: $1,000 â†’ $2,500 â†’ $5,000
- Update `RiskLimits` in script or config file

---

### Operational Skills (70 Available)

**Via Claude Code Interface:**

**Critical Operations:**
- `/preflight` - Pre-trade safety checks
- `/kill` - Emergency halt (creates KILL_SWITCH)
- `/resume` - Deactivate kill switch
- `/positions` - View open positions
- `/orders` - Order history
- `/reconcile` - Broker sync check
- `/audit` - Verify hash chain

**Strategy & Analysis:**
- `/backtest` - Run historical simulation
- `/wf` - walk-forward validation
- `/showdown` - Strategy comparison
- `/signals` - View generated signals

**Data Management:**
- `/universe` - Manage stock universe
- `/prefetch` - Cache historical data
- `/polygon` - Polygon API validation

**Monitoring:**
- `/status` - System health dashboard
- `/logs` - View recent events
- `/metrics` - Performance stats
- `/pnl` - Profit & loss summary

**System Utilities:**
- `/backup` - State snapshot
- `/cleanup` - Purge old logs/cache
- `/test` - Run test suite
- `/health` - Start health server

Full skill definitions: `.claude/skills/*.md`

---

### Monitoring & Alerts

**Health Endpoint:**
```bash
# Start health server
python scripts/start_health.py

# Check liveness
curl http://localhost:8000/liveness
# {"alive": true}

# Check readiness
curl http://localhost:8000/readiness
# {"ready": true}
```

**Log Monitoring:**
```bash
# Real-time tail
tail -f logs/events.jsonl

# Error count today
grep '"level":"ERROR"' logs/events.jsonl | grep $(date +%Y-%m-%d) | wc -l

# Veto rate
grep "policy_veto" logs/events.jsonl | wc -l
```

**Metrics Dashboard:**
```bash
# Performance summary
python scripts/metrics.py

# Compare vs benchmark
python scripts/benchmark.py --symbol SPY

# Quant analysis
python scripts/quant_dashboard.py
```

---

## Appendix

### System Requirements

**Software:**
- Python 3.11 or higher
- pip package manager
- Git (for version control)
- Windows 10/11, macOS, or Linux

**Python Dependencies (requirements.txt):**
```
pandas>=2.0.0
numpy>=1.24.0
requests>=2.28.0
pytest>=7.2.0
```

**API Accounts:**
- Polygon.io (Starter plan minimum)
- Alpaca (Paper + Live accounts)

**Disk Space:**
- 5 GB minimum (cache + logs)
- 20 GB recommended (10-year data)

**Network:**
- Stable internet connection
- Firewall allows HTTPS (ports 443)

---

### Glossary

**Terms:**
- **AND Filter:** Requires both IBS+RSI/ICT AND ICT signals on same bar
- **ATR:** Average True Range (volatility measure)
- **Canary Budget:** Small test allocation ($75/order) for risk mitigation
- **Hash Chain:** Blockchain-style audit trail with SHA256 linkage
- **ICT:** Internal Bar Strength = (Close - Low) / (High - Low)
- **Idempotency:** Property ensuring duplicate submissions have no effect
- **IOC:** Immediate-or-Cancel (order type)
- **Lookahead Bias:** Using future data in backtests (avoided via shift)
- **PolicyGate:** Risk management layer enforcing budgets
- **walk-forward:** Rolling out-of-sample validation method

---

### Contact & Support

**Documentation:**
- Quick Start: `README.md`
- 24/7 Setup: `docs/RUN_24x7.md`
- Architecture: `docs/COMPLETE_ROBOT_ARCHITECTURE.md`
- Claude Guidance: `CLAUDE.md`

**Operational Commands:**
- View skills: `/help` (via Claude Code)
- System status: `python scripts/preflight.py`
- Health check: `curl http://localhost:8000/liveness`

**Emergency Contacts:**
- Kill Switch: `touch state/KILL_SWITCH`
- Broker Support: Alpaca support portal
- Data Issues: Polygon.io support

---

**End of Technical Architecture Documentation**

*Generated: 2025-12-26*
*System Version: Kobe81 v1.0*
*Total Python Modules: 112*
*Total Lines of Code: ~8,500*









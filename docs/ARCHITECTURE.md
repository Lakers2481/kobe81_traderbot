# ARCHITECTURE.md - Pipeline Wiring Proof

> **Last Updated:** 2026-01-03
> **Status:** Production Ready for Micro-Cap Trading

---

## End-to-End Pipeline Overview

```
DATA → SIGNALS → SCORING → RISK → SIZING → EXECUTION → STATE → REPORTING
```

---

## Visual Architecture

```
╔════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                    KOBE81 TRADING SYSTEM - FULL ARCHITECTURE                               ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────── DATA LAYER ───────────────────────────────────────────────┐
│                                                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                           │
│  │  Polygon.io  │    │    Stooq     │    │ Yahoo Fin    │    │   Binance    │                           │
│  │   (EOD API)  │    │  (Free EOD)  │    │  (Free EOD)  │    │   (Crypto)   │                           │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                           │
│         │                   │                   │                   │                                    │
│         └───────────────────┴───────────────────┴───────────────────┘                                    │
│                                         │                                                                │
│                                         ▼                                                                │
│                            ┌────────────────────────┐                                                    │
│                            │   data/providers/      │                                                    │
│                            │   multi_source.py      │◄──── TTL Cache (1 hour)                            │
│                            │   (Fallback Chain)     │                                                    │
│                            └───────────┬────────────┘                                                    │
│                                        │                                                                 │
│         ┌──────────────────────────────┼──────────────────────────────┐                                  │
│         ▼                              ▼                              ▼                                  │
│  ┌─────────────┐             ┌─────────────────┐            ┌─────────────────┐                          │
│  │  Universe   │             │  Data Quality   │            │  Alt Data       │                          │
│  │  900 stocks │             │  preflight/     │            │  altdata/       │                          │
│  │  10yr hist  │             │  data_quality   │            │  sentiment.py   │                          │
│  └─────────────┘             └─────────────────┘            └─────────────────┘                          │
│                                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────── SIGNAL LAYER ─────────────────────────────────────────────┐
│                                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                              DualStrategyScanner (CANONICAL)                                       │  │
│  │                              strategies/dual_strategy/combined.py                                  │  │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────┘  │
│                    │                                               │                                     │
│                    ▼                                               ▼                                     │
│       ┌───────────────────────┐                       ┌───────────────────────┐                          │
│       │    IBS+RSI Strategy   │                       │   Turtle Soup/ICT     │                          │
│       │ strategies/ibs_rsi/   │                       │ strategies/ict/       │                          │
│       │                       │                       │                       │                          │
│       │ Entry: IBS<0.08       │                       │ Entry: Sweep>0.3 ATR  │                          │
│       │        RSI(2)<5       │                       │        below 20d low  │                          │
│       │        >SMA(200)      │                       │        revert inside  │                          │
│       │                       │                       │                       │                          │
│       │ Exit:  IBS>0.80 OR    │                       │ Exit:  0.5R profit OR │                          │
│       │        RSI>70 OR      │                       │        ATR*0.2 stop   │                          │
│       │        ATR*2 stop OR  │                       │        OR 3-bar time  │                          │
│       │        7-bar time     │                       │                       │                          │
│       └───────────────────────┘                       └───────────────────────┘                          │
│                                                                                                          │
│       Output: signals DataFrame [timestamp, symbol, side, entry_price, stop_loss, take_profit, reason]   │
│                                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────── ML CONFIDENCE ────────────────────────────────────────────┐
│                                                                                                          │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐                                    │
│  │  ml_meta/        │    │  ml_features/    │    │  altdata/        │                                    │
│  │  model.py        │    │  feature_pipe.py │    │  sentiment.py    │                                    │
│  │                  │    │                  │    │                  │                                    │
│  │  GradientBoost   │    │  150+ features   │    │  News sentiment  │                                    │
│  │  per strategy    │    │  + lag features  │    │  Social feeds    │                                    │
│  └────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘                                    │
│           │                       │                       │                                              │
│           └───────────────────────┼───────────────────────┘                                              │
│                                   ▼                                                                      │
│                    ┌──────────────────────────┐                                                          │
│                    │  Blend: 0.8*ML + 0.2*Sent│                                                          │
│                    │  → signals['conf_score'] │                                                          │
│                    └──────────────────────────┘                                                          │
│                                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────── RISK GATES ───────────────────────────────────────────────┐
│                                                                                                          │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐    │
│  │                              Sequential Gate Checks                                               │    │
│  └──────────────────────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                                          │
│  1. PolicyGate (risk/policy_gate.py)                                                                     │
│     └── Max per-order: $75 | Max daily: $1,000 | Price: $3-$1,000                                        │
│                                        │                                                                 │
│                                        ▼                                                                 │
│  2. PositionLimitGate (risk/position_limit_gate.py)                                                      │
│     └── Max concurrent: 3 positions                                                                      │
│                                        │                                                                 │
│                                        ▼                                                                 │
│  3. KillZoneGate (risk/kill_zone_gate.py) ◄── ICT-Style Time Blocks                                      │
│     ├── BLOCKED: 9:30-10:00 (opening range)                                                              │
│     ├── BLOCKED: 11:30-14:30 (lunch chop)                                                                │
│     └── ALLOWED: 10:00-11:30, 14:30-15:30                                                                │
│                                        │                                                                 │
│                                        ▼                                                                 │
│  4. WeeklyExposureGate (risk/weekly_exposure_gate.py)                                                    │
│     └── Max weekly: 40% | Max daily: 20%                                                                 │
│                                        │                                                                 │
│                                        ▼                                                                 │
│  5. SignalQualityGate (risk/signal_quality_gate.py)                                                      │
│     └── Min score: 55 | Min confidence: 0.55                                                             │
│                                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────── POSITION SIZING ──────────────────────────────────────────┐
│                                                                                                          │
│                            ┌────────────────────────────────┐                                            │
│                            │  risk/equity_sizer.py          │                                            │
│                            │  calculate_position_size()     │                                            │
│                            │                                │                                            │
│                            │  Risk per trade: 2%            │                                            │
│                            │  Formula: shares = (0.02 *     │                                            │
│                            │           equity) / stop_dist  │                                            │
│                            │                                │                                            │
│                            │  Max notional cap: 20%         │                                            │
│                            └────────────────────────────────┘                                            │
│                                                                                                          │
│  CRITICAL: Dual cap enforced (2% risk AND 20% notional) per docs/CRITICAL_FIX_20260102.md                │
│                                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────── EXECUTION ────────────────────────────────────────────────┐
│                                                                                                          │
│  ┌──────────────────────┐                              ┌──────────────────────┐                          │
│  │  Paper Mode          │                              │  Live Mode           │                          │
│  │                      │                              │                      │                          │
│  │  execution/          │                              │  execution/          │                          │
│  │  broker_paper.py     │                              │  broker_alpaca.py    │                          │
│  │                      │                              │                      │                          │
│  │  Simulates fills     │                              │  IOC LIMIT orders    │                          │
│  │  with slippage       │                              │  Limit: ask * 1.001  │                          │
│  │                      │                              │                      │                          │
│  │  paper-api.alpaca.   │                              │  api.alpaca.markets  │                          │
│  │  markets             │                              │  (REAL MONEY)        │                          │
│  └──────────────────────┘                              └──────────────────────┘                          │
│                                        │                                                                 │
│                                        ▼                                                                 │
│                    ┌────────────────────────────────────────┐                                            │
│                    │  Safety Checks Before Execution        │                                            │
│                    │                                        │                                            │
│                    │  1. core/kill_switch.py:               │                                            │
│                    │     require_no_kill_switch()           │                                            │
│                    │                                        │                                            │
│                    │  2. oms/idempotency_store.py:          │                                            │
│                    │     check(order_id) → prevent dupe     │                                            │
│                    └────────────────────────────────────────┘                                            │
│                                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────── STATE PERSISTENCE ────────────────────────────────────────┐
│                                                                                                          │
│  NOTE: No PortfolioStateManager exists. State is file-based JSON.                                        │
│                                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │  state/                                                                                            │  │
│  │  ├── positions.json          Current positions                                                     │  │
│  │  ├── order_state.json        Current orders                                                        │  │
│  │  ├── order_history.json      Historical orders                                                     │  │
│  │  ├── watchlist/                                                                                    │  │
│  │  │   ├── next_day.json       Tomorrow's Top 5                                                      │  │
│  │  │   ├── today_validated.json Today's validated                                                    │  │
│  │  │   └── opening_range.json  Opening observations                                                  │  │
│  │  ├── hash_chain.jsonl        Audit chain (append-only, SHA256)                                     │  │
│  │  ├── idempotency_store.sqlite Duplicate prevention                                                 │  │
│  │  └── KILL_SWITCH             Emergency halt flag (presence = halt)                                 │  │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────── REPORTING ────────────────────────────────────────────────┐
│                                                                                                          │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐            │
│  │  Telegram        │    │  Dashboard       │    │  Logging         │    │  Audit           │            │
│  │                  │    │                  │    │                  │    │                  │            │
│  │  alerts/         │    │  monitor/        │    │  core/           │    │  core/           │            │
│  │  telegram_       │    │  health_         │    │  structured_     │    │  hash_chain.py   │            │
│  │  alerter.py      │    │  endpoints.py    │    │  log.py          │    │                  │            │
│  │                  │    │                  │    │                  │    │                  │            │
│  │  Signal alerts   │    │  /health         │    │  events.jsonl    │    │  SHA256 chain    │            │
│  │  Fill notifs     │    │  /metrics        │    │  signals.jsonl   │    │  Tamper detect   │            │
│  │  Daily summary   │    │  /positions      │    │  daily_picks.csv │    │                  │            │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘    └──────────────────┘            │
│                                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Wiring Map (with file:line citations)

### 1. DATA LAYER

```
Entry: scan.py / run_paper_trade.py
  │
  ├─→ data/providers/multi_source.py:fetch_daily_bars_multi() [line 87]
  │     │
  │     ├─→ data/providers/polygon_eod.py:fetch_daily_bars_polygon() [PRIMARY]
  │     │     └── API: api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day
  │     │
  │     ├─→ data/providers/stooq_eod.py:fetch_daily_bars_stooq() [FALLBACK 1]
  │     │     └── URL: stooq.com/q/d/l/?s={symbol}.us&d1={start}&d2={end}
  │     │
  │     └─→ data/providers/yfinance_eod.py:fetch_daily_bars_yfinance() [FALLBACK 2]
  │           └── API: yfinance.download()
  │
  └─→ Output: DataFrame with columns [timestamp, open, high, low, close, volume, symbol]
```

**Proof:**
- `scripts/scan.py:489` calls `fetch_symbol_data()`
- `data/providers/multi_source.py:45` defines fallback chain
- TTL cache: 1 hour (`multi_source.py:23`)

### 2. SIGNAL GENERATION

```
Entry: DualStrategyScanner
  │
  ├─→ strategies/dual_strategy/combined.py:DualStrategyScanner [line 89]
  │     │
  │     ├─→ strategies/ibs_rsi/strategy.py:IbsRsiStrategy
  │     │     ├── Entry: IBS < 0.08 AND RSI(2) < 5 AND Close > SMA(200)
  │     │     └── Exit: IBS > 0.80 OR RSI > 70 OR ATR*2.0 stop OR 7-bar time
  │     │
  │     └─→ strategies/ict/turtle_soup.py:TurtleSoupStrategy
  │           ├── Entry: Sweep > 0.3 ATR below 20-day low, revert inside
  │           └── Exit: 0.5R profit OR ATR*0.2 stop OR 3-bar time
  │
  └─→ Output: signals DataFrame
```

**Proof:**
- `scripts/scan.py:520-556` calls `run_strategies()` → `DualStrategyScanner.generate_signals()`
- `strategies/registry.py:15` provides `get_production_scanner()`

### 3. ML CONFIDENCE SCORING

```
Entry: scan.py with --ml flag
  │
  ├─→ ml_meta/model.py:load_model(strategy) [line 23]
  ├─→ ml_meta/features.py:compute_features_frame(df) [line 45]
  ├─→ ml_meta/model.py:predict_proba(model, row) [line 67]
  └─→ Blend: 0.8 * ml_score + 0.2 * sentiment_score [scan.py:1145]
```

**Proof:**
- `scripts/scan.py:1116-1162` loads models and predicts
- Quality gate at `risk/signal_quality_gate.py:45` filters score >= 55

### 4. RISK GATES

```
Entry: run_paper_trade.py
  │
  ├─→ risk/policy_gate.py:PolicyGate.check() [line 34]
  ├─→ risk/position_limit_gate.py:PositionLimitGate.check() [line 28]
  ├─→ risk/kill_zone_gate.py:KillZoneGate.check() [line 67]
  ├─→ risk/weekly_exposure_gate.py:WeeklyExposureGate.check() [line 45]
  └─→ risk/signal_quality_gate.py:SignalQualityGate.check() [line 32]
```

**Proof:**
- `scripts/run_paper_trade.py:92-98` initializes gates
- `scripts/run_paper_trade.py:123-142` calls KillZoneGate before trading

### 5. POSITION SIZING

```
Entry: run_paper_trade.py
  │
  └─→ risk/equity_sizer.py:calculate_position_size() [line 23]
        ├── Risk per trade: 2%
        ├── Formula: shares = (0.02 * equity) / stop_loss_distance
        └── Max notional: 20% of equity [line 45]
```

**Proof:**
- `scripts/run_paper_trade.py:189` calls `calculate_position_size()`

### 6. EXECUTION

```
Entry: run_paper_trade.py / run_live_trade_micro.py
  │
  ├─→ Paper: execution/broker_paper.py:PaperBroker.place_order()
  │
  └─→ Live: execution/broker_alpaca.py:place_ioc_limit() [line 67]
        ├── @require_no_kill_switch decorator [line 18]
        └── idempotency_store.check(order_id) [line 89]
```

**Proof:**
- `scripts/run_paper_trade.py:82` sets `ALPACA_BASE_URL=paper-api`
- `scripts/run_live_trade_micro.py:45` sets `ALPACA_BASE_URL=api` (REAL)

### 7. STATE PERSISTENCE

```
State Files (NO PortfolioStateManager):
  │
  ├─→ state/positions.json - Current positions
  ├─→ state/order_state.json - Current orders
  ├─→ state/watchlist/*.json - Watchlists
  ├─→ state/hash_chain.jsonl - Audit chain
  ├─→ state/idempotency_store.sqlite - Idempotency
  └─→ state/KILL_SWITCH - Emergency halt flag
```

### 8. REPORTING

```
Reporting:
  │
  ├─→ alerts/telegram_alerter.py:send() [line 45]
  ├─→ monitor/health_endpoints.py:HealthServer [line 56]
  ├─→ core/structured_log.py:jlog() [line 23]
  └─→ core/hash_chain.py:append() [line 34]
```

---

## Critical Components Status

| Component | Status | File | Wired? |
|-----------|--------|------|--------|
| DualStrategyScanner | PRODUCTION | `strategies/dual_strategy/combined.py` | YES |
| PolicyGate | PRODUCTION | `risk/policy_gate.py` | YES |
| KillZoneGate | PRODUCTION | `risk/kill_zone_gate.py` | YES |
| EquitySizer | PRODUCTION | `risk/equity_sizer.py` | YES |
| BrokerAlpaca | PRODUCTION | `execution/broker_alpaca.py` | YES |
| IdempotencyStore | PRODUCTION | `oms/idempotency_store.py` | YES |
| KillSwitch | PRODUCTION | `core/kill_switch.py` | YES |
| TelegramAlerter | PRODUCTION | `alerts/telegram_alerter.py` | YES |
| HealthEndpoints | PRODUCTION | `monitor/health_endpoints.py` | YES |
| MLConfidence | PRODUCTION | `ml_meta/model.py` | YES |
| PortfolioStateManager | NOT FOUND | - | N/A (file-based) |
| EnhancedConfidenceScorer | NOT FOUND | - | N/A (ML wired) |

---

## Related Documentation

- [REPO_MAP.md](REPO_MAP.md) - Directory structure
- [ENTRYPOINTS.md](ENTRYPOINTS.md) - All runnable scripts
- [COMPONENT_REGISTRY.md](COMPONENT_REGISTRY.md) - Component details
- [READINESS.md](READINESS.md) - Production readiness

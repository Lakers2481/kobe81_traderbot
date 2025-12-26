# PROJECT CONTEXT: Kobe81 Trading Bot

> **Purpose:** This document provides complete context for any AI or developer to understand and continue work on this project.

---

## What Is This Project?

**Kobe81** is a production-grade algorithmic trading system implementing institutional-quality mean-reversion strategies. Named after Kobe Bryant's legendary 81-point game, it aims for consistent, disciplined execution.

**Repository:** https://github.com/Lakers2481/kobe81_traderbot
**Location:** `C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot`
**Environment:** `C:\Users\Owner\OneDrive\Desktop\GAME_PLAN_2K28\.env`

---

## Current Status (2025-12-26)

| Component | Status | Notes |
|-----------|--------|-------|
| Core Architecture | COMPLETE | 10-layer system fully implemented |
| Strategies | COMPLETE | RSI-2, IBS, AND filter |
| Backtesting | COMPLETE | Walk-forward validated |
| Universe | COMPLETE | 950 stocks, 10Y coverage |
| Paper Trading | READY | Micro budgets enforced |
| Live Trading | READY | Pending paper validation |
| CI/CD | PASSING | GitHub Actions (63 tests) |
| Documentation | COMPLETE | Architecture + onboarding docs |

---

## 10-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 10: RUNNER         scripts/runner.py                  │
│           24/7 scheduler, scan times, state persistence     │
├─────────────────────────────────────────────────────────────┤
│ Layer 9: MONITOR         monitor/health_endpoints.py        │
│          Health checks, reconciliation, metrics             │
├─────────────────────────────────────────────────────────────┤
│ Layer 8: CORE            core/hash_chain.py, structured_log │
│          Audit trail, JSON logging, config pinning          │
├─────────────────────────────────────────────────────────────┤
│ Layer 7: EXECUTION       execution/broker_alpaca.py         │
│          IOC LIMIT orders, Alpaca API integration           │
├─────────────────────────────────────────────────────────────┤
│ Layer 6: OMS             oms/order_state.py, idempotency    │
│          Order lifecycle, duplicate prevention              │
├─────────────────────────────────────────────────────────────┤
│ Layer 5: RISK            risk/policy_gate.py                │
│          $75/order, $1k/day, kill switch, price bounds      │
├─────────────────────────────────────────────────────────────┤
│ Layer 4: BACKTEST        backtest/engine.py, walk_forward   │
│          Historical simulation, rolling validation          │
├─────────────────────────────────────────────────────────────┤
│ Layer 3: STRATEGY        strategies/connors_rsi2, ibs       │
│          Signal generation, lookahead prevention            │
├─────────────────────────────────────────────────────────────┤
│ Layer 2: UNIVERSE        data/universe/loader.py            │
│          950 stocks, optionable, liquid, 10Y coverage       │
├─────────────────────────────────────────────────────────────┤
│ Layer 1: DATA            data/providers/polygon_eod.py      │
│          Polygon API, CSV caching, rate limiting            │
└─────────────────────────────────────────────────────────────┘
```

---

## Trading Strategies

### 1. Connors RSI-2 (Primary)
- **Entry:** RSI(2) ≤ 10 AND Close > SMA(200)
- **Exit:** RSI(2) ≥ 70 OR ATR(14)×2 stop OR 5-bar timeout
- **Expected:** 58-62% win rate, Sharpe 1.2-1.8

### 2. IBS (Internal Bar Strength)
- **Entry:** IBS < 0.2 AND Close > SMA(200)
- **Exit:** ATR(14)×2 stop OR 5-bar timeout
- **Expected:** 54-58% win rate, Sharpe 1.0-1.5

### 3. AND Filter (Combined)
- **Entry:** BOTH RSI-2 AND IBS signal on same bar
- **Higher selectivity:** 60-65% win rate, fewer trades

---

## Key Files Reference

### Configuration
- `config/settings.json` - Global settings
- `config/strategies/connors_rsi2.yaml` - RSI-2 parameters
- `config/strategies/ibs.yaml` - IBS parameters
- `.env` - API keys (POLYGON, ALPACA) - NOT in git

### Core Modules
- `strategies/connors_rsi2/strategy.py` - RSI-2 implementation
- `strategies/ibs/strategy.py` - IBS implementation
- `backtest/engine.py` - Backtesting engine
- `execution/broker_alpaca.py` - Broker integration
- `risk/policy_gate.py` - Risk controls

### Entry Points
- `scripts/preflight.py` - Pre-deployment validation
- `scripts/run_wf_polygon.py` - Walk-forward testing
- `scripts/run_paper_trade.py` - Paper trading
- `scripts/run_live_trade_micro.py` - Live micro trading
- `scripts/runner.py` - 24/7 scheduler

### Data
- `data/universe/optionable_liquid_final.csv` - 950 stocks
- `data/cache/` - Polygon data cache (203 MB)

### State
- `state/hash_chain.jsonl` - Audit trail
- `state/idempotency.sqlite` - Duplicate prevention
- `state/positions.json` - Current positions
- `logs/events.jsonl` - Structured logs

---

## Safety Mechanisms

1. **Kill Switch:** `touch state/KILL_SWITCH` halts all trading
2. **PolicyGate:** $75/order max, $1k/day max, $3-$1000 price bounds
3. **Idempotency:** SQLite prevents duplicate order submissions
4. **Hash Chain:** Tamper-proof audit trail
5. **Lookahead Prevention:** All indicators shifted by 1 bar

---

## How to Run

### 1. Preflight Check
```bash
python scripts/preflight.py --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env
```

### 2. Walk-Forward Validation
```bash
python scripts/run_wf_polygon.py \
  --universe data/universe/optionable_liquid_final.csv \
  --start 2015-01-01 --end 2024-12-31 \
  --train-days 252 --test-days 63 \
  --cap 900 --outdir wf_outputs \
  --cache data/cache \
  --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env
```

### 3. Paper Trading
```bash
python scripts/run_paper_trade.py \
  --universe data/universe/optionable_liquid_final.csv \
  --cap 50 \
  --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env
```

### 4. 24/7 Runner
```bash
python scripts/runner.py \
  --scan-times 09:35,10:30,15:55 \
  --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env
```

---

## Recent Changes

| Commit | Description |
|--------|-------------|
| `17ce36a` | Fixed 55 import paths (configs → config) |
| `9a61a5b` | Added CI/CD workflow, fixed tests |
| `2350278` | Added 70 Claude skills |

---

## Next Steps (Roadmap)

1. **Paper Trading Phase** (30+ days)
   - Run paper trading with micro budgets
   - Monitor win rate, Sharpe, drawdown
   - Verify reconciliation daily

2. **Live Micro Phase**
   - Graduate to live after paper validation
   - Start with 10 stocks, $75/order
   - Scale up based on performance

3. **Optional Enhancements**
   - Commission modeling
   - Short execution support
   - Additional strategies

---

## Required Environment Variables

```bash
POLYGON_API_KEY=your_polygon_key
ALPACA_API_KEY_ID=your_alpaca_key
ALPACA_API_SECRET_KEY=your_alpaca_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # or https://api.alpaca.markets for live
```

---

## Test Coverage

- **Unit Tests:** 63 tests passing
- **CI/CD:** GitHub Actions on push/PR
- **Strategies:** Signal generation tested
- **Risk:** PolicyGate budget tests
- **Core:** Hash chain, idempotency tests

Run tests:
```bash
pytest tests/unit -v
```

---

## File Structure Overview

```
kobe81_traderbot/
├── strategies/          # Trading strategies (RSI-2, IBS)
├── backtest/            # Backtesting engine
├── data/                # Data providers, universe
├── execution/           # Broker integration
├── risk/                # Risk management
├── oms/                 # Order management
├── core/                # Infrastructure (logs, audit)
├── monitor/             # Health monitoring
├── config/              # Configuration files
├── scripts/             # 78 operational scripts
├── tests/               # Unit + integration tests
├── .claude/skills/      # 70 Claude AI skills
├── docs/                # Documentation
├── state/               # Runtime state
├── logs/                # Event logs
└── wf_outputs/          # Walk-forward results
```

---

## Contact / Support

- **GitHub Issues:** https://github.com/Lakers2481/kobe81_traderbot/issues
- **Documentation:** See `docs/` folder

---

*Last Updated: 2025-12-26*

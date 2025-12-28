**Alignment Note**: Canonical setup uses IBS+RSI + ICT Turtle Soup only, with a 900-symbol universe. README.md and AI_HANDOFF_PROMPT.md are the source of truth for commands. Some legacy references remain here for historical context.

# PROJECT CONTEXT: Kobe81 Trading Bot

> **Purpose:** This document provides complete context for any AI or developer to underst  continue work on this project.

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
| Strategies | COMPLETE | IBS+RSI, ICT Turtle Soup,  filter |
| Backtesting | COMPLETE | Walk-forward validated |
| Universe | COMPLETE | 900 stocks, 10Y coverage |
| Paper Trading | READY | Micro budgets enforced |
| Live Trading | READY | Pending paper validation |
| CI/CD | PASSING | GitHub Actions (63 tests) |
| Documentation | COMPLETE | Architecture + onboarding docs |

---

## 10-Layer Architecture

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ Layer 10: RUNNER         scripts/scheduler_kobe.py                  â”‚
â”‚           24/7 scheduler, scan times, state persistence     â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚ Layer 9: MONITOR         monitor/health_endpoints.py        â”‚
â”‚          Health checks, reconciliation, metrics             â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚ Layer 8: CORE            core/hash_chain.py, structured_log â”‚
â”‚          Audit trail, JSON logging, config pinning          â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚ Layer 7: EXECUTION       execution/broker_alpaca.py         â”‚
â”‚          IOC LIMIT orders, Alpaca API integration           â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚ Layer 6: OMS             oms/order_state.py, idempotency    â”‚
â”‚          Order lifecycle, duplicate prevention              â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚ Layer 5: RISK            risk/policy_gate.py                â”‚
â”‚          $75/order, $1k/day, kill switch, price bounds      â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚ Layer 4: BACKTEST        backtest/engine.py, walk_forward   â”‚
â”‚          Historical simulation, rolling validation          â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚ Layer 3: STRATEGY        strategies/connors_IBS+RSI, ICT Turtle Soup       â”‚
â”‚          Signal generation, lookahead prevention            â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚ Layer 2: UNIVERSE        data/universe/loader.py            â”‚
â”‚          900 stocks, optionable, liquid, 10Y coverage       â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚ Layer 1: DATA            data/providers/polygon_eod.py      â”‚
â”‚          Polygon API, CSV caching, rate limiting            â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

---

## Trading Strategies

### 1. IBS+RSI (Trend)\n- Entry: Breakout above IBS+RSI high (e.g., 20–55)\n- Exit: ATR-based stop, time stop, optional R-multiple take profit
- **Entry:** RSI(2) â‰¤ 10  Close > SMA(200)
- **Exit:** RSI(2) â‰¥ 70 OR ATR(14)Ã—2 stop OR 5-bar timeout
- **Expected:** 58-62% win rate, Sharpe 1.2-1.8

### 2. ICT Turtle Soup (Mean Reversion)\n- Entry: Failed breakout (liquidity sweep) against prior extreme\n- Exit: ATR-based stop, time stop; R-multiple target
- **Entry:** ICT Turtle Soup < 0.2  Close > SMA(200)
- **Exit:** ATR(14)Ã—2 stop OR 5-bar timeout
- **Expected:** 54-58% win rate, Sharpe 1.0-1.5

### 3.  Filter (Combined)
- **Entry:** BOTH IBS+RSI  ICT Turtle Soup signal on same bar
- **Higher selectivity:** 60-65% win rate, fewer trades

---

## Key Files Reference

### Configuration
- `config/settings.json` - Global settings
- `config/strategies/connors_IBS+RSI.yaml` - IBS+RSI parameters
- `config/strategies/ICT Turtle Soup.yaml` - ICT Turtle Soup parameters
- `.env` - API keys (POLYGON, ALPACA) - NOT in git

### Core Modules
- `strategies/connors_IBS+RSI/strategy.py` - IBS+RSI implementation
- `strategies/ICT Turtle Soup/strategy.py` - ICT Turtle Soup implementation
- `backtest/engine.py` - Backtesting engine
- `execution/broker_alpaca.py` - Broker integration
- `risk/policy_gate.py` - Risk controls

### Entry Points
- `scripts/preflight.py` - Pre-deployment validation
- `scripts/run_wf_polygon.py` - Walk-forward testing
- `scripts/run_paper_trade.py` - Paper trading
- `scripts/run_live_trade_micro.py` - Live micro trading
- `scripts/scheduler_kobe.py` - 24/7 scheduler

### Data
- `data/universe/optionable_liquid_900.csv` - 900 stocks
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
  --universe data/universe/optionable_liquid_900.csv \
  --start 2015-01-01 --end 2024-12-31 \
  --train-days 252 --test-days 63 \
  --cap 900 --outdir wf_outputs \
  --cache data/cache \
  --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env
```

### 3. Paper Trading
```bash
python scripts/run_paper_trade.py \
  --universe data/universe/optionable_liquid_900.csv \
  --cap 50 \
  --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env
```

### 4. 24/7 Runner
```bash
python scripts/scheduler_kobe.py \
  --scan-times 09:35,10:30,15:55 \
  --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env
```

---

## Recent Changes

| Commit | Description |
|--------|-------------|
| `17ce36a` | Fixed 55 import paths (configs â†’ config) |
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
â”œâ”€â”€ strategies/          # Trading strategies (IBS+RSI, ICT Turtle Soup)
â”œâ”€â”€ backtest/            # Backtesting engine
â”œâ”€â”€ data/                # Data providers, universe
â”œâ”€â”€ execution/           # Broker integration
â”œâ”€â”€ risk/                # Risk management
â”œâ”€â”€ oms/                 # Order management
â”œâ”€â”€ core/                # Infrastructure (logs, audit)
â”œâ”€â”€ monitor/             # Health monitoring
â”œâ”€â”€ config/              # Configuration files
â”œâ”€â”€ scripts/             # 78 operational scripts
â”œâ”€â”€ tests/               # Unit + integration tests
â”œâ”€â”€ .claude/skills/      # 70 Claude AI skills
â”œâ”€â”€ docs/                # Documentation
â”œâ”€â”€ state/               # Runtime state
â”œâ”€â”€ logs/                # Event logs
â””â”€â”€ wf_outputs/          # Walk-forward results
```

---

## Contact / Support

- **GitHub Issues:** https://github.com/Lakers2481/kobe81_traderbot/issues
- **Documentation:** See `docs/` folder

---

*Last Updated: 2025-12-26*






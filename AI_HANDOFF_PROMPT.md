# AI Handoff Prompt for Kobe81 Trading Bot

> **Instructions:** Copy everything below the line and paste it as your first message to a new AI assistant.

---

## COPY FROM HERE ↓

I'm continuing work on the Kobe81 algorithmic trading bot. Please read the project context below to understand where we are.

### Project Location
- **Root:** `C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot`
- **GitHub:** https://github.com/Lakers2481/kobe81_traderbot
- **Environment File:** `C:\Users\Owner\OneDrive\Desktop\GAME_PLAN_2K28\.env`

### What Is Kobe81?
A production-grade algorithmic trading system implementing:
- **RSI-2 Strategy:** Mean reversion on RSI(2) ≤ 10, exit ≥ 70
- **IBS Strategy:** Internal Bar Strength < 0.2 entries
- **AND Filter:** Both strategies agree for higher selectivity

### Architecture (10 Layers)
1. DATA - Polygon API, CSV caching
2. UNIVERSE - 950 optionable liquid stocks, 10Y coverage
3. STRATEGY - Signal generation with lookahead prevention
4. BACKTEST - Walk-forward validation
5. RISK - PolicyGate ($75/order, $1k/day limits)
6. OMS - Order management, idempotency
7. EXECUTION - Alpaca broker, IOC LIMIT orders
8. CORE - Hash chain audit, structured logging
9. MONITOR - Health endpoints
10. RUNNER - 24/7 scheduler

### Current Status (as of 2025-12-26)
- All 10 layers implemented and tested
- 63 unit tests passing in CI/CD
- Import paths fixed (configs → config)
- Universe: 950 stocks validated
- Documentation complete

### Key Files to Know
- `strategies/connors_rsi2/strategy.py` - Main strategy
- `backtest/engine.py` - Backtesting
- `execution/broker_alpaca.py` - Broker integration
- `risk/policy_gate.py` - Risk controls
- `scripts/runner.py` - 24/7 scheduler
- `config/settings.json` - Global settings

### Safety Mechanisms
- Kill switch: `state/KILL_SWITCH` file
- PolicyGate: $75/order, $1k/day max
- Idempotency: SQLite duplicate prevention
- Hash chain: Tamper-proof audit trail

### How to Run
```bash
# Preflight check
python scripts/preflight.py --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

# Walk-forward validation
python scripts/run_wf_polygon.py --universe data/universe/optionable_liquid_final.csv --start 2015-01-01 --end 2024-12-31 --train-days 252 --test-days 63 --cap 900 --outdir wf_outputs --cache data/cache --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

# Paper trading
python scripts/run_paper_trade.py --universe data/universe/optionable_liquid_final.csv --cap 50 --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env
```

### Environment Variables Needed
```
POLYGON_API_KEY=xxx
ALPACA_API_KEY_ID=xxx
ALPACA_API_SECRET_KEY=xxx
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

### Recent Commits
- `17ce36a` - Fixed 55 import paths (configs → config)
- `9a61a5b` - Added CI/CD workflow, fixed tests

### Documentation
Read these files for full context:
- `PROJECT_CONTEXT.md` - Complete project overview
- `docs/COMPLETE_ROBOT_ARCHITECTURE.md` - Layer details
- `docs/RUN_24x7.md` - 24/7 deployment guide

### What I Need Help With
[DESCRIBE YOUR SPECIFIC TASK HERE]

---

## END COPY ↑

---

## Notes for Human User

When starting a new AI session:
1. Copy the text between "COPY FROM HERE" and "END COPY"
2. Paste it as your first message
3. Replace "[DESCRIBE YOUR SPECIFIC TASK HERE]" with your actual request
4. The AI will have full context to continue work

For even faster onboarding, you can also say:
> "Read PROJECT_CONTEXT.md at C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot to understand this project"

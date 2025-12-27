# AI Handoff Prompt for Kobe81 Trading Bot

> Instructions: Copy everything below the line and paste it as your first message to a new AI assistant.

---

## COPY FROM HERE

I'm continuing work on the Kobe81 algorithmic trading bot. Please read the project context below to understand where we are.

### Project Location
- Root: `C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot`
- GitHub: https://github.com/Lakers2481/kobe81_traderbot
- Environment File: `./.env`

### What Is Kobe81?
A production-grade algorithmic trading system implementing exactly two strategies:
- Donchian Breakout (trend-following): Channel breakout with ATR-based stop, time stop, optional R-multiple take profit.
- ICT Turtle Soup (mean reversion): Failed breakout (liquidity sweep) reversion with R-multiple target and time stop.

Selection/TOPN ranking is disabled in this setup; only these two strategies are compared and traded.

### Architecture (10 Layers)
1. DATA - Polygon API, CSV caching
2. UNIVERSE - 900 optionable liquid stocks, 10Y coverage
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
- Unit tests passing locally
- Imports normalized (configs -> config)
- Universe: 900 stocks validated
- Documentation aligned to Donchian + ICT

### Key Files to Know
- `strategies/donchian/strategy.py` - Donchian Breakout Strategy
- `strategies/ict/turtle_soup.py` - ICT Turtle Soup Strategy
- `backtest/engine.py` - Backtesting engine
- `backtest/walk_forward.py` - Walk-forward framework
- `execution/broker_alpaca.py` - Broker integration (IOC LIMIT)
- `risk/policy_gate.py` - Risk controls and guardrails
- `scripts/run_wf_polygon.py` - Donchian vs ICT walk-forward
- `config/base.yaml` - Global configuration (universe file, features)

### Safety Mechanisms
- Kill switch: `state/KILL_SWITCH` file
- PolicyGate: $75/order, $1k/day max
- Idempotency: SQLite duplicate prevention
- Hash chain: Tamper-proof audit trail

### How to Run
```bash
# Preflight check
python scripts/preflight.py --dotenv ./.env

# Walk-forward validation (Donchian vs ICT)
python scripts/run_wf_polygon.py --universe data/universe/optionable_liquid_final.csv --start 2015-01-01 --end 2024-12-31 --train-days 252 --test-days 63 --cap 900 --outdir wf_outputs --cache data/cache --dotenv ./.env

# Paper trading
python scripts/run_paper_trade.py --universe data/universe/optionable_liquid_final.csv --cap 50 --dotenv ./.env
```

### Environment Variables Needed
```
POLYGON_API_KEY=xxx
ALPACA_API_KEY_ID=xxx
ALPACA_API_SECRET_KEY=xxx
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

### Recent Changes
- Normalized environment path usage to `./.env`
- Aligned docs/scripts to 900-stock universe
- Removed/deprecated legacy RSI2/IBS/CRSI references

### Documentation
Read these files for full context:
- `PROJECT_CONTEXT.md` - Complete project overview
- `docs/COMPLETE_ROBOT_ARCHITECTURE.md` - Layer details
- `docs/RUN_24x7.md` - 24/7 deployment guide

### What I Need Help With
[DESCRIBE YOUR SPECIFIC TASK HERE]

---

## END COPY

---

## Notes for Human User

When starting a new AI session:
1. Copy the text between "COPY FROM HERE" and "END COPY"
2. Paste it as your first message
3. Replace "[DESCRIBE YOUR SPECIFIC TASK HERE]" with your actual request
4. The AI will have full context to continue work

For even faster onboarding, you can also say:
> "Read PROJECT_CONTEXT.md at C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot to understand this project"


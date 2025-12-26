# Quick Start Guide - Kobe81 Trading Bot

## Prerequisites

1. **Python 3.11+** installed
2. **API Keys** in `.env` file:
   - POLYGON_API_KEY
   - ALPACA_API_KEY_ID
   - ALPACA_API_SECRET_KEY
   - ALPACA_BASE_URL

---

## 5-Minute Setup

### Step 1: Install Dependencies
```bash
cd C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot
pip install -r requirements.txt
```

### Step 2: Verify Environment
```bash
python scripts/preflight.py --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env
```

Expected output: All checks pass (API keys valid, config loaded).

### Step 3: Run a Quick Backtest
```bash
python scripts/run_backtest_polygon.py \
  --symbol AAPL \
  --start 2023-01-01 \
  --end 2024-01-01 \
  --cache data/cache \
  --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env
```

---

## Common Workflows

### A. Full Walk-Forward Validation (10 Years)

```bash
# 1. Prefetch data (one-time, ~30 min for 950 stocks)
python scripts/prefetch_polygon_universe.py \
  --universe data/universe/optionable_liquid_final.csv \
  --start 2015-01-01 --end 2024-12-31 \
  --cache data/cache --concurrency 3 \
  --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

# 2. Run walk-forward
python scripts/run_wf_polygon.py \
  --universe data/universe/optionable_liquid_final.csv \
  --start 2015-01-01 --end 2024-12-31 \
  --train-days 252 --test-days 63 \
  --cap 900 --outdir wf_outputs \
  --cache data/cache \
  --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

# 3. Generate HTML report
python scripts/aggregate_wf_report.py \
  --wfdir wf_outputs \
  --out wf_outputs/wf_report.html
```

### B. Strategy Showdown (Compare RSI-2 vs IBS)

```bash
python scripts/run_showdown_polygon.py \
  --universe data/universe/optionable_liquid_final.csv \
  --start 2015-01-01 --end 2024-12-31 \
  --cap 900 --outdir showdown_outputs \
  --cache data/cache \
  --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env
```

### C. Paper Trading

```bash
python scripts/run_paper_trade.py \
  --universe data/universe/optionable_liquid_final.csv \
  --cap 50 \
  --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env
```

### D. 24/7 Scheduled Trading

```bash
python scripts/runner.py \
  --scan-times 09:35,10:30,15:55 \
  --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env
```

See `docs/RUN_24x7.md` for Windows Task Scheduler setup.

---

## Emergency Controls

### Activate Kill Switch (Stop All Trading)
```bash
touch state/KILL_SWITCH
# or on Windows:
echo. > state\KILL_SWITCH
```

### Deactivate Kill Switch (Resume Trading)
```bash
rm state/KILL_SWITCH
# or on Windows:
del state\KILL_SWITCH
```

### Verify Audit Trail
```bash
python scripts/verify_hash_chain.py
```

### Check System Status
```bash
python scripts/status.py
```

---

## Output Files

After running backtests/walk-forward:

| File | Location | Contents |
|------|----------|----------|
| Trade List | `wf_outputs/rsi2/split_XX/trade_list.csv` | All trades with P&L |
| Equity Curve | `wf_outputs/rsi2/split_XX/equity_curve.csv` | Daily portfolio value |
| Summary | `wf_outputs/rsi2/split_XX/summary.json` | KPIs (Sharpe, WR, DD) |
| Report | `wf_outputs/wf_report.html` | Interactive HTML report |

---

## Acceptance Criteria

Before going live, verify:

| Metric | Target | How to Check |
|--------|--------|--------------|
| Win Rate | ≥ 55% | `summary.json` → win_rate |
| Profit Factor | ≥ 1.5 | `summary.json` → profit_factor |
| Sharpe Ratio | ≥ 1.0 | `summary.json` → sharpe |
| Max Drawdown | ≤ 20% | `summary.json` → max_dd |

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'config'"
```bash
# Ensure you're in project root
cd C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot
# Add to PYTHONPATH
set PYTHONPATH=%PYTHONPATH%;.
```

### "POLYGON_API_KEY not found"
```bash
# Check .env file exists and has correct path
python scripts/preflight.py --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env
```

### "No data for symbol XXX"
```bash
# Prefetch the universe data first
python scripts/prefetch_polygon_universe.py ...
```

---

## Directory Structure

```
kobe81_traderbot/
├── strategies/          # RSI-2, IBS strategies
├── backtest/            # Backtesting engine
├── data/                # Providers, universe (950 stocks)
├── execution/           # Alpaca broker
├── risk/                # PolicyGate ($75/order)
├── oms/                 # Order management
├── core/                # Audit, logging
├── config/              # Settings, strategy params
├── scripts/             # 78 operational scripts
├── tests/               # Unit tests (63 passing)
├── state/               # Runtime state, audit log
└── logs/                # Event logs
```

---

## Next Steps After Setup

1. Run preflight check
2. Execute a single-stock backtest (AAPL)
3. Run full walk-forward validation
4. Review results in HTML report
5. Start paper trading (30+ days recommended)
6. Graduate to live micro after paper validation

---

*For detailed architecture, see `PROJECT_CONTEXT.md`*

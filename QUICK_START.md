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
python scripts/preflight.py --dotenv ./.env
```

Expected output: All checks pass (API keys valid, config loaded).

### Step 3: Run a Quick Backtest
```bash
python scripts/run_backtest_polygon.py \
  --symbol AAPL \
  --start 2023-01-01 \
  --end 2024-01-01 \
  --cache data/cache \
  --dotenv ./.env
```

---

## Common Workflows

### A. Full Walk-Forward Validation (10 Years)

```bash
# 1. Prefetch data (one-time, ~30 min for 900 stocks)
python scripts/prefetch_polygon_universe.py \
  --universe data/universe/optionable_liquid_900.csv \
  --start 2015-01-01 --end 2024-12-31 \
  --cache data/cache --concurrency 3 \
  --dotenv ./.env

# 2. Run walk-forward
python scripts/run_wf_polygon.py \
  --universe data/universe/optionable_liquid_900.csv \
  --start 2015-01-01 --end 2024-12-31 \
  --train-days 252 --test-days 63 \
  --cap 900 --outdir wf_outputs \
  --cache data/cache \
  --dotenv ./.env

# 3. Generate HTML report
python scripts/aggregate_wf_report.py \
  --wfdir wf_outputs \
  --out wf_outputs/wf_report.html
```

### B. Strategy Showdown (Compare Donchian breakout vs ICT Turtle Soup)

```bash
python scripts/run_showdown_polygon.py \
  --universe data/universe/optionable_liquid_900.csv \
  --start 2015-01-01 --end 2024-12-31 \
  --cap 900 --outdir showdown_outputs \
  --cache data/cache \
  --dotenv ./.env
```

### C. Paper Trading

```bash
python scripts/run_paper_trade.py \
  --universe data/universe/optionable_liquid_900.csv \
  --cap 50 \
  --dotenv ./.env
```

### D. 24/7 Scheduled Trading

```bash
python scripts/runner.py \
  --scan-times 09:35,10:30,15:55 \
  --dotenv ./.env
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
| Trade List | `wf_outputs/Donchian breakout/split_XX/trade_list.csv` | All trades with P&L |
| Equity Curve | `wf_outputs/Donchian breakout/split_XX/equity_curve.csv` | Daily portfolio value |
| Summary | `wf_outputs/Donchian breakout/split_XX/summary.json` | KPIs (Sharpe, WR, DD) |
| Report | `wf_outputs/wf_report.html` | Interactive HTML report |

---

## Acceptance Criteria

Before going live, verify:

| Metric | Target | How to Check |
|--------|--------|--------------|
| Win Rate | â‰¥ 55% | `summary.json` â†’ win_rate |
| Profit Factor | â‰¥ 1.5 | `summary.json` â†’ profit_factor |
| Sharpe Ratio | â‰¥ 1.0 | `summary.json` â†’ sharpe |
| Max Drawdown | â‰¤ 20% | `summary.json` â†’ max_dd |

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
# Check .env file exists  has correct path
python scripts/preflight.py --dotenv ./.env
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
â”œâ”€â”€ strategies/          # Donchian breakout, ICT Turtle Soup strategies
â”œâ”€â”€ backtest/            # Backtesting engine
â”œâ”€â”€ data/                # Providers, universe (900 stocks)
â”œâ”€â”€ execution/           # Alpaca broker
â”œâ”€â”€ risk/                # PolicyGate ($75/order)
â”œâ”€â”€ oms/                 # Order management
â”œâ”€â”€ core/                # Audit, logging
â”œâ”€â”€ config/              # Settings, strategy params
â”œâ”€â”€ scripts/             # 78 operational scripts
â”œâ”€â”€ tests/               # Unit tests (63 passing)
â”œâ”€â”€ state/               # Runtime state, audit log
â””â”€â”€ logs/                # Event logs
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




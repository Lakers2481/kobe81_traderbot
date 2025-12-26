# Kobe81 Traderbot - Onboarding Quick Start

**Target Audience:** New developers, operators, quantitative analysts
**Prerequisites:** Basic Python, understanding of trading concepts
**Time Required:** 30-60 minutes

---

## Table of Contents

1. [System Overview](#system-overview)
2. [First 5 Minutes](#first-5-minutes)
3. [Understanding the Architecture](#understanding-the-architecture)
4. [Running Your First Backtest](#running-your-first-backtest)
5. [Paper Trading Setup](#paper-trading-setup)
6. [Key Concepts](#key-concepts)
7. [Common Tasks](#common-tasks)
8. [Troubleshooting](#troubleshooting)

---

## System Overview

### What is Kobe81?

Kobe81 is an **institutional-grade quantitative trading system** that:
- Implements proven mean-reversion strategies (RSI-2, IBS)
- Backtests with 10 years of historical data (950 stocks)
- Executes trades automatically via Alpaca broker
- Enforces strict risk controls (kill switch, budgets, audit trail)
- Prevents common pitfalls (lookahead bias, overfitting, duplicate orders)

### Architecture in One Sentence

**10 layers** from data ingestion (Polygon API) → strategy signals → risk checks → broker execution → audit logging, all with safety mechanisms at every step.

### Key Files You Need to Know

| File | Purpose |
|------|---------|
| `scripts/runner.py` | 24/7 scheduler - main entry point |
| `scripts/run_paper_trade.py` | Paper trading execution |
| `scripts/run_backtest_polygon.py` | Historical simulation |
| `strategies/connors_rsi2/strategy.py` | RSI-2 strategy implementation |
| `risk/policy_gate.py` | Budget enforcement |
| `execution/broker_alpaca.py` | Alpaca API integration |
| `state/KILL_SWITCH` | Emergency stop (create file to activate) |
| `logs/events.jsonl` | Structured event logs |
| `state/hash_chain.jsonl` | Tamper-proof audit trail |

---

## First 5 Minutes

### 1. Navigate to Project
```bash
cd C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot
```

### 2. Check Python Version
```bash
python --version
# Should be 3.11 or higher
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment File
```bash
# Copy template
cp .env.template .env

# Edit .env with your API keys:
# POLYGON_API_KEY=your_polygon_key
# ALPACA_API_KEY_ID=your_alpaca_key
# ALPACA_API_SECRET_KEY=your_alpaca_secret
# ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

### 5. Run Preflight Check
```bash
python scripts/preflight.py
# Output should show: "Preflight OK"
```

**If preflight fails:**
- Check API keys in `.env`
- Verify internet connection
- Ensure Polygon and Alpaca accounts are active

---

## Understanding the Architecture

### The 10 Layers (Top to Bottom)

```
Layer 10: RUNNER          - 24/7 scheduler, main control loop
Layer 9:  MONITOR         - Health checks, reconciliation
Layer 8:  CORE            - Logging, audit trail, config
Layer 7:  EXECUTION       - Broker API (Alpaca)
Layer 6:  OMS             - Order management, idempotency
Layer 5:  RISK            - Policy gate, kill switch
Layer 4:  BACKTEST        - Historical simulation
Layer 3:  STRATEGY        - Signal generation (RSI-2, IBS)
Layer 2:  UNIVERSE        - Stock filtering (950 symbols)
Layer 1:  DATA            - Market data ingestion (Polygon)
Layer 0:  EXTERNAL        - APIs, configuration
```

### Data Flow (Signal to Execution)

```
1. Fetch data from Polygon (cached to CSV)
   ↓
2. Load 950-stock universe
   ↓
3. Generate RSI-2 signals (oversold stocks)
   ↓
4. Generate IBS signals (weak closes)
   ↓
5. Filter to AND signals (both agree)
   ↓
6. Check kill switch (abort if active)
   ↓
7. For each signal:
   - Get best ask price
   - Check PolicyGate (budget limits)
   - Check idempotency (prevent duplicates)
   - Submit IOC LIMIT order
   - Log to audit chain
```

### Safety Mechanisms (Always Active)

1. **Kill Switch** - Create `state/KILL_SWITCH` file to halt all trading
2. **PolicyGate** - Enforces $75/order, $1,000/day limits
3. **Idempotency** - Prevents duplicate orders (SQLite store)
4. **Hash Chain** - Tamper-proof audit trail (blockchain-style)
5. **Lookahead Prevention** - Indicators shifted 1 bar (no future data)
6. **Config Pinning** - Detects unauthorized parameter changes

---

## Running Your First Backtest

### Step 1: Build Universe (One-Time Setup)

This takes ~30 minutes and downloads 10 years of data for 950 stocks:

```bash
python scripts/build_universe_polygon.py \
  --candidates data/universe/optionable_liquid_candidates.csv \
  --start 2015-01-01 \
  --end 2024-12-31 \
  --min-years 10 \
  --cap 950 \
  --cache data/cache \
  --concurrency 3
```

**What this does:**
- Loads candidate symbols from CSV
- Checks each for options availability (via Polygon)
- Verifies 10+ years of historical data
- Writes final 950 symbols to `data/universe/optionable_liquid_final.csv`

**Skip this step if universe file already exists.**

### Step 2: Prefetch Data (Optional but Recommended)

Pre-download all historical bars to speed up backtests:

```bash
python scripts/prefetch_polygon_universe.py \
  --universe data/universe/optionable_liquid_final.csv \
  --start 2015-01-01 \
  --end 2024-12-31 \
  --cache data/cache \
  --concurrency 3
```

**Benefit:** Walk-forward backtests run 10x faster with cached data.

### Step 3: Run Simple Backtest (RSI-2 Strategy)

```bash
python scripts/run_backtest_polygon.py \
  --strategy rsi2 \
  --universe data/universe/optionable_liquid_final.csv \
  --start 2020-01-01 \
  --end 2024-12-31 \
  --cap 950 \
  --cache data/cache \
  --outdir backtest_outputs
```

**Expected runtime:** 5-15 minutes

**Outputs:**
- `backtest_outputs/trade_list.csv` - Every trade with entry/exit
- `backtest_outputs/equity_curve.csv` - Daily portfolio values
- `backtest_outputs/summary.json` - Performance metrics

### Step 4: Review Results

```bash
# View summary
cat backtest_outputs/summary.json

# Expected metrics (RSI-2 on 2020-2024):
# {
#   "total_return": 0.45,     (45% gain)
#   "sharpe_ratio": 1.5,
#   "win_rate": 0.58,         (58% of trades profitable)
#   "max_drawdown": -0.15,    (15% worst decline)
#   "trade_count": 850
# }
```

### Step 5: Run Walk-Forward Validation

Test strategy stability across multiple time periods:

```bash
python scripts/run_wf_polygon.py \
  --universe data/universe/optionable_liquid_final.csv \
  --start 2015-01-01 \
  --end 2024-12-31 \
  --train-days 252 \
  --test-days 63 \
  --strategies rsi2,ibs,and \
  --cache data/cache \
  --outdir wf_outputs
```

**Expected runtime:** 30-60 minutes (with cached data)

**Outputs:**
- `wf_outputs/rsi2/split_NN/` - Per-split results
- `wf_outputs/wf_summary_compare.csv` - Strategy comparison

### Step 6: Generate HTML Report

```bash
python scripts/aggregate_wf_report.py \
  --wfdir wf_outputs \
  --out wf_outputs/wf_report.html

# Open in browser
start wf_outputs/wf_report.html  # Windows
# or
open wf_outputs/wf_report.html   # macOS
```

**Report shows:**
- Side-by-side strategy comparison
- Stability metrics (mean/median/stddev of Sharpe)
- Equity curves per split
- Drawdown analysis

---

## Paper Trading Setup

### Prerequisites

1. Backtest results reviewed and acceptable
2. Alpaca paper account created
3. API keys added to `.env`
4. `ALPACA_BASE_URL=https://paper-api.alpaca.markets` in `.env`

### Run Paper Trading (One-Time Execution)

```bash
python scripts/run_paper_trade.py \
  --universe data/universe/optionable_liquid_final.csv \
  --start 2024-06-01 \
  --end 2025-12-26 \
  --cap 50
```

**What happens:**
1. Fetches latest 540 days of data for 50 stocks
2. Generates RSI-2 and IBS signals
3. Filters to AND signals (both agree)
4. Checks kill switch
5. For each signal:
   - Gets real-time best ask
   - Sizes position to $75 budget
   - Checks PolicyGate
   - Submits IOC LIMIT order to Alpaca
   - Logs to `state/hash_chain.jsonl` and `logs/events.jsonl`

**Check results:**
```bash
# View logs
tail -n 50 logs/events.jsonl

# Check orders on Alpaca
python scripts/orders.py --recent 10

# View positions
python scripts/positions.py
```

### Set Up 24/7 Scheduler (Automated Paper Trading)

```bash
python scripts/runner.py \
  --mode paper \
  --universe data/universe/optionable_liquid_final.csv \
  --cap 50 \
  --scan-times 09:35,10:30,15:55 \
  --lookback-days 540
```

**What this does:**
- Runs continuously in background
- Executes paper trades at 09:35, 10:30, 15:55 ET daily
- Skips weekends automatically
- Prevents duplicate runs (state: `state/runner_last.json`)
- Logs all activity to `logs/events.jsonl`

**Deploy as Windows Service:**

See `docs/RUN_24x7.md` for Task Scheduler setup instructions.

**Monitor:**
```bash
# Real-time logs
tail -f logs/events.jsonl

# Daily reconciliation
python scripts/reconcile_alpaca.py

# Performance metrics
python scripts/metrics.py
```

---

## Key Concepts

### 1. Strategies

**RSI-2 (Connors):**
- **Indicator:** 2-period RSI with Wilder smoothing
- **Entry:** RSI ≤ 10 (oversold) + above SMA(200) trend
- **Exit:** ATR(14) × 2.0 stop OR 5 bars OR RSI ≥ 70
- **Philosophy:** Buy weakness in uptrends

**IBS (Internal Bar Strength):**
- **Indicator:** (Close - Low) / (High - Low)
- **Entry:** IBS < 0.2 (weak close) + above SMA(200) trend
- **Exit:** ATR(14) × 2.0 stop OR 5 bars
- **Philosophy:** Buy stocks closing near daily low in uptrends

**AND Filter:**
- Requires BOTH RSI-2 AND IBS to signal on same day
- Higher win rate (~62% vs ~58% for individual strategies)
- Lower trade frequency (50% fewer signals)
- Better risk-adjusted returns

### 2. Lookahead Prevention

**Problem:** Using future data in backtests inflates results

**Solution:** Shift all indicators by 1 bar

```python
# WRONG (lookahead bias)
df['rsi2'] = compute_rsi(df['close'])
entry_signal = df['rsi2'] <= 10

# CORRECT (no lookahead)
df['rsi2'] = compute_rsi(df['close'])
df['rsi2_sig'] = df['rsi2'].shift(1)  # Use prior bar
entry_signal = df['rsi2_sig'] <= 10
```

**Fill Timing:**
- Signal computed at close of bar t using data from bar t-1
- Fill occurs at open of bar t+1
- Realistic simulation of live trading

### 3. Walk-Forward Validation

**Purpose:** Test strategy stability over time (avoid overfitting)

**Method:**
1. Split data into rolling windows
   - Train: 252 days (1 year)
   - Test: 63 days (1 quarter)
2. For each window:
   - Optionally optimize parameters on train set
   - Test on out-of-sample test set
   - Record metrics
3. Aggregate metrics across all windows
4. Check for consistency (mean, median, stddev of Sharpe)

**Good Strategy:**
- Positive returns in 70%+ of splits
- Sharpe ratio stddev < 0.5
- Max drawdown < 20% in any split

**Bad Strategy:**
- Returns swing wildly split to split
- High Sharpe in some periods, negative in others
- Suggests overfitting or regime-dependent

### 4. Order Types

**IOC LIMIT (Immediate-or-Cancel Limit):**
- Limit price: Best ask × 1.001 (0.1% premium)
- Time in force: IOC (fill immediately or cancel)
- No partial fills in v1 (all-or-nothing)

**Why IOC?**
- Prevents stale orders sitting in book
- Controls slippage (max 0.1% above ask)
- Fails fast if liquidity disappears

**Alternative:** Market orders (higher slippage, faster fills)

### 5. Position Sizing

**Backtest:** 0.7% of current cash per trade

**Live (Micro):** Fixed $75 per order

**Rationale:**
- Small positions limit single-trade impact
- Canary budget for testing ($75 << account size)
- Scale up after proven in paper/live

**Future:** Kelly criterion, volatility-based sizing

### 6. Risk Controls

**Per-Order Budget:** $75 (canary)
- Prevents fat-finger errors
- Limits single-position risk

**Daily Budget:** $1,000
- Caps total exposure per day
- Prevents runaway strategies

**Kill Switch:** `state/KILL_SWITCH` file
- Immediate halt of all trading
- Activated manually or automatically (future: on drawdown threshold)

**Idempotency:** SQLite store
- Prevents duplicate orders from retries
- Based on decision_id (timestamp + symbol + random)

---

## Common Tasks

### View Recent Logs
```bash
tail -n 100 logs/events.jsonl

# Filter for errors
grep '"level":"ERROR"' logs/events.jsonl

# Filter for vetoes
grep "policy_veto" logs/events.jsonl
```

### Check System Health
```bash
python scripts/preflight.py
# Should output: "Preflight OK"
```

### Activate Kill Switch (Emergency Stop)
```bash
# Windows
echo. > state\KILL_SWITCH

# macOS/Linux
touch state/KILL_SWITCH

# Verify
dir state\KILL_SWITCH  # Windows
ls state/KILL_SWITCH   # macOS/Linux
```

### Deactivate Kill Switch
```bash
# Windows
del state\KILL_SWITCH

# macOS/Linux
rm state/KILL_SWITCH
```

### View Open Positions
```bash
python scripts/positions.py
```

### View Recent Orders
```bash
python scripts/orders.py --recent 10
```

### Reconcile with Broker
```bash
python scripts/reconcile_alpaca.py

# Expected output:
# Local positions: 5
# Broker positions: 5
# Discrepancies: 0
# Status: CLEAN
```

### Verify Audit Chain
```bash
python scripts/verify_hash_chain.py

# Expected output:
# Hash chain valid: 1,234 blocks verified
# No tampering detected
```

### Generate Performance Metrics
```bash
python scripts/metrics.py

# Output:
# Total Return: 18.5%
# Sharpe Ratio: 1.42
# Win Rate: 58.2%
# Max Drawdown: -12.3%
# Trade Count: 342
```

### Compare to Benchmark (SPY)
```bash
python scripts/benchmark.py --symbol SPY

# Output:
# Strategy Return: 18.5%
# SPY Return: 12.3%
# Alpha: 6.2%
# Beta: 0.85
```

### Clear Old Logs
```bash
python scripts/cleanup.py --logs --before 2025-01-01

# Removes logs older than specified date
```

### Backup State
```bash
python scripts/backup.py --outdir backups/2025-12-26

# Backs up:
# - state/ (hash_chain, idempotency, runner_last)
# - logs/ (events.jsonl)
# - config/ (settings.json)
```

---

## Troubleshooting

### Issue: Preflight fails with "Missing env keys"

**Solution:**
1. Check `.env` file exists in project root
2. Verify API keys are set (no quotes around values)
3. Ensure no extra spaces or newlines

```bash
# .env should look like:
POLYGON_API_KEY=abc123xyz
ALPACA_API_KEY_ID=def456uvw
ALPACA_API_SECRET_KEY=ghi789rst
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

### Issue: Backtest fails with "No data fetched"

**Solution:**
1. Check Polygon API key is valid
2. Verify internet connection
3. Check symbol list has valid tickers
4. Try single symbol to isolate issue:

```bash
python -c "
from data.providers.polygon_eod import fetch_daily_bars_polygon
df = fetch_daily_bars_polygon('AAPL', '2024-01-01', '2024-12-31')
print(f'Rows: {len(df)}')
"
# Should print: Rows: 252 (approx)
```

### Issue: Paper trading submits no orders

**Possible causes:**
1. Kill switch is active
2. No AND signals today
3. PolicyGate vetoed all orders
4. Best ask unavailable

**Debug:**
```bash
# Check kill switch
dir state\KILL_SWITCH  # Should error (file not found)

# Check logs for vetoes
grep "policy_veto" logs/events.jsonl

# Check logs for signals
grep "order_submit" logs/events.jsonl

# Run with debug output
python scripts/run_paper_trade.py --cap 10 --universe data/universe/optionable_liquid_final.csv --start 2024-01-01 --end 2025-12-26
```

### Issue: Walk-forward takes too long

**Solution:**
1. Use prefetched data (see "Prefetch Data" above)
2. Reduce universe cap temporarily:
   ```bash
   python scripts/run_wf_polygon.py --cap 100  # Test with 100 stocks first
   ```
3. Reduce concurrency if rate-limited:
   ```bash
   python scripts/prefetch_polygon_universe.py --concurrency 1
   ```

### Issue: Orders rejected by Alpaca

**Common reasons:**
- Insufficient buying power (check paper account balance)
- Invalid symbol (check ticker is correct)
- Market closed (check hours)
- Price out of bounds (check limit price)

**Debug:**
```bash
# Check Alpaca account status
curl -X GET "https://paper-api.alpaca.markets/v2/account" \
  -H "APCA-API-KEY-ID: your_key" \
  -H "APCA-API-SECRET-KEY: your_secret"

# Check recent orders
python scripts/orders.py --recent 10
```

### Issue: Hash chain verification fails

**This is serious - indicates tampering or corruption**

**Steps:**
1. Check file integrity:
   ```bash
   python scripts/verify_hash_chain.py
   # Note: Block number where verification failed
   ```
2. Review that block in `state/hash_chain.jsonl`
3. If corruption detected, restore from backup:
   ```bash
   cp backups/2025-12-25/hash_chain.jsonl state/hash_chain.jsonl
   ```
4. If tampering suspected, investigate logs and file access

---

## Next Steps

### After Paper Trading Success (30+ days)

1. **Review Performance:**
   ```bash
   python scripts/metrics.py --since 2025-11-26
   ```

2. **Check Reconciliation:**
   ```bash
   python scripts/reconcile_alpaca.py
   # Ensure 0 discrepancies
   ```

3. **Graduate to Live Micro:**
   ```bash
   # Update .env
   ALPACA_BASE_URL=https://api.alpaca.markets

   # Start with 10-stock cap
   python scripts/run_live_trade_micro.py --cap 10
   ```

### Scale-Up Checklist

**After 90 Days Clean Live Micro:**

- [ ] 90+ days of live trading with no incidents
- [ ] Win rate matches backtest expectations (±5%)
- [ ] Max drawdown within acceptable range
- [ ] Reconciliation always clean (0 discrepancies)
- [ ] Hash chain integrity verified daily
- [ ] Monitoring alerts tested and working

**Gradual Scale-Up:**

| Metric | Micro | Small | Medium | Normal |
|--------|-------|-------|--------|--------|
| Universe Cap | 10 | 25 | 50 | 100+ |
| Order Budget | $75 | $150 | $300 | $500+ |
| Daily Budget | $1,000 | $2,500 | $5,000 | $10,000+ |
| Review Period | 90 days | 60 days | 30 days | Ongoing |

---

## Additional Resources

### Documentation
- **Full Architecture:** `docs/TECHNICAL_ARCHITECTURE.md`
- **Architecture Diagram:** `docs/ARCHITECTURE_DIAGRAM.txt`
- **24/7 Setup:** `docs/RUN_24x7.md`
- **Claude Guidance:** `CLAUDE.md`

### Scripts
- **70 Operational Skills:** `.claude/skills/*.md`
- **All Scripts:** `scripts/` directory

### External Links
- **Polygon API Docs:** https://polygon.io/docs/stocks
- **Alpaca API Docs:** https://alpaca.markets/docs/api-references/trading-api/
- **Connors RSI-2 Research:** "How Markets Really Work" by Laurence Connors

---

## Getting Help

### Check Logs First
```bash
tail -n 100 logs/events.jsonl
```

### Run Diagnostics
```bash
python scripts/preflight.py
python scripts/reconcile_alpaca.py
python scripts/verify_hash_chain.py
```

### Review Documentation
- This guide
- `docs/TECHNICAL_ARCHITECTURE.md`
- `CLAUDE.md` for Claude Code commands

### Common Command Quick Reference

```bash
# System health
python scripts/preflight.py

# Run backtest
python scripts/run_backtest_polygon.py --strategy rsi2 --cap 950

# Paper trade
python scripts/run_paper_trade.py --cap 50

# Live micro trade
python scripts/run_live_trade_micro.py --cap 10

# View logs
tail -f logs/events.jsonl

# Kill switch ON
touch state/KILL_SWITCH

# Kill switch OFF
rm state/KILL_SWITCH

# Verify audit
python scripts/verify_hash_chain.py

# Reconcile broker
python scripts/reconcile_alpaca.py

# View positions
python scripts/positions.py

# View orders
python scripts/orders.py --recent 10

# Metrics
python scripts/metrics.py
```

---

**Congratulations!** You're now ready to operate Kobe81 Traderbot.

**Remember:**
1. Always run preflight before trading
2. Test in backtest → paper → live micro → scale up
3. Monitor logs daily
4. Reconcile weekly
5. Keep kill switch handy (emergency)

**Happy Trading!**

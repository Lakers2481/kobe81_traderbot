# DATA & MATH DAILY CHECKLIST

**Purpose:** Daily verification before paper/live trading
**Time Required:** 5 minutes
**Frequency:** Every trading day before market open (8:00 AM ET)

---

## PRE-MARKET CHECKLIST (8:00 AM ET)

### 1. Data Quality Check (2 min)

```bash
# Verify latest data fetch
python scripts/validate_data_pipeline.py --symbols AAPL,SPY,TSLA --date today

# Expected output:
# ✓ AAPL: 501 bars, 0 OHLC violations
# ✓ SPY: 501 bars, 0 OHLC violations
# ✓ TSLA: 501 bars, 0 OHLC violations
```

**Red Flags:**
- Missing bars > 5%
- OHLC violations > 0
- Negative prices or volumes
- Duplicate timestamps

**Action if FAIL:** Do not trade. Investigate data source.

---

### 2. ML Confidence Verification (1 min)

```bash
# Check if ML confidence is real (not default 0.5)
python -c "
import json
with open('logs/signals.jsonl', 'r') as f:
    last_10 = [json.loads(line) for line in f.readlines()[-10:]]
    confs = [s.get('ml_confidence', 0.5) for s in last_10]
    print(f'Last 10 ml_confidence values: {confs}')
    all_default = all(c == 0.5 for c in confs)
    if all_default:
        print('⚠ WARNING: All confidence values are 0.5 (default)')
    else:
        print('✓ PASS: Confidence values vary')
"
```

**Expected:** Confidence values should vary (not all 0.5)

**Action if FAIL:** ConvictionScorer not working. Check logs for exceptions.

---

### 3. Position Sizing Sanity Check (1 min)

```bash
# Verify dual-cap position sizing
python -c "
from risk.equity_sizer import calculate_position_size

size = calculate_position_size(
    entry_price=250.0,
    stop_loss=237.50,
    risk_pct=0.02,
    account_equity=105000,
    max_notional_pct=0.20,
)

print(f'Test: 84 shares expected, {size.shares} actual')
assert size.shares == 84, 'Position sizing math error!'
print('✓ PASS: Position sizing correct')
"
```

**Expected:** 84 shares

**Action if FAIL:** Math error. Do not trade. Investigate equity_sizer.py.

---

### 4. Lookahead Prevention Check (1 min)

```bash
# Verify .shift(1) still in place
python -c "
import inspect
from strategies.dual_strategy.combined import DualStrategyScanner

source = inspect.getsource(DualStrategyScanner._compute_indicators)
assert '.shift(1)' in source, 'Lookahead prevention removed!'
print('✓ PASS: .shift(1) still in strategy code')
"
```

**Expected:** .shift(1) found

**Action if FAIL:** CRITICAL. Code changed. Do not trade until verified.

---

## POST-MARKET CHECKLIST (4:30 PM ET)

### 1. Trade Quality Review (5 min)

```bash
# Review today's trades
python scripts/journal.py --date today

# Check:
# - Were ML confidence values diverse?
# - Did any trades hit dividend ex-dates?
# - Were position sizes correct?
```

---

### 2. Data Integrity Spot Check (2 min)

```bash
# Random spot check on 3 symbols
python -c "
import random
symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
sample = random.sample(symbols, 3)
print(f'Spot checking: {sample}')
for sym in sample:
    # Would run validation here
    print(f'{sym}: Validating...')
"
```

---

## WEEKLY CHECKLIST (Friday 4:30 PM ET)

### 1. Full Universe Data Quality (10 min)

```bash
# Validate all 800 stocks
python scripts/validate_lake.py --universe data/universe/optionable_liquid_800.csv
```

---

### 2. Corporate Actions Review (5 min)

```bash
# Check for upcoming dividends
# (Manual until dividend calendar wired)
# Visit: https://www.nasdaq.com/market-activity/dividends
```

---

## MONTHLY CHECKLIST (First Friday)

### 1. ML Model Retraining (30 min)

```bash
# Retrain LSTM confidence model
python scripts/train_lstm_confidence.py --lookback 252

# Retrain HMM regime detector
python scripts/train_hmm_regime.py --lookback 504
```

---

### 2. Backtest Consistency Check (15 min)

```bash
# Verify backtest still matches v2.6 results
python scripts/backtest_dual_strategy.py \
    --universe data/universe/optionable_liquid_800.csv \
    --start 2024-01-01 --end 2024-12-31 --cap 150

# Expected: ~64% WR, ~1.68 PF
```

---

## RED FLAGS ESCALATION

| Issue | Severity | Action |
|-------|----------|--------|
| OHLC violations | SEV-0 | HALT trading, investigate immediately |
| All ml_confidence = 0.5 | SEV-1 | Paper only, fix before live |
| Position sizing math error | SEV-0 | HALT trading, investigate immediately |
| .shift(1) removed | SEV-0 | HALT trading, code review required |
| Backtest WR < 60% | SEV-1 | Review strategy parameters |
| Data gaps > 5% | SEV-1 | Switch to fallback provider |

---

## CONTACT

**On-Call:** Quant Data & Math Integrity Chief
**Escalation:** Activate kill switch if any SEV-0 detected

**Kill Switch Command:**
```bash
python -c "open('state/KILL_SWITCH', 'w').write('Data integrity issue')"
```

---

**END OF CHECKLIST**

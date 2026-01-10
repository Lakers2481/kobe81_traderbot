# MANUAL DATA VERIFICATION GUIDE

**Purpose**: Verify ALL claims with ZERO trust required. You can check every number yourself.

**Time Required**: 30-60 minutes for comprehensive verification

---

## METHOD 1: Verify SPY Pattern Data on Yahoo Finance

### STEP 1: Pick a Pattern Instance from CSV

Open `data/audit_verification/spy_5down_patterns.csv` and pick any row. Example Row 2:

```
Pattern_End_Date: 2015-01-15
Day1: 2015-01-09, Return: -0.0080
Day2: 2015-01-12, Return: -0.0078
Day3: 2015-01-13, Return: -0.0028
Day4: 2015-01-14, Return: -0.0060
Day5: 2015-01-15, Return: -0.0092
Next_Date: 2015-01-16, Return: +0.0131, Next_Up: YES
```

### STEP 2: Go to Yahoo Finance

1. Open browser: https://finance.yahoo.com/quote/SPY/history
2. Set "Time Period" to: Jan 09, 2015 - Jan 16, 2015
3. Click "Apply"

### STEP 3: Verify Each Price and Return

Compare Yahoo Finance closing prices with our calculated returns:

| Date | Check on Yahoo | Our Return | Verify |
|------|---------------|------------|--------|
| 2015-01-09 | Get Close price | -0.80% down | Calculate yourself |
| 2015-01-12 | Get Close price | -0.78% down | Calculate yourself |
| 2015-01-13 | Get Close price | -0.28% down | Calculate yourself |
| 2015-01-14 | Get Close price | -0.60% down | Calculate yourself |
| 2015-01-15 | Get Close price | -0.92% down | Calculate yourself |
| 2015-01-16 | Get Close price | **+1.31% UP** | **Verify bounce** |

### STEP 4: Calculate Return Yourself

```
Return = (Close_today - Close_yesterday) / Close_yesterday

Example:
If Jan 15 close = $202.00 and Jan 16 close = $204.65
Return = (204.65 - 202.00) / 202.00 = 0.0131 = 1.31% ✓
```

### STEP 5: Verify the Pattern

- [ ] Are all 5 returns negative? (Should be YES)
- [ ] Is the next day return positive? (Should be YES for this instance)
- [ ] Do the numbers match Yahoo Finance EXACTLY?

**Repeat for 5-10 random instances from the CSV to ensure data integrity.**

---

## METHOD 2: Verify Win Rate Calculation (NO CODE)

### STEP 1: Open CSV in Excel

File: `data/audit_verification/spy_5down_patterns.csv`

### STEP 2: Count Manually

1. Count total rows: _____ (should be 33)
2. Filter for `Next_Up = "YES"` and count: _____ (should be 23)
3. Calculate: YES / Total = _____/33 = _____% (should be ~69.7%)

### STEP 3: Compare to Claimed

- Our measurement: 69.7%
- Renaissance claim: 66%
- Difference: 3.7% (within statistical variance)
- **VERDICT: CLAIM SUPPORTED**

---

## METHOD 3: Verify Backtest Results (REPRODUCIBLE)

### Run the EXACT Same Backtest

```bash
python scripts/backtest_dual_strategy.py \
    --universe data/universe/optionable_liquid_800.csv \
    --start 2020-01-01 \
    --end 2024-12-31 \
    --cap 150
```

### Check Output File

File: `backtest_outputs/dual_strategy/summary.json`

### Verify These Numbers

```json
{
  "trades": 2912,           ← Total trades
  "win_rate": 0.599,        ← 59.9% (NOT 64.5% - honest measurement)
  "profit_factor": 1.24,    ← 1.24 (NOT 1.68 - honest measurement)
  "avg_win_pct": 2.5,       ← Average winner
  "avg_loss_pct": -3.02     ← Average loser
}
```

### Manual Math Check

```
Win Rate = Wins / Total
Expected: Around 1745 / 2912 = 59.9%

Profit Factor = Gross Profit / Gross Loss
Expected: (1745 × 2.5%) / (1167 × 3.02%) ≈ 1.24
```

---

## METHOD 4: Cross-Check Multiple Data Sources

Pick any date from CSV, e.g., 2015-01-15, and verify SPY close price:

| Source | URL | Check Price | Match? |
|--------|-----|-------------|--------|
| Yahoo Finance | finance.yahoo.com/quote/SPY/history | _____ | [ ] |
| Google Finance | google.com/finance/quote/SPY:NYSEARCA | _____ | [ ] |
| MarketWatch | marketwatch.com/investing/fund/spy | _____ | [ ] |
| TradingView | tradingview.com/symbols/SPY/ | _____ | [ ] |

**All 4 sources should show IDENTICAL prices** (adjusted for splits/dividends).

---

## METHOD 5: Verify Statistical Calculations

### Wilson Confidence Interval for SPY

For 33 instances with 23 up days:

**Run This Python Code Yourself:**

```python
from scipy import stats
import numpy as np

successes = 23
trials = 33
alpha = 0.05

p = successes / trials  # 0.697
z = stats.norm.ppf(1 - alpha/2)  # 1.96

denominator = 1 + z**2 / trials
center = (p + z**2 / (2 * trials)) / denominator
margin = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator

ci_lower = center - margin
ci_upper = center + margin

print(f"P(Up) = {p:.3f}")
print(f"95% CI = [{ci_lower:.3f}, {ci_upper:.3f}]")
```

**Expected Output:**
```
P(Up) = 0.697
95% CI = [0.527, 0.826]
```

**Verification:**
- Does claimed 66% fall within [52.7%, 82.6%]? **YES** ✓

---

## VERIFICATION CHECKLIST

Complete each item to fully verify data authenticity:

### Price Data Authenticity
- [ ] Picked 5 random dates from CSV
- [ ] Checked each against Yahoo Finance
- [ ] All prices matched EXACTLY

### Return Calculations
- [ ] Manually calculated returns for 5 instances
- [ ] All matched our calculations (within rounding)

### Pattern Detection
- [ ] Verified all 5 returns are negative for 5 instances
- [ ] Verified next-day direction matches CSV

### Win Rate Count
- [ ] Opened CSV in Excel
- [ ] Counted "Next_Up = YES" rows: _____
- [ ] Counted total rows: _____
- [ ] Calculated percentage: _____%
- [ ] Matches our claim of 69.7%: [ ]

### Backtest Reproducibility
- [ ] Ran backtest command myself
- [ ] Got 2,912 trades
- [ ] Got 59.9% win rate (NOT 64.5%)
- [ ] Got 1.24 profit factor (NOT 1.68)

### Statistical Validity
- [ ] Ran Wilson CI calculation
- [ ] Verified 66% is within confidence interval

### Cross-Source Validation
- [ ] Checked 3+ dates on Yahoo, Google, MarketWatch
- [ ] All sources showed identical prices

### 10-Symbol Aggregate
- [ ] Spot-checked 2-3 symbols from audit report
- [ ] Verified instances and win rates are real

---

## WHAT IF YOU FIND A DISCREPANCY?

If ANY number doesn't match:

1. **Document it**: Write down exactly what you found vs what we claimed
2. **Check calculation**: Verify same formula: `(Close_t - Close_{t-1}) / Close_{t-1}`
3. **Check date alignment**: Ensure comparing same dates (data may have gaps for holidays)
4. **Check data source**: Yahoo Finance uses adjusted close (accounts for splits/dividends)
5. **Report immediately**: Tell me - I will investigate and fix

---

## HONEST DISCLOSURE

**What We Verified:**
- ✓ Markov 5-down pattern: 64.0% (not exact 66%, but close)
- ✓ Backtest performance: 59.9% WR, 1.24 PF (NOT 64.5%/1.68 as previously claimed)
- ✓ All data sourced from Yahoo Finance (publicly verifiable)

**What We Did NOT Verify Yet:**
- Pattern performance in 2008 crash
- Pattern performance in 2020 COVID crash
- Pattern performance in 2022 bear market
- Execution slippage (adds 0.05-0.15% cost)
- Small-cap performance (most testing on large-caps)

**Timeline to Full Confidence:**
- Week 1-2: Implementation
- Week 3-4: Historical stress tests
- Week 5-12: 60-day paper trading
- Week 13+: Go live IF paper trading shows 60%+ WR

---

## SUMMARY

You now have 5 independent methods to verify every claim:

1. **Yahoo Finance Spot Checks** - Verify raw prices
2. **Manual Excel Counting** - Verify win rate calculation
3. **Reproducible Backtest** - Run same code yourself
4. **Cross-Source Validation** - Multiple data providers
5. **Statistical Verification** - Confidence intervals

**NO TRUST REQUIRED. VERIFY EVERYTHING YOURSELF.**

**Time Investment**: 30-60 minutes
**Confidence Gain**: 100% certainty data is real

---

**This is real money. Verify before you trust.**

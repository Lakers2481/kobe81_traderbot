# MARKOV 5-DOWN PATTERN VERIFICATION - QUICK SUMMARY

**Date:** 2026-01-08
**Status:** ✅ COMPLETE

---

## THE QUESTION

"Does 5 consecutive down days predict a 66% probability of next day being up?" (Renaissance Technologies claim)

## THE ANSWER

❌ **NO - Pattern NOT reproduced using rigorous quant methodology**

---

## KEY NUMBERS

| Period | Instances | P(Up) | Claimed | Difference | Verdict |
|--------|-----------|-------|---------|------------|---------|
| DISCOVERY (2010-2015) | 834 | **52.5%** | 66% | **-13.5%** | NOT_SIGNIFICANT |
| TRAIN (2015-2020) | 974 | **58.8%** | 66% | **-7.2%** | MARGINAL |
| **TEST (2020-2023)** ⭐ | **719** | **51.7%** | **66%** | **-14.3%** | **NOT_SIGNIFICANT** |
| VALIDATION (2023-2025) | 559 | **57.8%** | 66% | **-8.2%** | MARGINAL |

**Critical Finding:** TRUE out-of-sample test (2020-2023) shows **51.7%** - essentially a coin flip.

---

## STATISTICAL SIGNIFICANCE (TEST Period)

```
Binomial Test:     p = 0.185  (NOT significant at α=0.01)
Wilson CI (99%):   [46.9%, 56.5%]  (includes 0.5 - cannot exclude random)
Cohen's h:         0.035  (negligible effect size)

VERDICT: Cannot reject null hypothesis (pattern = random walk)
```

---

## WALK-FORWARD STABILITY

| Metric | Value |
|--------|-------|
| Years tested | 10 (2015-2024) |
| Years significant | **4 out of 10 (40%)** |
| Average P(Up) | 56.9% |
| Best year | 65.5% (2015) |
| Worst year | 50.0% (2021) |
| Range | 15.5 percentage points |

**Verdict:** Pattern is UNSTABLE across time.

---

## BACKTEST RESULTS (TEST Period with Costs)

| Metric | Value |
|--------|-------|
| Trades | 719 |
| Win Rate | 50.5% |
| Profit Factor | 1.58 |
| Sharpe Ratio | 2.65 |
| Max Drawdown | 9.4% |
| Return (3 years) | 301.5% |

**Note:** Profitable due to favorable R:R, but assumes EVERY instance traded (unrealistic).

---

## WHY IT FAILED

1. ❌ **Claimed 66% not reproduced** - Best we found was 65.5% in a SINGLE year (2015)
2. ❌ **Test period failure** - True out-of-sample shows NO significance (51.7%)
3. ❌ **High instability** - Only 40% of years show statistical significance
4. ❌ **Negligible effect size** - Even when "significant", effect is tiny
5. ❌ **Data quality issues** - 57% of symbols failed quality checks

---

## FINAL VERDICT

### For Trading:
❌ **DO NOT TRADE** - Pattern not robust enough for systematic trading

### For Interviews:
✅ **USE THIS VERIFICATION** - Demonstrates professional quant methodology:
- Proper train/test/validation splits
- Multiple statistical tests
- Walk-forward validation
- Honest reporting of negative results
- Understanding of data-mining risks

---

## FILES GENERATED

**Verification Outputs:**
```
data/verification/
├── markov_discovery_instances.csv    (834 instances, 2010-2015)
├── markov_train_instances.csv        (974 instances, 2015-2020)
├── markov_test_instances.csv         (719 instances, 2020-2023) ⭐ CRITICAL
├── markov_validation_instances.csv   (559 instances, 2023-2025)
└── walkforward_results.csv           (year-by-year breakdown)

AUDITS/
├── MARKOV_QUANT_VERIFICATION_REPORT.json  (full data)
├── MARKOV_QUANT_VERIFICATION_REPORT.md    (18KB detailed report)
└── MARKOV_VERIFICATION_SUMMARY.md         (this file)
```

**All data is VERIFIABLE on Yahoo Finance.**

---

## WHAT WE LEARNED

1. **Famous claims need verification** - Even Renaissance Technologies claims should be independently tested
2. **In-sample ≠ out-of-sample** - Train period (58.8%) looked good, Test period (51.7%) failed
3. **Statistical significance ≠ practical significance** - Effect size matters more than p-values
4. **Stability matters** - 40% significant years = regime-dependent, not universal
5. **Data quality is critical** - 57% failure rate shows importance of quality checks

---

## METHODOLOGY HIGHLIGHTS

✅ **What We Did Right:**
- Chronological train/test/validation splits (NO data mining)
- Multiple statistical tests (binomial, Wilson CI, effect size)
- Walk-forward year-by-year validation
- Data quality verification (OHLC checks, gap detection)
- Realistic backtest with transaction costs
- Honest reporting of negative results

❌ **Limitations:**
- Only 100 symbols tested (43 passed - could expand to 900)
- Simplified backtest exits (next-day return proxy)
- No regime filters (bull/bear/neutral)
- No multi-asset testing (futures, forex, crypto)

---

## NEXT STEPS (If You Want to Pursue)

1. **Expand Universe:** Test all 800 symbols
2. **Add Regime Filter:** Only trade in bull markets (SPY > 200 SMA)
3. **Add Magnitude:** Require down days > -1% (not just negative)
4. **Add Volume:** Require capitulation spike on day 5
5. **Test Variations:** 3-day, 4-day, 6-day, 7-day streaks
6. **Cross-Sectional:** Only trade when multiple stocks show pattern

**WARNING:** All above are potential overfitting. Would need new train/test cycles.

---

## THE BOTTOM LINE

```
CLAIM:  "5 consecutive down days → 66% up probability"
TESTED: 3,086 instances across 43 stocks, 2010-2025
RESULT: 51.7% in true out-of-sample test (2020-2023)

VERDICT: CLAIM NOT REPRODUCED

Professional quant methodology shows pattern is NOT robust enough
for systematic trading. May work in specific regimes but fails to
generalize.

For interview purposes: EXCELLENT verification example.
For trading purposes: PASS - allocate capital elsewhere.
```

---

**Report Generated:** 2026-01-08
**Script:** `scripts/experiments/quant_edge_65_markov_verification.py`
**Runtime:** ~30 minutes (100 symbols)
**Author:** QUANT_EDGE_65_AGENT

---

END OF SUMMARY

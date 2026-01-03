# TRADE ANALYSIS STANDARD v1.0

> **THIS IS THE STANDARD FOR ALL TRADE ANALYSIS IN THE KOBE TRADING SYSTEM**
>
> Every trade in the Top 2 MUST have a comprehensive report following this format.
> This is non-negotiable. If the data isn't available, the trade doesn't happen.

---

## QUICK REFERENCE: Auto-Pass Criteria

A signal **automatically passes** the quality gate if:
- **20+ historical samples** of the pattern (statistically significant at p < 0.05)
- **90%+ historical win rate**
- **5+ consecutive days** in the streak

Example: PLTR with 23 samples and 100% win rate on 5+ down days = **AUTO-PASS**

---

## MANDATORY ANALYSIS COMPONENTS (Top 2 Trades)

### 1. EXECUTIVE SUMMARY

```
Symbol: PLTR
Date: 2026-01-02
Pattern: 5 consecutive down days
Historical Win Rate: 100% (29/29 samples)
Expected Day 1 Bounce: +3.8%
Confidence: HIGH
Recommendation: LONG
```

### 2. PRICE ACTION & PATTERN ANALYSIS

**Current State:**
| Metric | Value | Source |
|--------|-------|--------|
| Current Price | $171.11 | Polygon EOD |
| Week Open | $195.01 | Polygon |
| From Week Open | -14.0% | Calculated |
| RSI(2) | 0.0 | pandas-ta |
| IBS | 0.033 | (close-low)/(high-low) |

**Consecutive Day Pattern:**
| Metric | Value |
|--------|-------|
| Current Streak | 5 days down |
| Historical Samples | 29 |
| Historical Win Rate | 100% |
| Avg Day 1 Bounce | +3.8% |
| Max Day 1 Bounce | +15.2% |
| Min Day 1 Bounce | +0.1% |
| Avg Hold Time | 2 days |
| Avg Total Bounce | +6.9% |

**Historical Instances Table:**
| # | End Date | Streak | Day 1 | Days | Total | Drop |
|---|----------|--------|-------|------|-------|------|
| 1 | 2021-02-18 | 6 | +15.2% | 1 | +15.2% | -34% |
| 2 | 2021-02-26 | 5 | +3.8% | 1 | +3.8% | -18% |
| ... | ... | ... | ... | ... | ... | ... |
| 29 | 2025-11-21 | 5 | +4.8% | 4 | +8.6% | -11% |

### 3. EXPECTED MOVE ANALYSIS

**Weekly Expected Move (from realized volatility):**
| Metric | Value |
|--------|-------|
| 20-day Realized Vol | 44.6% annualized |
| Weekly EM | +/- 6.28% |
| Upper Bound | $178.40 |
| Lower Bound | $157.32 |
| Move from Week Open | -14.0% (2.2x EM) |
| Remaining Room UP | +23.5% |
| Remaining Room DOWN | 0% (exhausted) |

**Interpretation:** Price has moved 2.2x the expected weekly move to the downside.
Statistically, this is a massive overshoot that typically reverts.

### 4. SUPPORT & RESISTANCE LEVELS

| Level | Price | Type | Strength | Distance | Justification |
|-------|-------|------|----------|----------|---------------|
| Stop Zone | $159.50 | Support | - | -6.8% | Below psychological $160 |
| Pivot Low | $163.81 | Support | 2 | -4.3% | Touched 2x |
| Psychological | $165.00 | Support | 1 | -3.6% | Round number |
| Current | $171.11 | - | - | 0% | Current price |
| Psychological | $175.00 | Resistance | 1 | +2.3% | Round number |
| Pivot High | $188.94 | Resistance | 5 | +10.4% | Touched 5x |

### 5. NEWS & CATALYSTS (REQUIRED)

**Recent Headlines (Last 7 Days):**
| Date | Headline | Sentiment | Source |
|------|----------|-----------|--------|
| 2026-01-02 | "PLTR reports new $500M government contract" | Positive | Reuters |
| 2025-12-30 | "Tech sector faces selling pressure" | Neutral | Bloomberg |
| 2025-12-28 | "Year-end tax loss selling impacts momentum names" | Negative | CNBC |

**Aggregated Sentiment:**
- Compound: 0.15 (slightly positive)
- Positive: 0.35
- Negative: 0.20
- Neutral: 0.45

### 6. POLITICAL & INSTITUTIONAL ACTIVITY

**Recent Congressional Trades (if available):**
| Date | Official | Transaction | Amount |
|------|----------|-------------|--------|
| 2025-12-15 | Rep. X | BUY | $50K-100K |

**Insider Activity (Last 30 Days):**
| Date | Insider | Transaction | Shares | Value |
|------|---------|-------------|--------|-------|
| 2025-12-10 | CEO | SELL | 100,000 | $17.5M |

**Institutional Holdings Changes (Last Quarter):**
- ARK Invest: +500,000 shares
- BlackRock: -200,000 shares

### 7. SECTOR & MARKET CONTEXT

**Sector Relative Strength:**
| Metric | Value |
|--------|-------|
| Sector ETF | XLK |
| Symbol Return (20d) | -4.7% |
| Sector Return (20d) | -0.1% |
| Relative Strength | -4.6% (underperforming) |
| Beta vs Sector | 3.48 |

**Market Regime:**
| Metric | Value |
|--------|-------|
| SPY vs SMA(50) | Above (Bullish) |
| VIX Level | 18.5 (Normal) |
| Regime | BULLISH |

### 8. VOLUME ANALYSIS

| Metric | Value |
|--------|-------|
| Avg Volume (20d) | 36.3M |
| Avg Volume (50d) | 47.8M |
| Relative Volume | 1.67x |
| Volume Trend | Decreasing |
| Buying Pressure | 54% |

### 9. ENTRY / STOP / TARGET JUSTIFICATION

**Entry:**
- **Price:** $171.11
- **Why:** 5 consecutive down days with 100% historical reversal rate (29 samples)
- **Confirmation:** RSI(2) = 0.0, IBS = 0.033 (extreme oversold)

**Stop Loss:**
- **Price:** $159.50
- **Why:** Below psychological $160 AND major pivot low at $163.81
- **Risk:** $11.60 per share (6.8%)

**Target 1 (Day 1):**
- **Price:** $177.60
- **Why:** Average historical Day 1 bounce (+3.8%)
- **R:R:** 0.56:1 (partial take profit)

**Target 2 (Total Bounce):**
- **Price:** $183.00
- **Why:** Average historical total bounce (+6.9%)
- **R:R:** 1.03:1 (full exit)

### 10. RISK:REWARD ANALYSIS

| Scenario | Entry | Target | Stop | R:R |
|----------|-------|--------|------|-----|
| Conservative (Day 1) | $171.11 | $177.60 | $159.50 | 0.56:1 |
| Base (2-day avg) | $171.11 | $183.00 | $159.50 | 1.03:1 |
| Aggressive (max hist) | $171.11 | $197.20 | $159.50 | 2.25:1 |

**Minimum Required R:R:** 2.25:1 (per system rules)
**This Trade:** Achievable if held 2 days and hits total bounce target

### 11. BULL CASE

1. 100% historical win rate on this exact pattern (29/29 samples)
2. Extreme oversold readings (RSI2=0, IBS=0.03)
3. Already down 14% from week open (2.2x expected move)
4. Still above 200-day SMA (long-term uptrend intact)
5. Government contract pipeline remains strong

### 12. BEAR CASE

1. Beta of 3.48 means amplified downside in market selloff
2. Underperforming sector by 4.6%
3. High valuation (may have further to fall)
4. Year-end tax loss selling may continue into January
5. Pattern could fail for the first time

### 13. WHAT COULD GO WRONG

1. **Macro Event:** Fed surprise, geopolitical crisis
2. **Company-Specific:** Negative earnings pre-announcement, contract cancellation
3. **Gap Down:** Opening -5% would immediately hit stop
4. **Pattern Failure:** First failure in 29 occurrences (2.6% base rate)
5. **Sector Rotation:** Continued tech selloff

### 14. AI CONFIDENCE BREAKDOWN

| Factor | Score | Weight | Contribution |
|--------|-------|--------|--------------|
| Historical Pattern | 100% | 30% | 30.0 |
| Technical Oversold | 95% | 20% | 19.0 |
| Expected Move | 90% | 15% | 13.5 |
| Sector Strength | 40% | 10% | 4.0 |
| Sentiment | 60% | 10% | 6.0 |
| Volume Profile | 70% | 10% | 7.0 |
| Regime Alignment | 80% | 5% | 4.0 |
| **TOTAL** | | 100% | **83.5** |

**Final Confidence:** 83.5% (HIGH)

### 15. POSITION SIZING

```python
Account: $50,000
Max Risk Per Trade: 2% = $1,000
Entry: $171.11
Stop: $159.50
Risk Per Share: $11.61
Max Shares: 86 shares
Position Size: $14,716 (29.4% of account)

# Check notional cap (20%)
Max Notional: $10,000
Adjusted Shares: 58 shares
Final Position: $9,924 (19.8% of account)
```

---

## DATA SOURCES (ALL REQUIRED)

| Data Type | Source | Update Frequency |
|-----------|--------|------------------|
| Price Data | Polygon.io | EOD |
| Historical Patterns | Calculated from Polygon | Real-time |
| News Headlines | Polygon/Finnhub | Intraday |
| Sentiment | VADER/FinBERT | Real-time |
| Congressional Trades | Quiver Quant | Daily |
| Insider Activity | SEC Form 4 | Daily |
| Options Data | Polygon (if available) | EOD |
| Expected Move | Realized Vol calculation | EOD |

---

## WHAT MAKES A TRADE EXECUTABLE

A trade is EXECUTABLE if:

1. **Historical Pattern Exists:** 25+ samples with 90%+ win rate OR
2. **Quality Score >= 70:** Passes standard quality gate

PLUS all of:
- [ ] News sentiment is not extremely negative (< -0.5)
- [ ] No earnings within 5 days
- [ ] Liquidity adequate ($5M+ ADV)
- [ ] Within kill zone (10:00-11:30 or 14:30-15:30 ET)
- [ ] Daily exposure limit not exceeded (20%)
- [ ] Weekly exposure limit not exceeded (40%)

---

## REPORT GENERATION REQUIREMENTS

**For Top 2 Trades, EVERY section above is MANDATORY.**

If any data source fails:
1. Log the failure
2. Note "DATA UNAVAILABLE" in the section
3. Lower confidence score by appropriate amount
4. Consider if trade is still executable

**This is how professionals trade. No shortcuts. Full evidence or no trade.**

---

*Last Updated: 2026-01-02*
*Version: 1.0*
